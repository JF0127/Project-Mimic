import torch

import learning.nets.net_builder as net_builder
import learning.ppo_model as ppo_model
import util.torch_util as torch_util

class AMPModel(ppo_model.PPOModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        return
    
    def eval_disc(self, disc_obs):
        h = self._disc_layers(disc_obs)
        val = self._disc_logits(h)
        return val

    def get_disc_logit_weights(self):
        return torch.flatten(self._disc_logits.weight)
    
    def get_disc_weights(self):
        weights = []
        for m in self._disc_layers.modules():
            if hasattr(m, "weight"):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self._disc_logits.weight))
        return weights

    def _build_nets(self, config, env):
        self._enable_rma = getattr(env, "_enable_rma", False)
        self._enable_se = getattr(env, "_enable_se", False)
        super()._build_nets(config, env)
        self._build_disc(config, env)
        return

    def _build_disc(self, config, env):
        init_output_scale = 1.0
        net_name = config["disc_net"]

        input_dict = self._build_disc_input_dict(env)
        self._disc_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                                 activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(self._disc_layers)
        self._disc_logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(self._disc_logits.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._disc_logits.bias)
        return

    def _build_disc_input_dict(self, env):
        obs_space = env.get_disc_obs_space()
        input_dict = {"disc_obs": obs_space}
        return input_dict

    def eval_actor(self, obs, use_hist=False):
        if self._enable_rma:
            # 1. 严格按照 env 里拼装的顺序进行切片
            prop_obs = obs[..., :self._prop_dim]
            priv_obs = obs[..., self._prop_dim: self._prop_dim + self._priv_dim]
            hist_obs = obs[..., self._prop_dim + self._priv_dim:]

            # 2. 分别算出两个空间的隐向量
            priv_latent = self._priv_encoder(priv_obs)
            hist_latent = self._hist_encoder(hist_obs)

            # 3. 存下来，供 PPO Agent 在反向传播时算 RMA MSE Loss
            self.current_priv_latent = priv_latent
            self.current_hist_latent = hist_latent

            # 4. 动态选择：训练时用特权，测试/实机部署时只能用历史
            if self.training:
                latent = hist_latent if use_hist else priv_latent
            else:
                latent = hist_latent

            # 5. 拼装真正送入骨干网络的特征
            actor_input = torch.cat([prop_obs, latent], dim=-1)
        # elif self._enable_se:
        #     # 1. 维度切片
        #     prop_obs = obs[..., :self._prop_dim]
        #     priv_obs = obs[..., self._prop_dim: self._prop_dim + self._priv_dim]
        #     hist_obs = obs[..., self._prop_dim + self._priv_dim:]
        #
        #     # 2. 预测特权信息
        #     pred_priv = self._se_net(hist_obs)
        #
        #     # 3. 切换流向
        #     if self.training:
        #         # 训练策略时：让 Actor 看绝对真实的上帝视角信息 (Teacher)
        #         actor_input = torch.cat([prop_obs, priv_obs], dim=-1)
        #     else:
        #         # 评估或实机部署时：剥离真实特权，强行拼接预测的特权信息 (Student)
        #         actor_input = torch.cat([prop_obs, pred_priv], dim=-1)

        else:
            # 如果不开 RMA，直接走原始逻辑
            actor_input = obs

        # 接下来走你原版的骨干网络推理
        x = self._actor_layers(actor_input)
        dist = self._action_dist(x)
        return dist

    def _build_actor(self, config, env):
        if self._enable_rma:
            # 1. 初始化 RMA 的维度和网络
            self._prop_dim = env.prop_dim
            self._priv_dim = env.priv_dim
            self._hist_dim = env.hist_dim
            self._rma_latent_dim = config.get("rma_latent_dim", 8)

            self._priv_encoder = torch.nn.Sequential(
                torch.nn.Linear(self._priv_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self._rma_latent_dim)
            )

            # self._hist_encoder = torch.nn.Sequential(
            #     torch.nn.Linear(self._hist_dim, 128),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(128, 64),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(64, self._rma_latent_dim)
            # )
            # 加入 Dropout 提高历史特征提取的泛化性和稳定性
            self._hist_encoder = torch.nn.Sequential(
                torch.nn.Linear(self._hist_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.1),  # <--- 新增
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.1),  # <--- 新增
                torch.nn.Linear(64, self._rma_latent_dim)
            )

            # 2. 构造 Actor 主干网络 (重写 input_dict 欺骗 net_builder)
            import gymnasium.spaces as spaces
            import numpy as np
            net_name = config["actor_net"]
            input_dict = self._build_actor_input_dict(env)

            # RMA 模式下，Actor骨干网络的真实输入是: 本体感知 + Latent
            rma_actor_dim = self._prop_dim + self._rma_latent_dim
            input_dict["obs"] = spaces.Box(low=-np.inf, high=np.inf, shape=(rma_actor_dim,), dtype=np.float32)
            self._actor_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                                    activation=self._activation)
            self._action_dist = self._build_action_distribution(config, env, self._actor_layers)
        elif self._enable_se:
            # 1. 初始化维度
            self._prop_dim = env.prop_dim
            self._priv_dim = env.priv_dim
            self._hist_dim = env.hist_dim

            # 2. 构建显式状态估计网络 (SE Net)
            self._se_net = torch.nn.Sequential(
                torch.nn.Linear(self._hist_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self._priv_dim)
            )

            # 3. 构造 Actor 主干网络
            import gymnasium.spaces as spaces
            import numpy as np
            net_name = config["actor_net"]
            input_dict = self._build_actor_input_dict(env)

            # SE 模式下，Actor骨干网络的真实输入是: 本体感知 + 特权信息(真实或预测)
            se_actor_dim = self._prop_dim + self._priv_dim
            input_dict["obs"] = spaces.Box(low=-np.inf, high=np.inf, shape=(se_actor_dim,), dtype=np.float32)
            self._actor_layers, layers_info = net_builder.build_net(net_name, input_dict, activation=self._activation)
            self._action_dist = self._build_action_distribution(config, env, self._actor_layers)

        else:
            # 如果不开 RMA，直接走父类 (PPOModel) 的默认初始化
            super()._build_actor(config, env)

        return

    def get_actor_params(self):
        # 1. 获取原版 PPO 的主干网络参数
        params = super().get_actor_params()

        # 2. 将 RMA 的编码器参数追加进优化器
        if getattr(self, "_enable_rma", False):
            params.extend(list(self._priv_encoder.parameters()))
            params.extend(list(self._hist_encoder.parameters()))

        # 3. 兼容你可能正在并行的显式 SE 网络
        elif getattr(self, "_enable_se", False):
            params.extend(list(self._se_net.parameters()))

        return params