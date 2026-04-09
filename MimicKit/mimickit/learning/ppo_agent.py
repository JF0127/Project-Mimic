import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.ppo_model as ppo_model
import learning.rl_util as rl_util
import util.mp_util as mp_util
import util.torch_util as torch_util

class PPOAgent(base_agent.BaseAgent):
    def __init__(self, config, env, device):
        self._action_pd_threshold = getattr(env, "_action_pd_threshold", 0.2)
        self._use_physical_bound_loss = getattr(env, "_use_physical_bound_loss", False)
        self._enable_action_pd_limit = getattr(env, "enable_action_pd_limit", False)
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)
        
        self._update_epochs = config["update_epochs"]
        self._batch_size = config["batch_size"]

        self._td_lambda = config["td_lambda"]
        self._ppo_clip_ratio = config["ppo_clip_ratio"]
        self._norm_adv_clip = config["norm_adv_clip"]

        self._action_bound_weight = config["action_bound_weight"]
        self._action_entropy_weight = config["action_entropy_weight"]
        self._action_reg_weight = config["action_reg_weight"]
        # === 新增：读取动作平滑惩罚权重 ===
        self._action_smooth_weight = config.get("action_smooth_weight", 0.0)

        self._action_pd_weight = config.get("action_pd_weight", 0.0) if self._enable_action_pd_limit else 0

        self._critic_loss_weight = config.get("critic_loss_weight", 0)
        self._critic_eval_batch_size = int(config.get("critic_eval_batch_size", 0))
        
        self._exp_anneal_samples = config.get("exp_anneal_samples", np.inf)
        self._exp_prob_beg = config.get("exp_prob_beg", 1.0)
        self._exp_prob_end = config.get("exp_prob_end", 1.0)
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = ppo_model.PPOModel(model_config, self._env)
        return
    
    def _get_exp_buffer_length(self):
        return self._steps_per_iter
    
    def _init_iter(self):
        super()._init_iter()
        self._exp_buffer.reset()
        return

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        norm_action_dist = self._model.eval_actor(norm_obs)

        if (self._mode == base_agent.AgentMode.TRAIN):
            norm_a_rand = norm_action_dist.sample()
            norm_a_mode = norm_action_dist.mode

            exp_prob = self._get_exp_prob()
            exp_prob = torch.full([norm_a_rand.shape[0], 1], exp_prob, device=self._device, dtype=torch.float)
            rand_action_mask = torch.bernoulli(exp_prob)
            norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
            rand_action_mask = rand_action_mask.squeeze(-1)

        elif (self._mode == base_agent.AgentMode.TEST):
            norm_a = norm_action_dist.mode  # 这不是均值，而是众数
            rand_action_mask = torch.zeros_like(norm_a[..., 0])
            
        else:
            assert(False), "Unsupported agent mode: {}".format(self._mode)
            
        norm_a_logp = norm_action_dist.log_prob(norm_a)

        norm_a = norm_a.detach()
        norm_a_logp = norm_a_logp.detach()
        a = self._a_norm.unnormalize(norm_a)

        a_info = {
            "a_logp": norm_a_logp,
            "rand_action_mask": rand_action_mask
        }
        return a, a_info

    def _record_data_pre_step(self, obs, info, action, action_info):
        # === 新增：在存入经验池前，先记录上一帧的动作和真实的关节角 ===
        if not hasattr(self, "_last_action_buf"):
            self._last_action_buf = info["dof_pos"].clone()  # 首次启动完美衔接

        self._exp_buffer.record("last_action", self._last_action_buf.clone())
        self._exp_buffer.record("dof_pos", info["dof_pos"].clone())  # 存下真实的 dof_pos

        # 更新缓存为当前帧动作
        self._last_action_buf.copy_(action)
        # ============================================
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("a_logp", action_info["a_logp"])
        self._exp_buffer.record("rand_action_mask", action_info["rand_action_mask"])
        return

    def _build_train_data(self):
        self.eval()
        
        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")

        # === 新增：取出特权观测 ===
        critic_obs = self._exp_buffer.get_data("critic_obs")
        next_critic_obs = self._exp_buffer.get_data("next_critic_obs")

        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")

        # norm_next_obs = self._obs_norm.normalize(next_obs)
        # next_critic_inputs = {"obs": norm_next_obs}
        # === 修改：用 next_critic_obs 评估下一个状态的价值 ===
        norm_next_critic_obs = self._critic_obs_norm.normalize(next_critic_obs)
        next_critic_inputs = {"obs": norm_next_critic_obs}
        next_vals = torch_util.eval_minibatch(self._model.eval_critic, next_critic_inputs, self._critic_eval_batch_size)
        next_vals = next_vals.squeeze(-1).detach()

        succ_val = self._compute_succ_val()
        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        next_vals[succ_mask] = succ_val

        fail_val = self._compute_fail_val()
        fail_mask = (done == base_env.DoneFlags.FAIL.value)
        next_vals[fail_mask] = fail_val

        new_vals = rl_util.compute_td_lambda_return(r, next_vals, done, self._discount, self._td_lambda)

        # norm_obs = self._obs_norm.normalize(obs)
        # critic_inputs = {"obs": norm_obs}
        # === 修改：用 critic_obs 评估当前状态的价值 ===
        norm_critic_obs = self._critic_obs_norm.normalize(critic_obs)
        critic_inputs = {"obs": norm_critic_obs}
        vals = torch_util.eval_minibatch(self._model.eval_critic, critic_inputs, self._critic_eval_batch_size)
        vals = vals.squeeze(-1).detach()
        adv = new_vals - vals
        
        rand_action_mask = (rand_action_mask == 1.0).flatten()
        adv_flat = adv.flatten()
        rand_action_adv = adv_flat[rand_action_mask]
        adv_mean, adv_std = mp_util.calc_mean_std(rand_action_adv)
        norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)
        
        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)
        
        info = {
            "adv_mean": adv_mean,
            "adv_std": adv_std
        }
        return info
    
    def _get_exp_prob(self):
        if (np.isfinite(self._exp_anneal_samples)):
            samples = self._sample_count
            l = float(samples) / self._exp_anneal_samples
            l = np.clip(l, 0.0, 1.0)
            prob = (1.0 - l) * self._exp_prob_beg + l * self._exp_prob_end
        else:
            prob = self._exp_prob_beg
        return prob

    def _update_model(self):
        self.train()

        num_envs = self.get_num_envs()
        num_samples = self._exp_buffer.get_sample_count()
        batch_size = self._batch_size * num_envs
        num_batches = int(np.ceil(float(num_samples) / batch_size))
        train_info = dict()

        for i in range(self._update_epochs):
            for b in range(num_batches):
                batch = self._exp_buffer.sample(batch_size)
                loss_info = self._compute_loss(batch)
                loss = loss_info["loss"]
                self._optimizer.step(loss)

                torch_util.add_torch_dict(loss_info, train_info)
        
        num_steps = self._update_epochs * num_batches
        torch_util.scale_torch_dict(1.0 / num_steps, train_info)

        return train_info
    
    def _compute_loss(self, batch):
        batch["norm_obs"] = self._obs_norm.normalize(batch["obs"])

        # === 新增：对 critic_obs 做单独的归一化 ===
        batch["norm_critic_obs"] = self._critic_obs_norm.normalize(batch["critic_obs"])

        batch["norm_action"] = self._a_norm.normalize(batch["action"])

        critic_info = self._compute_critic_loss(batch)
        actor_info = self._compute_actor_loss(batch)

        critic_loss = critic_info["critic_loss"]
        actor_loss = actor_info["actor_loss"]

        loss = actor_loss + self._critic_loss_weight * critic_loss

        info = {"loss":loss, **critic_info, **actor_info}
        return info

    def _compute_critic_loss(self, batch):
        # norm_critic_obs = batch["norm_obs"]
        # === 修改：让 critic 吃属于它的特权数据 ===
        norm_critic_obs = batch["norm_critic_obs"]

        tar_val = batch["tar_val"]
        pred = self._model.eval_critic(norm_critic_obs)
        pred = pred.squeeze(-1)

        diff = tar_val - pred
        loss = torch.mean(torch.square(diff))

        info = {
            "critic_loss": loss
        }
        return info

    def _compute_actor_loss(self, batch):
        norm_obs = batch["norm_obs"]
        norm_a = batch["norm_action"]
        old_a_logp = batch["a_logp"]
        adv = batch["adv"]
        rand_action_mask = batch["rand_action_mask"]

        # === 提取历史常量目标 (无梯度)，保留掩码对齐逻辑 ===
        last_a_unnorm = batch.get("last_action", batch["action"])
        dof_pos = batch.get("dof_pos", batch["action"])
        # ==================================

        # loss should only be computed using samples with random actions
        rand_action_mask = (rand_action_mask == 1.0)
        norm_obs = norm_obs[rand_action_mask]
        norm_a = norm_a[rand_action_mask]
        old_a_logp = old_a_logp[rand_action_mask]
        adv = adv[rand_action_mask]

        # === 新增：对历史状态目标施加相同掩码 ===
        last_a_unnorm = last_a_unnorm[rand_action_mask]
        dof_pos = dof_pos[rand_action_mask]
        # ==================================

        # ==========================================================
        # 【严格对齐 LCP 1：奇偶轮次交替】
        # self._iter 对应 LCP 的 current_learning_iteration
        use_hist = (self._iter % 2 == 1)

        # 将标志传入 eval_actor
        a_dist = self._model.eval_actor(norm_obs, use_hist=use_hist)
        # ==========================================================
        a_logp = a_dist.log_prob(norm_a)

        a_ratio = torch.exp(a_logp - old_a_logp)
        actor_loss0 = adv * a_ratio
        actor_loss1 = adv * torch.clamp(a_ratio, 1.0 - self._ppo_clip_ratio, 1.0 + self._ppo_clip_ratio)
        actor_loss = torch.minimum(actor_loss0, actor_loss1)
        actor_loss = -torch.mean(actor_loss)
        
        clip_frac = (torch.abs(a_ratio - 1.0) > self._ppo_clip_ratio).type(torch.float)
        clip_frac = torch.mean(clip_frac)
        imp_ratio = torch.mean(a_ratio)
        
        info = {
            "actor_loss": actor_loss,
            "clip_frac": clip_frac.detach(),
            "imp_ratio": imp_ratio.detach()
        }

        # # ==================== 新增: RMA 隐式蒸馏 Loss ====================
        # if getattr(self._model, "_enable_rma", False):
        #     priv_latent = self._model.current_priv_latent
        #     hist_latent = self._model.current_hist_latent
        #
        #     # # 使用 MSE 约束 hist_latent 逼近 priv_latent，注意 .detach() 阻断特权特征被错误拉扯
        #     if getattr(self, "use_ts_loss", False):
        #
        #         # 还原 LCP 真正的 priv_reg_loss / hist_latent_loss
        #         # (1) 学生拟合老师 (更新 hist_encoder)
        #         hist_latent_loss = (priv_latent.detach() - hist_latent).norm(p=2, dim=-1).mean()
        #         # (2) 老师拟合学生 (更新 priv_encoder，约束上帝视角)
        #         priv_reg_loss = (priv_latent - hist_latent.detach()).norm(p=2, dim=-1).mean()
        #         rma_weight = self._config.get("rma_loss_weight", 1.0)
        #         priv_reg_weight = self._config.get("priv_reg_weight", 1.0)
        #
        #         # 直接累加到 actor_loss，通过 PPO 原本的反向传播图一起更新
        #         actor_loss = actor_loss + rma_weight * hist_latent_loss+ priv_reg_weight * priv_reg_loss
        #
        #         # 必须更新 info 字典中的 actor_loss，因为外部 _compute_loss 是从 info 里面拿的
        #         info["actor_loss"] = actor_loss
        #         info["rma_loss"] = hist_latent_loss.detach()
        #         info["priv_reg_loss"] = priv_reg_loss.detach()
        #     # ===============================================================
        #     else:
        #         rma_loss = torch.nn.functional.mse_loss(hist_latent, priv_latent.detach())
        #         rma_weight = self._config.get("rma_loss_weight", 1.0)
        #         actor_loss = actor_loss + rma_weight * rma_loss
        #
        #         # 必须更新 info 字典中的 actor_loss，因为外部 _compute_loss 是从 info 里面拿的
        #         info["actor_loss"] = actor_loss
        #         info["rma_loss"] = rma_loss.detach()

        # === 核心修复：提取最新网络预测以接通计算图梯度 ===
        # a_dist.mode 是基于当前权重算出来的均值/众数，具有完整的 requires_grad=True
        curr_a_norm = a_dist.mode
        curr_a_unnorm = self._a_norm.unnormalize(curr_a_norm)

        # === 核心新增：在 return info 之前，加入双重惩罚 ===
        # 1. 动作平滑度 (限制网络当前意图 a_t 与物理历史上一帧 a_{t-1} 的剧烈跳变)
        if getattr(self, "_action_smooth_weight", 0.0) > 0:
            action_smooth_loss = torch.mean(torch.sum(torch.square(curr_a_unnorm - last_a_unnorm), dim=-1))
            actor_loss = actor_loss + self._action_smooth_weight * action_smooth_loss
            info["action_smooth_loss"] = action_smooth_loss.detach()

        # 2. 输出越界限制 (限制网络当前意图 a_t 偏离历史物理真实关节角 dof_pos_t 过大)
        if self._enable_action_pd_limit and getattr(self, "_action_pd_weight", 0.0) > 0:
            diff = torch.abs(curr_a_unnorm - dof_pos)
            violation = torch.clamp(diff - self._action_pd_threshold, min=0.0)
            action_pd_loss = torch.mean(torch.sum(torch.square(violation), dim=-1))

            actor_loss = actor_loss + self._action_pd_weight * action_pd_loss
            info["action_pd_loss"] = action_pd_loss.detach()
        # =========================================================

        if getattr(self, "_use_physical_bound_loss", False):
            # === 核心新增：物理边界惩罚 Loss (Boundary Loss) ===
            # 替代 Reward 扣分，直接用梯度逼迫网络遵守 XML 的物理限位
            bound_low = self._env._action_bound_low
            bound_high = self._env._action_bound_high

            # 算一下网络想输出的真实物理动作，超了边界多少
            violation_up = torch.nn.functional.relu(curr_a_unnorm - bound_high)
            violation_down = torch.nn.functional.relu(bound_low - curr_a_unnorm)

            # 平方求和算误差 (MSE)
            physical_bound_loss = torch.mean(
                torch.sum(torch.square(violation_up) + torch.square(violation_down), dim=-1))

            # 权重给大点（比如 10.0），只要越界，梯度直接一棒子打回来
            actor_loss = actor_loss + 10.0 * physical_bound_loss
            info["physical_bound_loss"] = physical_bound_loss.detach()
            # =========================================================

        if (self._action_bound_weight != 0):
            action_bound_loss = self._compute_action_bound_loss(a_dist)
            if (action_bound_loss is not None):
                action_bound_loss = torch.mean(action_bound_loss)
                actor_loss += self._action_bound_weight * action_bound_loss
                info["action_bound_loss"] = action_bound_loss.detach()

        if (self._action_entropy_weight != 0):
            action_entropy = a_dist.entropy()
            action_entropy = torch.mean(action_entropy)
            actor_loss += -self._action_entropy_weight * action_entropy
            info["action_entropy"] = action_entropy.detach()

        if (self._action_reg_weight != 0):
            action_reg_loss = a_dist.param_reg()
            action_reg_loss = torch.mean(action_reg_loss)
            actor_loss += self._action_reg_weight * action_reg_loss
            info["action_reg_loss"] = action_reg_loss.detach()

        # 最后必须将包含了所有正则化约束的总 loss 更新回 info
        info["actor_loss"] = actor_loss
        return info

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        super()._log_train_info(train_info, test_info, env_diag_info, start_time)
        self._logger.log("Exp_Prob", self._get_exp_prob())
        # 新增：计算并记录进度百分比
        if hasattr(self, '_max_samples'):
            progress = 100.0 * self._sample_count / self._max_samples
            self._logger.log("Progress_Percent", progress, collection="1_Info")
        return