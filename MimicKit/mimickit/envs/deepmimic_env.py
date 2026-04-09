import numpy as np
import torch

import anim.motion as motion
import anim.motion_lib as motion_lib
import envs.base_env as base_env
import envs.char_env as char_env
import engines.engine as engine
import util.stats_tracker as stats_tracker
import util.torch_util as torch_util


class DeepMimicEnv(char_env.CharEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._enable_early_termination = env_config["enable_early_termination"]
        self._num_phase_encoding = env_config.get("num_phase_encoding", 0)

        self._pose_termination = env_config.get("pose_termination", False)
        self._pose_termination_dist = env_config.get("pose_termination_dist", 1.0)
        self._enable_phase_obs = env_config.get("enable_phase_obs", True)
        self._enable_tar_obs = env_config.get("enable_tar_obs", False)
        self._tar_obs_steps = env_config.get("tar_obs_steps", [1])
        self._tar_obs_steps = torch.tensor(self._tar_obs_steps, device=device, dtype=torch.int)
        self._rand_reset = env_config.get("rand_reset", True)
        
        self._ref_char_offset = torch.tensor(env_config["ref_char_offset"], device=device, dtype=torch.float)
        self._log_tracking_error = env_config.get("log_tracking_error", False)
        
        self._reward_pose_w = env_config.get("reward_pose_w")
        self._reward_vel_w = env_config.get("reward_vel_w")
        self._reward_root_pose_w = env_config.get("reward_root_pose_w")
        self._reward_root_vel_w = env_config.get("reward_root_vel_w")
        self._reward_key_pos_w = env_config.get("reward_key_pos_w")

        self._reward_pose_scale = env_config.get("reward_pose_scale")
        self._reward_vel_scale = env_config.get("reward_vel_scale")
        self._reward_root_pose_scale = env_config.get("reward_root_pose_scale")
        self._reward_root_vel_scale = env_config.get("reward_root_vel_scale")
        self._reward_key_pos_scale = env_config.get("reward_key_pos_scale")
        
        self._visualize_ref_char = env_config.get("visualize_ref_char", True)

        # ====== 新增：状态初始化域随机配置 ======
        self._state_init_mode = env_config.get("state_init_mode", "Ref")
        self._hybrid_init_probs = env_config.get("hybrid_init_probs", [1.0, 0.0, 0.0])
        self._init_noise_std = env_config.get("init_noise_std", [0.0, 0.0, 0.0])
        # ==================================
        self.use_commands = env_config.get("use_commands", False)

        # ====== 新增：读取非对称与 RMA 开关 ======
        self._asymmetric_obs = env_config.get("asymmetric_obs", False)
        self._enable_rma = env_config.get("enable_rma", False) if not self._asymmetric_obs else False
        self._rma_hist_len = engine_config.get("control_freq", 30)
        self.obs_noise = env_config.get("obs_noise", False)
        # self._rma_hist_len = env_config.get("rma_hist_len", 30)
        # 记录维度，方便外层 Agent/Model 知道如何切片
        self.prop_dim = 0
        self.priv_dim = 0
        self.hist_dim = 0
        self._prop_hist_buf = None
        self._enable_se = env_config.get("enable_se", False) if not self._asymmetric_obs else False  # <--- 新增
        # ==========================================
        self._enable_mirror_aug = env_config.get("enable_mirror_aug", False)
        if self._enable_mirror_aug:
            mirror_cfg = env_config.get("mirror_aug_config", {})
            # 存为 tensor 方便后续做 GPU 上的高效切片
            self._mirror_neg_idx = torch.tensor(mirror_cfg.get("negate_indices", []), dtype=torch.long,
                                                device=self._device)
            self._mirror_left_idx = torch.tensor(mirror_cfg.get("left_indices", []), dtype=torch.long,
                                                 device=self._device)
            self._mirror_right_idx = torch.tensor(mirror_cfg.get("right_indices", []), dtype=torch.long,
                                                  device=self._device)
            assert len(self._mirror_left_idx) == len(self._mirror_right_idx), "左侧和右侧的索引长度必须一致！"

        # ====== 新增：外力扰动配置 ======
        dr_cfg = env_config.get("domain_rand", {})
        self._enable_push = dr_cfg.get("push_robots", False)
        if self._enable_push:
            self._push_interval = dr_cfg.get("push_interval_s", 3.0)
            self._push_duration = dr_cfg.get("push_duration_s", 0.4)
            self._max_push_force = dr_cfg.get("max_push_force", 150.0)
            self._push_body_names = dr_cfg.get("push_bodies", ["head_link"])
        # ================================
        # ====== 新增：状态初始化域随机配置 ======
        self._state_init_mode = env_config.get("state_init_mode", "Ref")
        self._hybrid_init_probs = env_config.get("hybrid_init_probs", [1.0, 0.0, 0.0])
        self._init_noise_std = env_config.get("init_noise_std", [0.0, 0.0, 0.0, 0.0])  # <--- 这里加一个 0.0
        # >>> 新增：读取逐关节非对称噪声数组 >>>
        self._init_noise_std_dof = env_config.get("init_noise_std_dof", None)
        if self._init_noise_std_dof is not None:
            self._init_noise_std_dof = torch.tensor(self._init_noise_std_dof, device=device, dtype=torch.float32)
        # <<< 结束新增 <<<
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    # 新增通用镜像函数
    def _apply_mirror_aug(self, obs):
        """对输入的 amp_obs 批量应用镜像翻转"""
        flipped_obs = obs.clone()

        # 1. 对应维度取反 (比如 Y 轴移动，Yaw/Roll 旋转)
        if len(self._mirror_neg_idx) > 0:
            flipped_obs[:, self._mirror_neg_idx] *= -1.0

        # 2. 左右对称肢体数据互换
        if len(self._mirror_left_idx) > 0:
            left_data = obs[:, self._mirror_left_idx].clone()
            right_data = obs[:, self._mirror_right_idx].clone()
            flipped_obs[:, self._mirror_left_idx] = right_data
            flipped_obs[:, self._mirror_right_idx] = left_data

        return flipped_obs

    def get_critic_obs_space(self):
        import gymnasium.spaces as spaces
        import numpy as np
        if hasattr(self, "_critic_obs_buf"):
            shape = list(self._critic_obs_buf.shape[1:])
            import util.torch_util as torch_util
            dtype = torch_util.torch_dtype_to_numpy(self._critic_obs_buf.dtype)
        else:
            return self.get_obs_space()
        return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=dtype)

    def get_reward_succ(self):
        # setting the done flag flat to fail at the end of the motion avoids the
        # local minimal of a character just standing still until the end of the motion
        return 0.0
    
    def get_reward_fail(self):
        return 0.0
    
    def set_mode(self, mode):
        super().set_mode(mode)

        if (self._mode == base_env.EnvMode.TEST):
            if (self._log_tracking_error):      # ×
                self._error_tracker.reset()
        return

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        
        num_envs = self.get_num_envs()
        self._motion_ids = torch.zeros(num_envs, device=self._device, dtype=torch.int64)
        self._motion_time_offsets = torch.zeros(num_envs, device=self._device, dtype=torch.float32)
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        body_rot = self._engine.get_body_rot(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)

        self._ref_root_pos = torch.zeros_like(root_pos)
        self._ref_root_rot = torch.zeros_like(root_rot)
        self._ref_root_vel = torch.zeros_like(root_vel)
        self._ref_root_ang_vel = torch.zeros_like(root_ang_vel)
        self._ref_body_pos = torch.zeros_like(body_pos)
        self._ref_body_rot = torch.zeros_like(body_rot)
        self._ref_joint_rot = torch.zeros_like(body_rot[..., 1:, :])
        self._ref_dof_pos = torch.zeros_like(dof_pos) 
        self._ref_dof_vel = torch.zeros_like(dof_vel)
        
        contact_bodies = env_config.get("contact_bodies", [])
        self._contact_body_ids = self._build_body_ids_tensor(contact_bodies)

        joint_err_w = env_config.get("joint_err_w", None)
        self._parse_joint_err_weights(joint_err_w)

        self.commands = torch.zeros((self.get_num_envs(), 3), device=self._device, dtype=torch.float)
        self.command_resample_steps = torch.zeros(self.get_num_envs(), device=self._device, dtype=torch.long)

        # ====== 新增：外力扰动 Buffer ======
        if getattr(self, "_enable_push", False):
            # 将 body 名字映射为底层的 body_id
            self._push_body_ids = self._build_body_ids_tensor(self._push_body_names)
            num_envs = self.get_num_envs()

            # 保存每个环境当前的受力大小 (N, 3)
            self._push_force_buf = torch.zeros((num_envs, 3), device=self._device, dtype=torch.float)
            # 保存当前受力的部位ID
            self._push_body_idx_buf = torch.zeros(num_envs, dtype=torch.long, device=self._device)
            # 距离下次触发推力的计时器 (错开每个环境的初始触发时间)
            self._push_timer_buf = torch.rand(num_envs, device=self._device) * self._push_interval
            # 当前推力剩余的持续时间
            self._push_duration_buf = torch.zeros(num_envs, device=self._device)
        # ==================================

        return

    def _resample_commands(self, env_ids):
        # >>> 课程学习新增：动态指令范围 >>>
        progress = getattr(self, '_progress', 1.0)

        if progress < 0.2:
            # 阶段 1: 只能向前走 (限制 x 为正，y 和 yaw 为 0)
            min_x, max_x = 0.3, 1.0
            min_y, max_y = 0.0, 0.0
            min_yaw, max_yaw = 0.0, 0.0
        else:
            # 阶段 2+: 逐渐放开到全域指令
            alpha = min(1.0, (progress - 0.2) / 0.4)
            min_x = 0.3 + alpha * (-1.2 - 0.3)
            max_x = 1.0 + alpha * (1.2 - 1.0)
            min_y = 0.0 + alpha * (-0.5 - 0.0)
            max_y = 0.0 + alpha * (0.5 - 0.0)
            min_yaw = 0.0 + alpha * (-1.0 - 0.0)
            max_yaw = 0.0 + alpha * (1.0 - 0.0)

        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self._device).uniform_(min_x, max_x)
        self.commands[env_ids, 1] = torch.empty(len(env_ids), device=self._device).uniform_(min_y, max_y)
        self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self._device).uniform_(min_yaw, max_yaw)
        # <<< 课程学习新增 <<<
        #
        # 消除微小指令
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.1).unsqueeze(1)
        self.command_resample_steps[env_ids] = 0

    def _pre_physics_step(self, actions):
        # 先调用父类 (sim_env.py) 的方法，完成 action 下发
        super()._pre_physics_step(actions)

        # 施加外力扰动
        if getattr(self, "_enable_push", False):
            self._apply_push_forces()

    def _apply_push_forces(self):
        dt = self._engine.get_timestep()
        num_envs = self.get_num_envs()
        env_ids_all = torch.arange(num_envs, device=self._device)

        # 1. 更新时间倒计时
        self._push_timer_buf -= dt
        self._push_duration_buf -= dt

        # 2. 找到达到间隔、需要产生新推力的环境
        push_triggers = self._push_timer_buf <= 0
        if push_triggers.any():
            trigger_ids = env_ids_all[push_triggers]

            # 重置触发环境的计时器和持续时间
            self._push_timer_buf[trigger_ids] = self._push_interval
            self._push_duration_buf[trigger_ids] = self._push_duration

            # ====== 最终版：基于球坐标系的真实绳索拉力模拟 ======
            # 1. 随机绳索拉力大小 [0, max_force]
            forces = torch.rand(len(trigger_ids), device=self._device) * self._max_push_force

            # 2. 随机水平方位角 (Azimuth Angle) [0, 2π]，决定绳子从前后左右哪个方向拉
            azimuths = torch.rand(len(trigger_ids), device=self._device) * 2 * np.pi

            # 3. 随机仰角 (Elevation Angle)。
            # 0 代表纯水平拉扯，π/2 (90度) 代表纯垂直向上吊起。
            # 这里设在 [0, π/2.5] (即 0~72度)，既涵盖了平拉，也包含了朋友想要的强烈"向上提拉"感
            max_elevation = np.pi / 2.5
            elevations = torch.rand(len(trigger_ids), device=self._device) * max_elevation

            # 4. 将球坐标转换为笛卡尔坐标系的 X, Y, Z 受力分量
            # 严格保证：1. 实际合力大小永远等于 forces；2. Z 分量绝对 >= 0 (绝不会往下压)
            self._push_force_buf[trigger_ids, 0] = forces * torch.cos(elevations) * torch.cos(azimuths)
            self._push_force_buf[trigger_ids, 1] = forces * torch.cos(elevations) * torch.sin(azimuths)
            self._push_force_buf[trigger_ids, 2] = forces * torch.sin(elevations)
            # ======================================================

            # 随机选择施力部位 (head_link 还是 pelvis 等)
            rand_idx = torch.randint(0, len(self._push_body_ids), (len(trigger_ids),), device=self._device)
            self._push_body_idx_buf[trigger_ids] = self._push_body_ids[rand_idx]

        # 3. 对处于"正在受力持续期"的环境，下发受力到物理引擎
        active_push = self._push_duration_buf > 0
        if active_push.any():
            active_ids = env_ids_all[active_push]
            forces = self._push_force_buf[active_ids]
            body_ids = self._push_body_idx_buf[active_ids]
            char_id = self._get_char_id()

            # 批量操作：遍历配置中的部位ID，利用掩码下发，避免 Python 写 for env_id 循环造成卡顿
            for b_idx in self._push_body_ids:
                mask = (body_ids == b_idx)
                if mask.any():
                    env_subset = active_ids[mask]
                    force_subset = forces[mask]
                    # 这里的 set_body_forces 底层直接映射到 isaac gym 的 apply_rigid_body_force_tensors
                    self._engine.set_body_forces(env_subset, char_id, b_idx, force_subset)

    def _load_motions(self, motion_file):
        self._motion_lib = motion_lib.MotionLib(motion_file=motion_file, 
                                                kin_char_model=self._kin_char_model,
                                                device=self._device)
        return
    
    def _parse_joint_err_weights(self, joint_err_w):
        num_joints = self._kin_char_model.get_num_joints()

        if (joint_err_w is None):   # <=
            self._joint_err_w = torch.ones(num_joints - 1, device=self._device, dtype=torch.float32)
        else:
            self._joint_err_w = torch.tensor(joint_err_w, device=self._device, dtype=torch.float32)

        assert(self._joint_err_w.shape[-1] == num_joints - 1)
        
        dof_size = self._kin_char_model.get_dof_size()
        self._dof_err_w = torch.zeros(dof_size, device=self._device, dtype=torch.float32)

        for j in range(1, num_joints):
            dof_dim = self._kin_char_model.get_joint_dof_dim(j)
            if (dof_dim > 0):
                curr_w = self._joint_err_w[j - 1]
                dof_idx = self._kin_char_model.get_joint_dof_idx(j)
                self._dof_err_w[dof_idx:dof_idx + dof_dim] = curr_w
        return
    
    def _enable_ref_char(self):
        return self._visualize and self._visualize_ref_char

    def _get_ref_char_color(self):
        engine_name = self._engine.get_name()
        if (engine_name == "isaac_lab"):
            col = np.array([0.25, 0.4, 0.1])
        elif (engine_name == "newton"):
            col = np.array([0.3, 0.5, 0.1])
        else:
            col = np.array([0.5, 0.9, 0.1])
        return col

    # def _reset_char(self, env_ids):
    #     self._reset_ref_motion(env_ids)
    #     self._ref_state_init(env_ids)
    #
    #     if (self._enable_ref_char()):
    #         self._reset_ref_char(env_ids)
    #     return
    def _reset_char(self, env_ids):
        self._reset_ref_motion(env_ids)

        num_reset = len(env_ids)
        char_id = self._get_char_id()

        # 初始化目标状态 Tensor
        target_root_pos = torch.zeros((num_reset, 3), device=self._device)
        target_root_rot = torch.zeros((num_reset, 4), device=self._device)
        dof_size = self._kin_char_model.get_dof_size()
        target_dof_pos = torch.zeros((num_reset, dof_size), device=self._device)

        # >>> 课程学习新增：动态调整初始位姿分布 >>>
        progress = getattr(self, '_progress', 1.0)

        if progress < 0.6:
            # 阶段 1 和 2: 100% 使用参考动作初始化 (最简单)
            current_probs = [1.0, 0.0, 0.0, 0]
        else:
            # 阶段 3: 逐渐过渡到 YAML 里配置的高难度混合分布
            alpha = min(1.0, (progress - 0.6) / 0.4)
            p_ref = 1.0 + alpha * (self._hybrid_init_probs[0] - 1.0)
            p_init = 0.0 + alpha * (self._hybrid_init_probs[1] - 0.0)
            p_zero = 0.0 + alpha * (self._hybrid_init_probs[2] - 0.0)
            # 兼容旧配置：如果 yaml 里给了 4 个概率，就用第 4 个(Real)，否则默认为 0
            target_p_real = self._hybrid_init_probs[3] if len(self._hybrid_init_probs) > 3 else 0.0
            p_real = 0.0 + alpha * (target_p_real - 0.0)

            current_probs = [p_ref, p_init, p_zero, p_real]

        # 2. 决定每个 env 使用哪种基础位姿
        if self._state_init_mode == "Hybrid":
            probs = torch.tensor(current_probs, device=self._device)
            choices = torch.multinomial(probs, num_reset, replacement=True)
        elif self._state_init_mode == "Init":
            choices = torch.ones(num_reset, dtype=torch.long, device=self._device)
        elif self._state_init_mode == "Zero":
            choices = torch.full((num_reset,), 2, dtype=torch.long, device=self._device)
        elif self._state_init_mode == "Real":
            # 新增支持纯 Real 模式测试
            choices = torch.full((num_reset,), 3, dtype=torch.long, device=self._device)
        else:  # 默认 "Ref"
            choices = torch.zeros(num_reset, dtype=torch.long, device=self._device)
        # <<< 课程学习新增 <<<

        # --- 基础位姿 1: 参考动作 (Ref) ---
        ref_mask = choices == 0
        if ref_mask.any():
            target_root_pos[ref_mask] = self._ref_root_pos[env_ids[ref_mask]]
            target_root_rot[ref_mask] = self._ref_root_rot[env_ids[ref_mask]]
            target_dof_pos[ref_mask] = self._ref_dof_pos[env_ids[ref_mask]]

        # --- 基础位姿 2: YAML 初始位姿 (Init) ---
        init_mask = choices == 1
        if init_mask.any():
            # 广播单个 init_pose 到所有需要的 envs
            target_root_pos[init_mask] = self._init_root_pos.unsqueeze(0).expand(init_mask.sum(), -1)
            target_root_rot[init_mask] = self._init_root_rot.unsqueeze(0).expand(init_mask.sum(), -1)
            target_dof_pos[init_mask] = self._init_dof_pos.unsqueeze(0).expand(init_mask.sum(), -1)

        # --- 基础位姿 3: 全零位姿 (Zero) ---
        zero_mask = choices == 2
        if zero_mask.any():
            # Root 的位置和姿态继承 YAML 的安全初始状态 (Init)
            target_root_pos[zero_mask] = self._init_root_pos.unsqueeze(0).expand(zero_mask.sum(), -1)
            target_root_rot[zero_mask] = self._init_root_rot.unsqueeze(0).expand(zero_mask.sum(), -1)
            # 真正的 0位姿：仅仅是所有关节角归零
            target_dof_pos[zero_mask] = 0.0

        # --- 基础位姿 4: 真实的初始位姿 (Real) ---
        real_mask = choices == 3
        if real_mask.any():
            target_root_pos[real_mask] = self._init_root_pos_real.unsqueeze(0).expand(real_mask.sum(), -1)
            target_root_rot[real_mask] = self._init_root_rot_real.unsqueeze(0).expand(real_mask.sum(), -1)
            target_dof_pos[real_mask] = self._init_dof_pos_real.unsqueeze(0).expand(real_mask.sum(), -1)

        # >>> 课程学习新增：动态注入扰动噪声 >>>
        alpha = 0.0  # <--- 提前安全初始化，彻底杜绝作用域隐患
        # >>> 课程学习新增：动态注入扰动噪声 >>>
        if progress < 0.6:
            noise_pos_std, noise_rot_std, noise_dof_std = 0.0, 0.0, 0.0
            noise_dof_vel_std = 0.0  # <--- 新增
        else:
            alpha = min(1.0, (progress - 0.6) / 0.4)
            noise_pos_std = self._init_noise_std[0] * alpha
            noise_rot_std = self._init_noise_std[1] * alpha
            noise_dof_std = self._init_noise_std[2] * alpha
            # <--- 新增：安全获取第 4 个参数 (dof_vel)
            noise_dof_vel_std = self._init_noise_std[3] * alpha if len(self._init_noise_std) > 3 else 0.0

        if noise_pos_std > 0:
            pos_noise = torch.randn((num_reset, 2), device=self._device) * noise_pos_std
            target_root_pos[:, :2] += pos_noise

        if noise_rot_std > 0:
            rot_noise_vec = torch.randn((num_reset, 3), device=self._device) * noise_rot_std
            rot_noise_quat = torch_util.exp_map_to_quat(rot_noise_vec)
            target_root_rot = torch_util.quat_mul(rot_noise_quat, target_root_rot)

        if noise_dof_std > 0:
            # 1. 获取物理引擎真实的 XML 限位
            dof_low, dof_high = self._engine.get_obj_dof_limits(0, char_id)
            dof_low_tensor = torch.tensor(dof_low, device=self._device, dtype=torch.float32)
            dof_high_tensor = torch.tensor(dof_high, device=self._device, dtype=torch.float32)

            # 2. 确定随机波动的半范围 (包含课程学习 alpha 衰减)
            if getattr(self, "_init_noise_std_dof", None) is not None:
                dof_noise_range = self._init_noise_std_dof * alpha
            else:
                dof_noise_range = noise_dof_std

            # 3. 动态计算每个环境、每个关节的安全随机采样区间 [safe_low, safe_high]
            # 原理：目标位姿往外扩 noise_range，但绝不越过 XML limit
            safe_low = torch.maximum(target_dof_pos - dof_noise_range, dof_low_tensor)
            safe_high = torch.minimum(target_dof_pos + dof_noise_range, dof_high_tensor)

            # 4. 在安全区间内做标准均匀分布采样: value = low + rand(0,1) * (high - low)
            target_dof_pos = safe_low + torch.rand((num_reset, dof_size), device=self._device) * (safe_high - safe_low)
        # <<< 结束修改 <<<

        # ====== 新增修复：将屏蔽的关节强制重置为初始安全角度，剥夺其噪声干扰 ======
        if getattr(self, "_enable_joint_mask", False) and len(self._masked_dof_indices) > 0:
            target_dof_pos[:, self._masked_dof_indices] = self._init_dof_pos[self._masked_dof_indices]
        # =========================================================================

        # 4. 下发给底层物理引擎 (速度强制设为 0，因为只是初始态)
        # >>> 新增局部：如果是焊死在空中的台架模式，绝对不能给根节点下发状态！ <<<
        if not getattr(self, "_fix_root", False):
            self._engine.set_root_pos(env_ids, char_id, target_root_pos)
            self._engine.set_root_rot(env_ids, char_id, target_root_rot)
            # self._engine.set_root_vel(env_ids, char_id, 0.0)
            # self._engine.set_root_ang_vel(env_ids, char_id, 0.0)

        self._engine.set_dof_pos(env_ids, char_id, target_dof_pos)
        # <--- 新增：应用关节速度噪声
        if noise_dof_vel_std > 0:
            dof_vel_noise = torch.randn((num_reset, dof_size), device=self._device) * noise_dof_vel_std

            # ====== 新增修复：屏蔽关节的速度强制为 0 ======
            if getattr(self, "_enable_joint_mask", False) and len(self._masked_dof_indices) > 0:
                dof_vel_noise[:, self._masked_dof_indices] = 0.0
            # ==============================================

            self._engine.set_dof_vel(env_ids, char_id, dof_vel_noise)
            # >>> 新增保护 <<<
            if not getattr(self, "_fix_root", False):
                root_vel_noise = torch.randn((num_reset, 3), device=self._device) * 0.005
                root_ang_vel_noise = torch.randn((num_reset, 3), device=self._device) * 0.005
                self._engine.set_root_vel(env_ids, char_id, root_vel_noise)
                self._engine.set_root_ang_vel(env_ids, char_id, root_ang_vel_noise)
        else:
            self._engine.set_dof_vel(env_ids, char_id, 0.0)
            # >>> 新增保护 <<<
            if not getattr(self, "_fix_root", False):
                self._engine.set_root_vel(env_ids, char_id, 0.0)
                self._engine.set_root_ang_vel(env_ids, char_id, 0.0)

        # >>> 修改局部：保护 set_body_vel 和 set_body_ang_vel <<<
        if not getattr(self, "_fix_root", False):
            self._engine.set_body_vel(env_ids, char_id, 0.0)
            self._engine.set_body_ang_vel(env_ids, char_id, 0.0)

        if (self._enable_ref_char()):
            self._reset_ref_char(env_ids)
        return



    def _reset_ref_char(self, env_ids):
        ref_char_id = self._get_ref_char_id()

        root_pos = self._ref_root_pos[env_ids] + self._ref_char_offset
        self._engine.set_root_pos(env_ids, ref_char_id, root_pos)
        self._engine.set_root_rot(env_ids, ref_char_id, self._ref_root_rot[env_ids])
        self._engine.set_root_vel(env_ids, ref_char_id, self._ref_root_vel[env_ids])
        self._engine.set_root_ang_vel(env_ids, ref_char_id, self._ref_root_ang_vel[env_ids])
        
        self._engine.set_dof_pos(env_ids, ref_char_id, self._ref_dof_pos[env_ids])
        self._engine.set_dof_vel(env_ids, ref_char_id, self._ref_dof_vel[env_ids])
        
        self._engine.set_body_vel(env_ids, ref_char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, ref_char_id, 0.0)
        return

    def _reset_ref_motion(self, env_ids):
        n = len(env_ids)
        motion_ids, motion_times = self._sample_motion_times(n)
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel
        
        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                             self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot

        dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
        self._ref_dof_pos[env_ids] = dof_pos
        return

    def _get_ref_char_id(self):
        return self._ref_char_ids[0]

    def _ref_state_init(self, env_ids):
        char_id = self._get_char_id()
        
        self._engine.set_root_pos(env_ids, char_id, self._ref_root_pos[env_ids])
        self._engine.set_root_rot(env_ids, char_id, self._ref_root_rot[env_ids])
        self._engine.set_root_vel(env_ids, char_id, self._ref_root_vel[env_ids])
        self._engine.set_root_ang_vel(env_ids, char_id, self._ref_root_ang_vel[env_ids])
        
        self._engine.set_dof_pos(env_ids, char_id, self._ref_dof_pos[env_ids])
        self._engine.set_dof_vel(env_ids, char_id, self._ref_dof_vel[env_ids])
        
        self._engine.set_body_vel(env_ids, char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)
        return

    def _get_motion_times(self, env_ids=None):
        if (env_ids is None):
            motion_times = self._time_buf + self._motion_time_offsets
        else:
            motion_times = self._time_buf[env_ids] + self._motion_time_offsets[env_ids]
        return motion_times

    def _update_misc(self):
        super()._update_misc()
        self._update_ref_motion()

        if (self._enable_ref_char()):
            self._update_ref_char()

        self.command_resample_steps += 1
        resample_ids = (self.command_resample_steps >= 200).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)
        # ====== 新增：调用外力可视化 ======
        if self._visualize and getattr(self, "_enable_push", False):
            self._draw_push_forces()
        # ==================================
        return

    def _draw_push_forces(self):
        # 1. 别在这里手动调 clear_lines，底层 render() 结束后会自动清理！
        active_push = self._push_duration_buf > 0
        if not active_push.any():
            return

        active_ids = torch.nonzero(active_push, as_tuple=False).flatten()
        forces = self._push_force_buf[active_ids].cpu().numpy()
        body_idxs = self._push_body_idx_buf[active_ids]
        char_id = self._get_char_id()

        max_render_envs = min(len(active_ids), 4)

        verts_list = []
        cols_list = []

        for i in range(max_render_envs):
            env_id = active_ids[i].item()
            b_idx = body_idxs[i].item()
            f_np = forces[i]

            # 这是绝对的世界坐标 (World Coordinates)
            body_pos = self._engine.get_body_pos(char_id)[env_id, b_idx].cpu().numpy()

            force_scale = 1.0 / 150.0
            start_pos = body_pos
            end_pos = start_pos + f_np * force_scale

            # [主线] 红色受力线
            verts_list.append([start_pos[0], start_pos[1], start_pos[2], end_pos[0], end_pos[1], end_pos[2]])
            cols_list.append([1.0, 0.0, 0.0])  # 红色

            # [锚记] 在受力源头画一个黄色三维十字，防止单根红线在特定视角下看不清
            s = 0.05
            verts_list.append(
                [start_pos[0] - s, start_pos[1], start_pos[2], start_pos[0] + s, start_pos[1], start_pos[2]])
            cols_list.append([1.0, 1.0, 0.0])  # 黄色
            verts_list.append(
                [start_pos[0], start_pos[1] - s, start_pos[2], start_pos[0], start_pos[1] + s, start_pos[2]])
            cols_list.append([1.0, 1.0, 0.0])
            verts_list.append(
                [start_pos[0], start_pos[1], start_pos[2] - s, start_pos[0], start_pos[1], start_pos[2] + s])
            cols_list.append([1.0, 1.0, 0.0])

        if len(verts_list) > 0:
            verts_np = np.array(verts_list, dtype=np.float32)
            cols_np = np.array(cols_list, dtype=np.float32)

            # 【终极修复】：第二个参数必须传 None！强行告诉引擎使用绝对世界坐标系！
            # 绝不能用 self._engine.draw_lines 传 env_id 进去。
            self._engine._gym.add_lines(self._engine._viewer, None, len(verts_list), verts_np, cols_np)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._resample_commands(env_ids)
            # ====== 新增：清空 Reset 环境的外力状态 ======
            if getattr(self, "_enable_push", False):
                # 给一个随机的初始倒计时，避免所有复活的环境在同一帧被拉拽
                self._push_timer_buf[env_ids] = torch.rand(len(env_ids), device=self._device) * self._push_interval
                # 持续时间设为0，立刻停止受力
                self._push_duration_buf[env_ids] = 0.0
            # =============================================
        return



    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_joint_rot[:] = joint_rot
        self._ref_dof_vel[:] = dof_vel

        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                             self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot

        if (self._enable_ref_char()):
            dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
            self._ref_dof_pos[:] = dof_pos
        return

    def _update_ref_char(self):
        ref_char_id = self._get_ref_char_id()

        root_pos = self._ref_root_pos + self._ref_char_offset
        body_pos = self._ref_body_pos + self._ref_char_offset

        self._engine.set_root_pos(None, ref_char_id, root_pos)
        self._engine.set_root_rot(None, ref_char_id, self._ref_root_rot)
        self._engine.set_root_vel(None, ref_char_id, 0.0)
        self._engine.set_root_ang_vel(None, ref_char_id, 0.0)
        
        self._engine.set_dof_pos(None, ref_char_id, self._ref_dof_pos)
        self._engine.set_dof_vel(None, ref_char_id, 0.0)

        self._engine.set_body_pos(None, ref_char_id, body_pos)
        self._engine.set_body_rot(None, ref_char_id, self._ref_body_rot)
        self._engine.set_body_vel(None, ref_char_id, 0.0)
        self._engine.set_body_ang_vel(None, ref_char_id, 0.0)
        return
    
    def _track_global_root(self):
        return self._enable_tar_obs and self._global_obs

    def _sample_motion_times(self, n):
        motion_ids = self._motion_lib.sample_motions(n)

        if (self._rand_reset):      # <=
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(n, dtype=torch.float, device=self._device)

        return motion_ids, motion_times

    def _build_data_buffers(self):
        super()._build_data_buffers()

        if (self._log_tracking_error):  # False
            num_track_errors = 7
            self._error_tracker = stats_tracker.StatsTracker(num_track_errors, device=self._device)
        return
    
    def _build_envs(self, env_config, num_envs):
        self._ref_char_ids = []

        super()._build_envs(env_config, num_envs)

        motion_file = env_config["motion_file"]
        self._load_motions(motion_file)
        return
    
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)

        if (self._enable_ref_char()):   # False
            ref_char_col = self._get_ref_char_color()
            ref_char_id = self._build_ref_character(env_id, config, color=ref_char_col)
            self._ref_char_ids.append(ref_char_id)
            
            if (env_id == 0):
                self._ref_char_ids.append(ref_char_id)
            else:
                ref_char_id0 = self._ref_char_ids[0]
                assert(ref_char_id0 == ref_char_id)
        return 
    
    def _build_ref_character(self, env_id, env_config, color):
        char_file = env_config["char_file"]
        char_id = self._engine.create_obj(env_id=env_id, 
                                          obj_type=engine.ObjType.articulated,
                                          asset_file=char_file, 
                                          name="ref_character",
                                          is_visual=True,
                                          enable_self_collisions=False,
                                          disable_motors=True,
                                          fix_root=self._fix_root,
                                          color=color)
        return char_id

    def _compute_obs(self, env_ids=None):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times(env_ids)
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]

            motion_ids = motion_ids[env_ids]

        # >>> 新增：如果你开启了 fix_root，强行屏蔽躯干姿态和速度，逼迫策略只看关节 <<<
        if getattr(self, "_fix_root", False):
            # 将四元数强制设为标准无旋转状态 (x,y,z,w) = (0,0,0,1)
            root_rot = torch.zeros_like(root_rot)
            root_rot[..., 3] = 1.0
            # 屏蔽线速度和角速度
            root_vel = torch.zeros_like(root_vel)
            root_ang_vel = torch.zeros_like(root_ang_vel)
        # <<< 结束新增 <<<

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        
        if (self._enable_phase_obs):
            motion_phase = self._motion_lib.calc_motion_phase(motion_ids, motion_times)
        else:                   # <=
            motion_phase = torch.zeros([0], device=self._device)

        if (self._has_key_bodies()):    # <=
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        if (self._enable_tar_obs):
            tar_root_pos, tar_root_rot, tar_joint_rot = self._fetch_tar_obs_data(motion_ids, motion_times)
            tar_root_pos_flat = torch.reshape(tar_root_pos, [tar_root_pos.shape[0] * tar_root_pos.shape[1], 
                                                             tar_root_pos.shape[-1]])
            tar_root_rot_flat = torch.reshape(tar_root_rot, [tar_root_rot.shape[0] * tar_root_rot.shape[1], 
                                                             tar_root_rot.shape[-1]])
            tar_joint_rot_flat = torch.reshape(tar_joint_rot, [tar_joint_rot.shape[0] * tar_joint_rot.shape[1], 
                                                               tar_joint_rot.shape[-2], tar_joint_rot.shape[-1]])
            tar_body_pos_flat, _ = self._kin_char_model.forward_kinematics(tar_root_pos_flat, tar_root_rot_flat,
                                                                           tar_joint_rot_flat)
            tar_body_pos = torch.reshape(tar_body_pos_flat, [tar_root_pos.shape[0], tar_root_pos.shape[1], 
                                                             tar_body_pos_flat.shape[-2], tar_body_pos_flat.shape[-1]])

            if (self._has_key_bodies()):
                tar_key_pos = tar_body_pos[..., self._key_body_ids, :]
            else:
                tar_key_pos = torch.zeros([0], device=self._device)
        else:                           # <=
            tar_root_pos = torch.zeros([0], device=self._device)
            tar_root_rot = tar_root_pos
            tar_joint_rot = tar_root_pos
            tar_key_pos = tar_root_pos

        if getattr(self, "use_commands", False) and hasattr(self, "commands"):
            commands = self.commands if env_ids is None else self.commands[env_ids]
        else:
            commands = torch.tensor([])
        # 此时已经拿到了 JIT 函数返回的纯净 tensor，且处于类方法中，可以安全使用 self
        if getattr(self, "_enable_push", False):
            # 只要 duration > 0，就认为是 1.0，否则是 0.0
            # 维度扩展为 [num_envs, 1]
            is_pushed_bool = (self._push_duration_buf > 0).float().unsqueeze(-1)
            # ====== 新增：修复维度炸裂的 BUG ======
            if env_ids is not None:
                is_pushed_bool = is_pushed_bool[env_ids]
            # ======================================
        else:
            is_pushed_bool = torch.tensor([])

        # ====== 新增：切除被屏蔽的关节数据 (包含 target 数据) ======
        if getattr(self, "_enable_joint_mask", False) and len(self._masked_dof_indices) > 0:
            joint_rot_in = joint_rot[..., self._unmasked_joint_rot_indices, :]
            dof_vel_in = dof_vel[..., self._unmasked_dof_indices]
            if self._enable_tar_obs:
                tar_joint_rot_in = tar_joint_rot[..., self._unmasked_joint_rot_indices, :]
            else:
                tar_joint_rot_in = tar_joint_rot
        else:
            joint_rot_in = joint_rot
            dof_vel_in = dof_vel
            tar_joint_rot_in = tar_joint_rot
        # ========================================================

        actor_obs, critic_obs, prop_obs, priv_obs = compute_deepmimic_obs(root_pos=root_pos,
                                    root_rot=root_rot, 
                                    root_vel=root_vel, 
                                    root_ang_vel=root_ang_vel,
                                    joint_rot=joint_rot_in,  # 修改此处
                                    dof_vel=dof_vel_in,  # 修改此处
                                    key_pos=key_pos,
                                    global_obs=self._global_obs,
                                    root_height_obs=self._root_height_obs,
                                    phase=motion_phase,
                                    num_phase_encoding=self._num_phase_encoding,
                                    enable_phase_obs=self._enable_phase_obs,
                                    enable_tar_obs=self._enable_tar_obs,
                                    tar_root_pos=tar_root_pos,
                                    tar_root_rot=tar_root_rot,
                                    tar_joint_rot=tar_joint_rot,
                                    tar_key_pos=tar_key_pos,
                                    cmds=commands,
                                    push=is_pushed_bool,
                                    asymmetric_obs=self._asymmetric_obs
                                                      )

            # # 将外力 Bool 强行塞入 特权观测(priv_obs) 和 评价网络观测(critic_obs)
            # priv_obs = torch.cat([priv_obs, is_pushed_bool], dim=-1)
            # critic_obs = torch.cat([critic_obs, is_pushed_bool], dim=-1)
        # ====== 关键修复：在这里全局注入持续的观测传感器噪声 ======
        if getattr(self, "obs_noise", False) and self._mode == base_env.EnvMode.TRAIN:
            # 采用 0.005 作为折中方案，避免 0.02 导致实机关节高频抖动
            prop_obs += torch.randn_like(prop_obs) * 0.005
        # ==================================================
        # 保存 Critic 观测，机制同上
        if not hasattr(self, '_critic_obs_buf'):
            self._critic_obs_buf = torch.zeros([self.get_num_envs(), critic_obs.shape[-1]], device=self._device)

        if env_ids is None:
            self._critic_obs_buf[:] = critic_obs
        else:
            self._critic_obs_buf[env_ids] = critic_obs

        # === 新增：暴露给 Agent ===
        if hasattr(self, '_info'):
            self._info["critic_obs"] = self._critic_obs_buf

        # ==========================================================
        # 【核心新增】：把物理引擎真实的 dof_pos 放进 info 字典，安全传递给 PPO！
        # 这里必须放缩减后的 dof_pos，否则 ppo_agent 计算 loss 维度不匹配！
        if getattr(self, "_enable_joint_mask", False) and len(self._masked_dof_indices) > 0:
            dof_pos_out = dof_pos[:, self._unmasked_dof_indices]
        else:
            dof_pos_out = dof_pos

        if not hasattr(self, '_dof_pos_buf'):
            self._dof_pos_buf = torch.zeros_like(dof_pos_out)
        if env_ids is None:
            self._dof_pos_buf[:] = dof_pos_out
        else:
            self._dof_pos_buf[env_ids] = dof_pos_out
        self._info["dof_pos"] = self._dof_pos_buf.clone()
        # ==========================================================

        # # 4. ====== RMA / SE 核心组装 ======
        # if self._enable_rma or self._enable_se:  # <--- 修改这里
        #     if self._prop_hist_buf is None:
        #         # 动态初始化，此时我们明确知道了 prop_obs 的精确维度
        #         self.prop_dim = prop_obs.shape[-1]
        #         self.priv_dim = priv_obs.shape[-1]
        #         self.hist_dim = self.prop_dim * self._rma_hist_len
        #         import util.circular_buffer as circular_buffer
        #         self._prop_hist_buf = circular_buffer.CircularBuffer(
        #             batch_size=self.get_num_envs(),
        #             buffer_len=self._rma_hist_len,
        #             shape=[self.prop_dim],
        #             dtype=prop_obs.dtype,
        #             device=self._device
        #         )
        #         # ====== 修复：直接用已经带噪的 prop_obs 填满历史 ======
        #         self._prop_hist_buf.fill(torch.arange(self.get_num_envs()),
        #                                  prop_obs.unsqueeze(1).repeat(1, self._rma_hist_len, 1))
        #
        #     # 推入最新帧
        #     if env_ids is None:
        #         self._prop_hist_buf.push(prop_obs)
        #     else:
        #         # # 【修改这里】：用当前的初始本体感知状态，填满被 reset 环境的整个历史窗口
        #         # # prop_obs[env_ids] 的维度是 [len(env_ids), prop_dim]
        #         # # unsqueeze 和 repeat 后变成 [len(env_ids), 30, prop_dim]，完美适配 fill 接口
        #         fill_data = prop_obs.unsqueeze(1).repeat(1, self._rma_hist_len, 1)
        #         self._prop_hist_buf.fill(env_ids, fill_data)
        #         # pass
        #
        #
        #         # 取出展平的历史 [N, T, prop_dim] -> [N, T*prop_dim]
        #     hist_obs = self._prop_hist_buf.get_all().reshape(self.get_num_envs(), -1)
        #     if env_ids is not None:
        #         hist_obs = hist_obs[env_ids]
        #
        #     # 强行重构 Actor Obs：[本体感知, 特权信息, 历史帧]
        #     rma_actor_obs = torch.cat([prop_obs, priv_obs, hist_obs], dim=-1)
        #     return rma_actor_obs

        # 如果不开 RMA，完美回退到原来的 actor_obs
        return actor_obs
    
    def _update_reward(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
            ref_key_pos = self._ref_body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)
            ref_key_pos = key_pos

        track_root_h = self._root_height_obs
        track_root = self._track_global_root()

        # 安全获取 commands，防止非控制环境报错
        cmds = self.commands if hasattr(self, "commands") else torch.zeros((self.get_num_envs(), 3),
                                                                           device=self._device)

        self._reward_buf[:] = compute_reward(root_pos=root_pos,
                                             root_rot=root_rot,
                                             root_vel=root_vel,
                                             root_ang_vel=root_ang_vel,
                                             joint_rot=joint_rot,
                                             dof_vel=dof_vel,
                                             key_pos=key_pos,
                                             
                                             tar_root_pos=self._ref_root_pos,
                                             tar_root_rot=self._ref_root_rot,
                                             tar_root_vel=self._ref_root_vel,
                                             tar_root_ang_vel=self._ref_root_ang_vel,
                                             tar_joint_rot=self._ref_joint_rot,
                                             tar_dof_vel=self._ref_dof_vel,
                                             tar_key_pos=ref_key_pos,
                                             
                                             joint_rot_err_w=self._joint_err_w,
                                             dof_err_w=self._dof_err_w,
                                             track_root_h=track_root_h,
                                             track_root=track_root,               
                                             
                                             pose_w=self._reward_pose_w,
                                             vel_w=self._reward_vel_w,
                                             root_pose_w=self._reward_root_pose_w,
                                             root_vel_w=self._reward_root_vel_w,
                                             key_pos_w=self._reward_key_pos_w,

                                             pose_scale=self._reward_pose_scale,
                                             vel_scale=self._reward_vel_scale,
                                             root_pose_scale=self._reward_root_pose_scale,
                                             root_vel_scale=self._reward_root_vel_scale,
                                             key_pos_scale=self._reward_key_pos_scale,

                                             cmds=cmds
                                             )
        return

    def _update_done(self):
        motion_times = self._get_motion_times()
        motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        motion_loop_mode = self._motion_lib.get_motion_loop_mode(self._motion_ids)
        motion_len_term = motion_loop_mode != motion.LoopMode.WRAP.value

        track_root = self._track_global_root()
        
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        ground_contact_forces = self._engine.get_ground_contact_forces(char_id)

        self._done_buf[:] = compute_done(done_buf=self._done_buf,
                                         time=self._time_buf, 
                                         ep_len=self._episode_length, 
                                         root_rot=root_rot,
                                         body_pos=body_pos,
                                         tar_root_rot=self._ref_root_rot,
                                         tar_body_pos=self._ref_body_pos,
                                         ground_contact_force=ground_contact_forces,
                                         contact_body_ids=self._contact_body_ids,
                                         pose_termination=self._pose_termination,
                                         pose_termination_dist=self._pose_termination_dist,
                                         global_obs=self._global_obs,
                                         enable_early_termination=self._enable_early_termination,
                                         motion_times=motion_times,
                                         motion_len=motion_len,
                                         motion_len_term=motion_len_term,
                                         track_root=track_root)
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)

        if (self._mode == base_env.EnvMode.TEST):
            if (self._log_tracking_error):      # ×
                self._record_tracking_error(env_ids)
        return
    
    def _record_tracking_error(self, env_ids=None):
        if (env_ids is None or len(env_ids) > 0):
            char_id = self._get_char_id()
            root_pos = self._engine.get_root_pos(char_id)
            root_rot = self._engine.get_root_rot(char_id)
            root_vel = self._engine.get_root_vel(char_id)
            root_ang_vel = self._engine.get_root_ang_vel(char_id)
            dof_pos = self._engine.get_dof_pos(char_id)
            dof_vel = self._engine.get_dof_vel(char_id)
            body_pos = self._engine.get_body_pos(char_id)
            body_rot = self._engine.get_body_rot(char_id)

            joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

            ref_root_pos = self._ref_root_pos
            ref_root_rot = self._ref_root_rot
            ref_joint_rot = self._ref_joint_rot
            ref_root_vel = self._ref_root_vel
            ref_root_ang_vel = self._ref_root_ang_vel
            ref_dof_vel = self._ref_dof_vel

            if env_ids is not None:
                root_pos = root_pos[env_ids]
                root_rot = root_rot[env_ids]
                joint_rot = joint_rot[env_ids]
                root_vel = root_vel[env_ids]
                root_ang_vel = root_ang_vel[env_ids]
                dof_vel = dof_vel[env_ids]
                body_pos = body_pos[env_ids]
                body_rot = body_rot[env_ids]

                ref_root_pos = ref_root_pos[env_ids]
                ref_root_rot = ref_root_rot[env_ids]
                ref_joint_rot = ref_joint_rot[env_ids]
                ref_root_vel = ref_root_vel[env_ids]
                ref_root_ang_vel = ref_root_ang_vel[env_ids]
                ref_dof_vel = ref_dof_vel[env_ids]
            
            ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(ref_root_pos, ref_root_rot, ref_joint_rot)

            tracking_error = compute_tracking_error(root_pos=root_pos,
                                                    root_rot=root_rot,
                                                    body_rot=body_rot,
                                                    body_pos=body_pos,

                                                    tar_root_pos=ref_root_pos,
                                                    tar_root_rot=ref_root_rot,
                                                    tar_body_rot=ref_body_rot,
                                                    tar_body_pos=ref_body_pos,

                                                    root_vel=root_vel,
                                                    root_ang_vel=root_ang_vel,
                                                    dof_vel=dof_vel,
                                                    tar_dof_vel=ref_dof_vel,
                                                    tar_root_vel=ref_root_vel,
                                                    tar_root_ang_vel=ref_root_ang_vel)

            self._error_tracker.update(tracking_error)

            err_stats = self._error_tracker.get_mean()
            self._diagnostics["root_pos_err"] = err_stats[0]
            self._diagnostics["root_rot_err"] = err_stats[1]
            self._diagnostics["body_pos_err"] = err_stats[2]
            self._diagnostics["body_rot_err"] = err_stats[3]
            self._diagnostics["dof_vel_err"] = err_stats[4]
            self._diagnostics["root_vel_err"] = err_stats[5]
            self._diagnostics["root_ang_vel_err"] = err_stats[6]
        return
    
    def _fetch_tar_obs_data(self, motion_ids, motion_times):
        n = motion_ids.shape[0]
        num_steps = self._tar_obs_steps.shape[0]
        assert(num_steps > 0)
        
        motion_times = motion_times.unsqueeze(-1)
        time_steps = self._engine.get_timestep() * self._tar_obs_steps
        motion_times = motion_times + time_steps
        motion_ids_tiled = torch.broadcast_to(motion_ids.unsqueeze(-1), motion_times.shape)

        motion_ids_tiled = motion_ids_tiled.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids_tiled, motion_times)
        
        root_pos = root_pos.reshape([n, num_steps, root_pos.shape[-1]])
        root_rot = root_rot.reshape([n, num_steps, root_rot.shape[-1]])
        joint_rot = joint_rot.reshape([n, num_steps, joint_rot.shape[-2], joint_rot.shape[-1]])

        return root_pos, root_rot, joint_rot


@torch.jit.script
def compute_phase_obs(phase, num_phase_encoding):
    # type: (Tensor, int) -> Tensor
    phase_obs = phase.unsqueeze(-1)

    # positional embedding of phase
    if (num_phase_encoding > 0):
        pe_exp = torch.arange(num_phase_encoding, device=phase.device, dtype=phase.dtype)
        pe_scale = 2.0 * np.pi * torch.pow(2.0, pe_exp)
        pe_scale = pe_scale.unsqueeze(0)
        pe_val = phase.unsqueeze(-1) * pe_scale
        pe_sin = torch.sin(pe_val)
        pe_cos = torch.cos(pe_val)

        phase_obs = torch.cat((phase_obs, pe_sin, pe_cos), dim=-1)

    return phase_obs

@torch.jit.script
def convert_to_local(root_rot, root_vel, root_ang_vel, key_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)

    local_root_rot = torch_util.quat_mul(heading_inv_rot, root_rot)
    local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)
    local_root_ang_vel = torch_util.quat_rotate(heading_inv_rot, root_ang_vel)
    
    if (len(key_pos) > 0):
        heading_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, key_pos.shape[1], 1))
        flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                                heading_rot_expand.shape[2])
        flat_key_pos = key_pos.reshape(key_pos.shape[0] * key_pos.shape[1], key_pos.shape[2])
        flat_local_key_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_key_pos)
        local_key_pos = flat_local_key_pos.reshape(key_pos.shape[0], key_pos.shape[1], key_pos.shape[2])
    else:
        local_key_pos = key_pos

    return local_root_rot, local_root_vel, local_root_ang_vel, local_key_pos

@torch.jit.script
def compute_tar_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, joint_rot, key_pos,
                    global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    ref_root_pos = ref_root_pos.unsqueeze(-2)
    root_pos_obs = root_pos - ref_root_pos
    
    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)

    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_pos.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                               heading_inv_rot_expand.shape[2]))
        root_pos_obs_flat = torch.reshape(root_pos_obs, [root_pos_obs.shape[0] * root_pos_obs.shape[1], root_pos_obs.shape[2]])
        root_pos_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
        root_pos_obs = torch.reshape(root_pos_obs_flat, root_pos.shape)
        
        root_rot = torch_util.quat_mul(heading_inv_rot_expand, root_rot)

        if (len(key_pos) > 0):
            heading_inv_rot_expand = heading_inv_rot_expand.unsqueeze(-2)
            heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 1, key_pos.shape[2], 1))
            heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1] * heading_inv_rot_expand.shape[2],
                                                                   heading_inv_rot_expand.shape[3]))
            key_pos_flat = key_pos.reshape((key_pos.shape[0] * key_pos.shape[1] * key_pos.shape[2],
                                            key_pos.shape[3]))
            key_pos_flat = torch_util.quat_rotate(heading_inv_rot_flat, key_pos_flat)
            key_pos = key_pos_flat.reshape(key_pos.shape)

    if (root_height_obs):
        root_pos_obs[..., 2] = root_pos[..., 2]
    else:
        root_pos_obs = root_pos_obs[..., :2]

    root_rot_flat = torch.reshape(root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]])
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(root_rot_obs_flat, [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]])

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1] * joint_rot.shape[2], joint_rot.shape[3]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2] * joint_rot_obs_flat.shape[-1]])
    
    obs = [root_pos_obs, root_rot_obs, joint_rot_obs]
    if (len(key_pos) > 0):
        key_pos = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1], key_pos.shape[2] * key_pos.shape[3]])
        obs.append(key_pos)

    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_deepmimic_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs, 
                          phase, num_phase_encoding, enable_phase_obs, 
                          enable_tar_obs, tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos,
                          cmds, push,
                          asymmetric_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, int, bool, bool, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    actor_char_obs, critic_char_obs, prop_obs, priv_obs = char_env.compute_char_obs(root_pos=root_pos,
                                            root_rot=root_rot,
                                            root_vel=root_vel,
                                            root_ang_vel=root_ang_vel,
                                            joint_rot=joint_rot,
                                            dof_vel=dof_vel,
                                            key_pos=key_pos,
                                            global_obs=global_obs,
                                            root_height_obs=root_height_obs,
                                            cmds=cmds,
                                            push=push,
                                            asymmetric_obs=asymmetric_obs
                                         )
    actor_obs_list = [actor_char_obs]
    critic_obs_list = [critic_char_obs]

    if (enable_phase_obs):  # False
        phase_obs = compute_phase_obs(phase=phase, num_phase_encoding=num_phase_encoding)
        actor_obs_list.append(phase_obs)
        critic_obs_list.append(phase_obs)

    if (enable_tar_obs):    # False
        if (global_obs):
            ref_root_pos = root_pos
            ref_root_rot = root_rot
        else:
            ref_root_pos = tar_root_pos[..., 0, :]
            ref_root_rot = tar_root_rot[..., 0, :]

        tar_obs = compute_tar_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=tar_root_pos, 
                                  root_rot=tar_root_rot, 
                                  joint_rot=tar_joint_rot,
                                  key_pos=tar_key_pos,
                                  global_obs=global_obs,
                                  root_height_obs=root_height_obs)
        
        tar_obs = torch.reshape(tar_obs, [tar_obs.shape[0], tar_obs.shape[1] * tar_obs.shape[2]])
        # 目标(target)观测通常两端都需要
        actor_obs_list.append(tar_obs)
        critic_obs_list.append(tar_obs)

    actor_obs = torch.cat(actor_obs_list, dim=-1)
    critic_obs = torch.cat(critic_obs_list, dim=-1)

    return actor_obs, critic_obs, prop_obs, priv_obs

@torch.jit.script
def compute_done(done_buf, time, ep_len, root_rot, body_pos, tar_root_rot, tar_body_pos, 
                 ground_contact_force, contact_body_ids,
                 pose_termination, pose_termination_dist, 
                 global_obs, enable_early_termination,
                 motion_times, motion_len, motion_len_term,
                 track_root):
    # type: (Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, float, bool, bool, Tensor, Tensor, Tensor, bool) -> Tensor
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    
    timeout = time >= ep_len
    done[timeout] = base_env.DoneFlags.TIME.value
    
    motion_end = motion_times >= motion_len
    motion_end = torch.logical_and(motion_end, motion_len_term)
    done[motion_end] = base_env.DoneFlags.SUCC.value

    if (enable_early_termination):
        failed = torch.zeros(done.shape, device=done.device, dtype=torch.bool)

        if (contact_body_ids.shape[0] > 0):
            masked_contact_buf = ground_contact_force.detach().clone()
            masked_contact_buf[:, contact_body_ids, :] = 0
            fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)

            has_fallen = torch.any(fall_contact, dim=-1)
            failed = torch.logical_or(failed, has_fallen)

        if (pose_termination):
            root_pos = body_pos[..., 0:1, :]
            tar_root_pos = tar_body_pos[..., 0:1, :]

            if (not global_obs):
                body_pos = body_pos[..., 1:, :] - root_pos
                tar_body_pos = tar_body_pos[..., 1:, :] - tar_root_pos
                body_pos = char_env.convert_to_local_root_body_pos(root_rot, body_pos)
                tar_body_pos = char_env.convert_to_local_root_body_pos(tar_root_rot, tar_body_pos)

            elif (not track_root):
                body_pos = body_pos[..., 1:, :] - root_pos
                tar_body_pos = tar_body_pos[..., 1:, :] - tar_root_pos

            body_pos_diff = tar_body_pos - body_pos
            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)
            body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]
            pose_fail = body_pos_dist > pose_termination_dist * pose_termination_dist

            if (track_root):
                root_pos_diff = tar_root_pos - root_pos
                root_pos_dist = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
                root_pos_fail = root_pos_dist > pose_termination_dist * pose_termination_dist
                root_pos_fail = root_pos_fail.squeeze(-1)
                pose_fail = torch.logical_or(pose_fail, root_pos_fail)

            failed = torch.logical_or(failed, pose_fail)
            
        # only fail after first timestep
        not_first_step = (time > 0.0)
        failed = torch.logical_and(failed, not_first_step)
        done[failed] = base_env.DoneFlags.FAIL.value
    
    return done

@torch.jit.script
def compute_reward(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos,
                   tar_root_pos, tar_root_rot, tar_root_vel, tar_root_ang_vel,
                   tar_joint_rot, tar_dof_vel, tar_key_pos,
                   joint_rot_err_w, dof_err_w, track_root_h, track_root,
                   pose_w, vel_w, root_pose_w, root_vel_w, key_pos_w,
                   pose_scale, vel_scale, root_pose_scale, root_vel_scale, key_pos_scale,
                   cmds):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, float, float, float, float, float, float, float, float, float, float, Tensor) -> Tensor
    pose_diff = torch_util.quat_diff_angle(joint_rot, tar_joint_rot)
    pose_err = torch.sum(joint_rot_err_w * pose_diff * pose_diff, dim=-1)

    vel_diff = tar_dof_vel - dof_vel
    vel_err = torch.sum(dof_err_w * vel_diff * vel_diff, dim=-1)

    root_pos_diff = tar_root_pos - root_pos

    if (not track_root):
        root_pos_diff[..., 0:2] = 0

    if (not track_root_h):
        root_pos_diff[..., 2] = 0

    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
    
    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)
        tar_key_pos = tar_key_pos - tar_root_pos.unsqueeze(-2)

    if (not track_root):
        root_rot, root_vel, root_ang_vel, key_pos = convert_to_local(root_rot, root_vel, root_ang_vel, key_pos)
        tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos = convert_to_local(tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos)
        
    root_rot_err = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err *= root_rot_err

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)

    if (len(key_pos) > 0):
        key_pos_diff = tar_key_pos - key_pos
        key_pos_err = torch.sum(key_pos_diff * key_pos_diff, dim=-1)
        key_pos_err = torch.sum(key_pos_err, dim=-1)
    else:
        key_pos_err = torch.zeros([0], device=key_pos.device)

    pose_r = torch.exp(-pose_scale * pose_err)
    vel_r = torch.exp(-vel_scale * vel_err)
    root_pose_r = torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
    root_vel_r = torch.exp(-root_vel_scale * (root_vel_err + 0.1 * root_ang_vel_err))
    key_pos_r = torch.exp(-key_pos_scale * key_pos_err)

    r = pose_w * pose_r \
        + vel_w * vel_r \
        + root_pose_w * root_pose_r \
        + root_vel_w * root_vel_r \
        + key_pos_w * key_pos_r

    return r

@torch.jit.script
def compute_tracking_error(root_pos, root_rot, body_rot, body_pos,
                            tar_root_pos, tar_root_rot,
                            tar_body_rot, tar_body_pos,
                            root_vel, root_ang_vel, dof_vel,
                            tar_root_vel, tar_root_ang_vel, tar_dof_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    body_pos = body_pos - root_pos.unsqueeze(-2)
    tar_body_pos = tar_body_pos - tar_root_pos.unsqueeze(-2)

    root_pos_diff = tar_root_pos - root_pos
    root_pos_err = torch.linalg.vector_norm(root_pos_diff, dim=-1)

    body_rot_diff = torch_util.quat_diff_angle(body_rot, tar_body_rot)
    body_rot_err = torch.abs(body_rot_diff)
    body_rot_err = torch.mean(body_rot_err, dim=-1)

    body_pos_diff = tar_body_pos - body_pos
    body_pos_diff_l2 = torch.linalg.vector_norm(body_pos_diff, dim=-1)
    body_pos_err = torch.mean(body_pos_diff_l2, dim=-1)

    root_rot_diff = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err = torch.abs(root_rot_diff)

    dof_vel_diff = tar_dof_vel - dof_vel
    dof_vel_err = torch.mean(torch.abs(dof_vel_diff), dim=-1)

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.mean(torch.abs(root_vel_diff), dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.mean(torch.abs(root_ang_vel_diff), dim=-1)

    tracking_error = torch.stack([root_pos_err, root_rot_err, body_pos_err, body_rot_err, dof_vel_err, root_vel_err, root_ang_vel_err], dim=-1)
    
    return tracking_error