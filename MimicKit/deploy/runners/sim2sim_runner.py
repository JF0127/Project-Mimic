import time

import mujoco
import numpy as np
import torch
import utils.observation as obs_utils
from config.base_config import BaseRobotConfig
from runners.base_runner import BaseRunner


class Sim2SimRunner(BaseRunner):
    """
    Sim2Sim 推理运行器 (集成运动学下降启动)。

    特性:
    1. [Kinematic Descent] 启动时锁定姿态匀速下降，直到双脚触地。
    2. [Pose Holding] 下降过程中，Target 锁定为 Default Pose，KP/KD 拉满，保持空中伸腿姿态。
    3. [Smooth Handover] 触地瞬间 (Phase Reset)，神经网络接管控制，且步态相位归零。
    """

    def __init__(self, config: BaseRobotConfig):
        super().__init__(config, log_path="logs/sim2sim_log.csv")

        # 1. 加载策略
        print(f"[Sim2Sim] Loading policy from: {config.policy_path}")
        try:
            self.policy = torch.jit.load(config.policy_path, map_location=self.device)
            self.policy.eval()
        except Exception as e:
            raise RuntimeError(f"无法加载模型文件: {e}")
        # 如果 Config 里没写 policy_joint_names，默认全用
        policy_names = getattr(self.cfg, "policy_joint_names", self.cfg.joint_names)

        # 找到策略关节在全身关节列表中的索引位置
        # 例如: left_knee 在 full_list 的第 3 位
        self.policy_indices = [self.cfg.joint_names.index(name) for name in policy_names]

        # 转为 Tensor 方便后续操作
        self.policy_indices_tensor = torch.tensor(self.policy_indices, dtype=torch.long, device=self.device)

        print(f"[Mapping] Policy controls {len(self.policy_indices)} / {len(self.cfg.joint_names)} joints.")

        # 2. 预处理常量
        # 1. 物理引擎用的 (Full Body, Numpy)
        self.kp_base_np = self.cfg.parse_params_to_array(self.cfg.kps)
        self.kd_base_np = self.cfg.parse_params_to_array(self.cfg.kds)
        self.torque_limits_np = self.cfg.parse_params_to_array(self.cfg.torque_limits)
        self.default_pos_np = self.cfg.parse_params_to_array(self.cfg.default_dof_pos)

        # 2. 策略用的 (Subset, Tensor) -> 需要切片！

        # 获取全身的 default pos tensor
        full_default_pos_tensor = torch.tensor(self.default_pos_np, dtype=torch.float32, device=self.device)

        # 策略只关心它控制的那部分 default pos (用于构建 Observation)
        self.policy_default_pos_tensor = full_default_pos_tensor[self.policy_indices_tensor].unsqueeze(0)

        # Action Scales 也只取策略部分的
        full_action_scales = torch.tensor(
            self.cfg.parse_params_to_array(self.cfg.action_scales),
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_action_scale = full_action_scales[self.policy_indices_tensor].unsqueeze(0)

        # Clip 也只取策略部分的
        full_clip = torch.tensor(
            self.cfg.parse_params_to_array(self.cfg.clip_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_clip = full_clip[self.policy_indices_tensor].unsqueeze(0)

        # 3. 运行变量 (维度 = Policy Output Dim)
        self.last_action = torch.zeros((1, len(policy_names)), dtype=torch.float32, device=self.device)
        self.command = torch.tensor([0.4, 0.0, 0.0], dtype=torch.float32, device=self.device).unsqueeze(0)

        # 4. 全身目标 buffer (用于最后传给 Engine)
        self.full_target_pos_tensor = full_default_pos_tensor.clone().unsqueeze(0)  # 初始值为 Default Pose

        # [关键] 记录触地时刻，用于重置 Policy Time
        self.touch_down_time = None

    def reset(self):
        """
        重写 Reset：初始化位置并开启下降模式
        """
        # 1. 物理重置
        init_pos = self.cfg.parse_params_to_array(self.cfg.default_dof_pos)
        base_pos = np.array([0.0, 0.0, 0.8])  # 抬高一点，给下降留空间

        self.engine.reset(initial_dof_pos=init_pos, base_pos=base_pos)

        # 2. 开启隐形电梯 (下降速度 -0.2 m/s)
        self.engine.start_kinematic_descent(height=0.8, descent_speed=-0.2)

        # 3. 状态重置
        self.last_action.fill_(0.0)
        self.touch_down_time = None  # 尚未触地
        self.step_counter = 0

        mujoco.mj_forward(self.engine.model, self.engine.data)
        print("[Sim2Sim] Reset complete. Descent Mode ON.")

    def _run_loop(self, viewer):
        print("\n=== 🤖 Sim2Sim Inference Started ===")
        print("💡 Phase 1: Kinematic Descent (Waiting for contact)...")

        while viewer.is_running():
            step_start = time.time()

            # 当前物理时间
            sim_time = self.step_counter * self.cfg.decimation * self.cfg.dt

            # ------------------------------------------------------------------
            # [状态检测] 检查是否刚刚触地
            # ------------------------------------------------------------------
            is_descending = self.engine.is_descent_mode

            if not is_descending and self.touch_down_time is None:
                # 刚刚从 True 变为 False -> 触地瞬间！
                self.touch_down_time = sim_time
                print(f"\n[Sim2Sim] 🦶 Touchdown at {sim_time:.2f}s! Policy taking over...")

            # ------------------------------------------------------------------
            # Step 1: 获取状态 (Full Body)
            # ------------------------------------------------------------------
            base_quat_np, base_ang_vel_np, dof_pos_np, dof_vel_np = self.engine.get_sensors()

            # 转 Tensor
            base_quat = torch.from_numpy(base_quat_np).float().to(self.device).unsqueeze(0)
            base_ang_vel = torch.from_numpy(base_ang_vel_np).float().to(self.device).unsqueeze(0)
            full_dof_pos = torch.from_numpy(dof_pos_np).float().to(self.device).unsqueeze(0)
            full_dof_vel = torch.from_numpy(dof_vel_np).float().to(self.device).unsqueeze(0)

            # ------------------------------------------------------------------
            # [关键] 观测切片 (Slice for Observation)
            # ------------------------------------------------------------------
            # 神经网络只想要它控制的那部分关节的状态
            policy_dof_pos = full_dof_pos[:, self.policy_indices_tensor]
            policy_dof_vel = full_dof_vel[:, self.policy_indices_tensor]

            # ------------------------------------------------------------------
            # Step 2: 计算策略时间 (Policy Time)
            # ------------------------------------------------------------------
            # 如果还没触地，Policy Time 锁定为 0
            # 如果已触地，Policy Time = 当前时间 - 触地时刻
            if self.touch_down_time is None:
                policy_time = 0.0
            else:
                policy_time = sim_time - self.touch_down_time

            # ------------------------------------------------------------------
            # Step 3: 构建观测 (Build Obs)
            # ------------------------------------------------------------------
            obs_list = [
                obs_utils.get_obs_ang_vel(base_ang_vel, self.cfg.ang_vel_scale),
                obs_utils.get_obs_gravity(base_quat, self.device),
                obs_utils.get_obs_cmd(self.command, self.cfg.lin_vel_scale),
                # 注意：这里传入的是切片后的 policy_dof_pos 和 policy_default_pos
                obs_utils.get_obs_dof_pos(
                    policy_dof_pos,
                    self.policy_default_pos_tensor,
                    self.cfg.dof_pos_scale,
                ),
                obs_utils.get_obs_dof_vel(policy_dof_vel, self.cfg.dof_vel_scale),
                obs_utils.get_obs_last_action(self.last_action),
            ]

            if hasattr(self.cfg, "gait_period") and self.cfg.gait_period > 0:
                phase_obs = obs_utils.get_obs_gait_phase(policy_time, self.cfg.gait_period, self.device)
                obs_list.append(phase_obs)

            obs = torch.cat(obs_list, dim=-1)

            # ------------------------------------------------------------------
            # Step 4: 决策 (Hold Pose vs Inference)
            # ------------------------------------------------------------------
            target_pos_np = None
            raw_actions = None
            scaled_action = None

            if is_descending:
                # 下降模式：全身都去 Default Pose (包括手臂)
                target_pos_np = self.default_pos_np
                self.last_action.fill_(0.0)
            else:
                # 推理模式
                with torch.no_grad():
                    raw_actions = self.policy(obs)

                # 后处理 (只针对 Policy 输出的部分)
                clipped_action = torch.clamp(raw_actions, -self.policy_clip, self.policy_clip)
                scaled_action = clipped_action * self.policy_action_scale

                # [核心逻辑] 动作拼接
                # 1. 这是一个全新的全是 Default Pose 的向量
                full_target = self.full_target_pos_tensor.clone()

                # 2. 计算受控关节的目标位置 (Default + Action)
                policy_target = self.policy_default_pos_tensor + scaled_action

                # 3. 将受控关节的目标位置填入全身向量
                # 这样：受控关节 = 新目标，未受控关节(手臂) = Default Pose
                full_target[:, self.policy_indices_tensor] = policy_target

                target_pos_np = full_target.squeeze(0).cpu().numpy()
                self.last_action = raw_actions.clone()

            # ------------------------------------------------------------------
            # Step 5: 物理执行
            # ------------------------------------------------------------------
            for _ in range(self.cfg.decimation):
                self.engine.step(
                    target_dof_pos=target_pos_np,  # 这是一个包含22个关节的完整向量
                    kp=self.kp_base_np,  # 22个关节的 KP
                    kd=self.kd_base_np,  # 22个关节的 KD
                    torque_limits=self.torque_limits_np,
                )

            # ------------------------------------------------------------------
            # Step 6: 日志与同步
            # ------------------------------------------------------------------
            if raw_actions is not None:
                self.logger.log(
                    {
                        "time": sim_time,
                        "obs": obs,
                        "act_raw": raw_actions,
                        "act_scaled": scaled_action,
                        "mode": 0 if is_descending else 1,  # 0=Descent, 1=RL
                        "phase": (
                            policy_time % self.cfg.gait_period
                            if hasattr(self.cfg, "gait_period")
                            else 0
                        ),
                    }
                )

            self.step_counter += 1
            self.sync_viewer(viewer, step_start)

            if self.step_counter % 50 == 0:
                mode_str = "DESCENT" if is_descending else "RL_RUN"
                phase_val = policy_time % self.cfg.gait_period if hasattr(self.cfg, "gait_period") else 0
                print(
                    f"\rTime: {sim_time:.2f}s | Mode: {mode_str} | Phase: {phase_val:.2f}",
                    end="",
                )
