# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import time

import mujoco
import numpy as np
from config.base_config import BaseRobotConfig
from runners.base_runner import BaseRunner
from utils.math_utils import get_rpy_from_quat


class PDRunner(BaseRunner):
    """
    PD 控制运行器 (满额 PD + 运动学触地检测).
    """

    def __init__(
        self,
        config: BaseRobotConfig,
        mode: str = "stand",
        target_joint: str | None = None,
    ):
        super().__init__(config, log_path="logs/pd_test_log.csv")

        self.mode = mode
        self.target_joint = target_joint
        self.initial_height = 0.8  # 初始下落高度

        # 数据预处理
        self.default_pos = self.cfg.parse_params_to_array(self.cfg.default_dof_pos)
        self.kp_base = self.cfg.parse_params_to_array(self.cfg.kps)
        self.kd_base = self.cfg.parse_params_to_array(self.cfg.kds)
        self.torque_limits = self.cfg.parse_params_to_array(self.cfg.torque_limits)

    def reset(self):
        """
        重写 reset：加入下降逻辑
        """
        init_pos = self.cfg.parse_params_to_array(self.cfg.default_dof_pos)
        base_pos = np.array([0.0, 0.0, self.initial_height])

        # 调用 Engine 重置
        self.engine.reset(initial_dof_pos=init_pos, base_pos=base_pos)

        # [启动下降] 恒定速度 -0.2 m/s，直到双脚触地
        self.engine.start_kinematic_descent(height=self.initial_height, descent_speed=-0.2)

        mujoco.mj_forward(self.engine.model, self.engine.data)
        self.step_counter = 0
        print("[PDRunner] Kinematic Descent: ON. Speed: -0.2 m/s. Waiting for contact...")

    def _run_loop(self, viewer):
        if self.mode == "stand":
            self._test_stand(viewer)

    def _test_stand(self, viewer):
        print("\n=== 🧍 启动站立测试 (详细 Ankle Log) ===")
        self.step_counter = 0

        # [新增] 动态获取关节索引，确保准确无误
        try:
            # 左脚
            l_ank_p_idx = self.cfg.joint_names.index("left_ankle_pitch_joint")
            l_ank_r_idx = self.cfg.joint_names.index("left_ankle_roll_joint")
            # 右脚
            r_ank_p_idx = self.cfg.joint_names.index("right_ankle_pitch_joint")
            r_ank_r_idx = self.cfg.joint_names.index("right_ankle_roll_joint")
        except ValueError as e:
            print(f"[Error] 无法在 joint_names 中找到踝关节: {e}")
            return

        current_target = self.default_pos

        while viewer.is_running():
            step_start = time.time()
            current_time = self.step_counter * self.cfg.decimation * self.cfg.dt

            last_torque, last_kp, last_kd = None, None, None

            # 物理步进
            for _ in range(self.cfg.decimation):
                last_torque, last_kp, last_kd = self.engine.step(
                    target_dof_pos=current_target,
                    kp=self.kp_base,
                    kd=self.kd_base,
                    torque_limits=self.torque_limits,
                )

            # 获取传感器数据
            base_quat, _, dof_pos, _ = self.engine.get_sensors()
            b_roll, b_pitch, b_yaw = get_rpy_from_quat(base_quat)

            # [修改] 详细记录 Ankle 数据 (Pos, Total Torque, KP, KD)
            self.logger.log({
                "time": current_time,
                # 基座姿态 (监控平衡)
                "Base_Roll": b_roll,
                "Base_Pitch": b_pitch,
                # === 左脚踝 Pitch (前后) ===
                "L_Ank_P_Pos": dof_pos[l_ank_p_idx],
                "L_Ank_P_Torque": last_torque[l_ank_p_idx],
                "L_Ank_P_KP": last_kp[l_ank_p_idx],  # 刚度贡献
                "L_Ank_P_KD": last_kd[l_ank_p_idx],  # 阻尼贡献
                # === 左脚踝 Roll (左右) ===
                "L_Ank_R_Pos": dof_pos[l_ank_r_idx],
                "L_Ank_R_Torque": last_torque[l_ank_r_idx],
                "L_Ank_R_KP": last_kp[l_ank_r_idx],
                "L_Ank_R_KD": last_kd[l_ank_r_idx],
                # === 右脚踝 Pitch (前后) ===
                "R_Ank_P_Pos": dof_pos[r_ank_p_idx],
                "R_Ank_P_Torque": last_torque[r_ank_p_idx],
                "R_Ank_P_KP": last_kp[r_ank_p_idx],
                "R_Ank_P_KD": last_kd[r_ank_p_idx],
                # === 右脚踝 Roll (左右) ===
                "R_Ank_R_Pos": dof_pos[r_ank_r_idx],
                "R_Ank_R_Torque": last_torque[r_ank_r_idx],
                "R_Ank_R_KP": last_kp[r_ank_r_idx],
                "R_Ank_R_KD": last_kd[r_ank_r_idx],
            })

            self.step_counter += 1
            self.sync_viewer(viewer, step_start)

            if self.step_counter % 50 == 0:
                status = "DESCENT" if self.engine.is_descent_mode else "LANDED"
                # 打印左脚 Roll 的力矩作为参考
                print(
                    f"\r[{status}] T:{current_time:.2f}s | Roll:{b_roll:.4f} |"
                    f" L_Ank_R_T:{last_torque[l_ank_r_idx]:.2f}",
                    end="",
                )
