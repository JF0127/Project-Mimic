# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field

from config.base_config import BaseRobotConfig


@dataclass
class Mvr10DofConfig(BaseRobotConfig):
    """
    Mvr_10dof 机器人的具体配置。
    数据来源：基于 2026-01-24 提供的 mvr_deploy.py 代码。
    """

    # ==========================================
    # 1. 身份与路径 (请确认路径是否正确)
    # ==========================================
    name: str = "Mvr_10dof"

    # [必须修改] 请确保这些是机器上的真实绝对路径
    xml_path: str = "/home/jf/lab/3_my_projects/assets/Mvr_10dof/Mvr_10dof_no_col_inf.xml"
    policy_path: str = (
        "/home/jf/lab/3_my_projects/Project01_Isaaclab/logs/rsl_rl/Mvr_10dof_unitree/2026-01-24_14-00-39/exported/policy.pt"
    )

    # ==========================================
    # 2. 仿真参数
    # ==========================================
    dt: float = 0.001  # 物理步长 1ms
    decimation: int = 10  # 控制频率 100Hz (10ms)

    lin_vel_scale: float = 1.0
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    gait_period: float = 0.8

    # 维度定义 (用于校验)
    # Obs: AngVel(3) + Gravity(3) + Cmd(3) + DofPos(10) + DofVel(10) + LastAct(10) + Phase(2) = 41
    num_observations: int = 41
    num_actions: int = 10

    # ==========================================
    # 3. 关节定义 (The Truth)
    # ==========================================
    foot_names: list[str] = field(default_factory=lambda: ["R_foot_1", "L_foot_1"])
    # 严格对应 Isaac Lab 训练时的顺序
    joint_names: list[str] = field(
        default_factory=lambda: [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ]
    )

    # ==========================================
    # 4. 物理参数 (全名显式定义)
    # ==========================================

    # --- A. 初始位置 (Nominal Pose) ---
    default_dof_pos: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_roll_joint": 0.00,
            "left_hip_yaw_joint": 0.00,
            "left_hip_pitch_joint": 0.25,
            "left_knee_joint": -0.60,
            "left_ankle_pitch_joint": 0.35,
            "right_hip_roll_joint": 0.00,
            "right_hip_yaw_joint": 0.00,
            "right_hip_pitch_joint": -0.25,
            "right_knee_joint": 0.60,
            "right_ankle_pitch_joint": 0.35,
        }
    )

    # --- B. 刚度 (Kp) ---
    # Hip Roll/Yaw: 40, Pitch/Knee: 150, Ankle: 80
    kps: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_roll_joint": 40.0,
            "left_hip_yaw_joint": 40.0,
            "left_hip_pitch_joint": 150.0,
            "left_knee_joint": 150.0,
            "left_ankle_pitch_joint": 80.0,
            "right_hip_roll_joint": 40.0,
            "right_hip_yaw_joint": 40.0,
            "right_hip_pitch_joint": 150.0,
            "right_knee_joint": 150.0,
            "right_ankle_pitch_joint": 80.0,
        }
    )

    # --- C. 阻尼 (Kd) ---
    # Hip Roll/Yaw: 2, Pitch/Knee: 8, Ankle: 3
    kds: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_roll_joint": 2.0,
            "left_hip_yaw_joint": 2.0,
            "left_hip_pitch_joint": 8.0,
            "left_knee_joint": 8.0,
            "left_ankle_pitch_joint": 3.0,
            "right_hip_roll_joint": 2.0,
            "right_hip_yaw_joint": 2.0,
            "right_hip_pitch_joint": 8.0,
            "right_knee_joint": 8.0,
            "right_ankle_pitch_joint": 3.0,
        }
    )

    # --- D. 动作缩放 (Action Scale) ---
    # 统一为 0.25
    action_scales: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_roll_joint": 0.25,
            "left_hip_yaw_joint": 0.25,
            "left_hip_pitch_joint": 0.25,
            "left_knee_joint": 0.25,
            "left_ankle_pitch_joint": 0.25,
            "right_hip_roll_joint": 0.25,
            "right_hip_yaw_joint": 0.25,
            "right_hip_pitch_joint": 0.25,
            "right_knee_joint": 0.25,
            "right_ankle_pitch_joint": 0.25,
        }
    )

    # --- E. 扭矩限制 (Torque Limits) ---
    # Roll/Yaw/Ankle (A4310): 35.9
    # Pitch/Knee (A8112): 54.2
    torque_limits: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_roll_joint": 35.9,
            "left_hip_yaw_joint": 35.9,
            "left_hip_pitch_joint": 54.2,
            "left_knee_joint": 54.2,
            "left_ankle_pitch_joint": 35.9,
            "right_hip_roll_joint": 35.9,
            "right_hip_yaw_joint": 35.9,
            "right_hip_pitch_joint": 54.2,
            "right_knee_joint": 54.2,
            "right_ankle_pitch_joint": 35.9,
        }
    )

    # --- F. 动作截断 (Clip Actions) ---
    # 默认为 100.0 (即不轻易截断)
    clip_actions: dict[str, float] = field(
        default_factory=lambda: {
            name: 100.0
            for name in [
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_hip_pitch_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_hip_pitch_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
            ]
        }
    )
