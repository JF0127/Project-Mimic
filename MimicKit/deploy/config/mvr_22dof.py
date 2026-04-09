# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field

from config.base_config import BaseRobotConfig


@dataclass
class Mvr22DofConfig(BaseRobotConfig):
    """
    Mvr_22dof 机器人的具体配置。
    包含腿部(12) + 躯干(2) + 手臂(8) = 22 DoF
    数据来源：基于之前提供的 mvr_deploy.py 代码中的 MVRConfig 类。
    """

    # ==========================================
    # 1. 身份与路径
    # ==========================================
    name: str = "Mvr_22dof"

    # [请确认] 基于你之前代码中的路径
    xml_path: str = "/home/jf/lab/3_my_projects/assets/Mvr_22dof/Mvr_22dof.xml"
    policy_path: str = (
        "/home/jf/lab/3_my_projects/Project01_Isaaclab/logs/rsl_rl/Mvr_22dof_legs/2026-01-28_10-58-55/exported/policy.pt"
    )

    # ==========================================
    # 2. 仿真参数
    # ==========================================
    dt: float = 0.001  # 物理步长
    decimation: int = 10  # 控制分频
    lin_vel_scale: float = 1.0
    ang_vel_scale: float = 0.2
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    gait_period: float = 0.8

    # 维度自检
    # Obs 维度需根据实际训练确定，这里假设结构类似
    # 22 DoF: Pos(22) + Vel(22) + LastAct(22) = 66
    # AngVel(3) + Gravity(3) + Cmd(3) + Phase(2) = 11
    # Total approx = 77 (请根据实际 pt 文件核对)
    num_observations: int = 77
    num_actions: int = 22

    # ==========================================
    # 3. 关节定义 (The Truth)
    # ==========================================
    foot_names: list[str] = field(default_factory=lambda: ["R_foot_1", "L_foot_1"])
    policy_joint_names: list[str] = field(
        default_factory=lambda: [
            # Left Leg
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # Right Leg
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]
    )
    # 必须与 Isaac Lab 训练顺序严格一致
    joint_names: list[str] = field(
        default_factory=lambda: [
            # Left Leg
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # Right Leg
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # Torso & Head
            "waist_joint",
            "head_joint",
            # Left Arm
            "left_arm_pitch_joint",
            "left_arm_roll_joint",
            "left_arm_yaw_joint",
            "left_elbow_joint",
            # Right Arm
            "right_arm_pitch_joint",
            "right_arm_roll_joint",
            "right_arm_yaw_joint",
            "right_elbow_joint",
        ]
    )

    # ==========================================
    # 4. 物理参数 (全名显式定义)
    # ==========================================

    # --- A. 初始位置 (Nominal Pose) ---
    # 严格保留了你之前代码中左右不对称的设定
    default_dof_pos: dict[str, float] = field(
        default_factory=lambda: {
            # Left Leg
            "left_hip_pitch_joint": -0.25,
            "left_hip_roll_joint": -0.03,
            "left_hip_yaw_joint": -0.01,
            "left_knee_joint": -0.50,
            "left_ankle_pitch_joint": 0.23,
            "left_ankle_roll_joint": -0.01,
            # Right Leg (注意：部分符号翻转，部分未翻转，源自你的实际调试)
            "right_hip_pitch_joint": 0.25,
            "right_hip_roll_joint": 0.03,
            "right_hip_yaw_joint": -0.01,  # Yaw 保持负号
            "right_knee_joint": 0.50,
            "right_ankle_pitch_joint": -0.23,
            "right_ankle_roll_joint": 0.01,
            # Torso
            "waist_joint": 0.0,
            "head_joint": 0.0,
            # Left Arm
            "left_arm_pitch_joint": 0.0,
            "left_arm_roll_joint": 0.0,
            "left_arm_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            # Right Arm
            "right_arm_pitch_joint": 0.0,
            "right_arm_roll_joint": 0.0,
            "right_arm_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
        }
    )

    # --- B. 刚度 (Kp) ---
    kps: dict[str, float] = field(
        default_factory=lambda: {
            # Hip: Pitch=200, Roll=150, Yaw=100
            "left_hip_pitch_joint": 400.0,
            "left_hip_roll_joint": 200.0,
            "left_hip_yaw_joint": 200.0,
            "right_hip_pitch_joint": 400.0,
            "right_hip_roll_joint": 200.0,
            "right_hip_yaw_joint": 200.0,
            # Knee: 400
            "left_knee_joint": 400.0,
            "right_knee_joint": 400.0,
            # Ankle: Pitch=40, Roll=30
            "left_ankle_pitch_joint": 40.0,
            "left_ankle_roll_joint": 10.0,
            "right_ankle_pitch_joint": 40.0,
            "right_ankle_roll_joint": 10.0,
            # Torso: Waist=100, Head=20
            "waist_joint": 200.0,
            "head_joint": 20.0,
            # Arm: Pitch=40, Roll=40, Yaw=20, Elbow=20
            "left_arm_pitch_joint": 40.0,
            "left_arm_roll_joint": 40.0,
            "left_arm_yaw_joint": 20.0,
            "left_elbow_joint": 20.0,
            "right_arm_pitch_joint": 40.0,
            "right_arm_roll_joint": 40.0,
            "right_arm_yaw_joint": 20.0,
            "right_elbow_joint": 20.0,
        }
    )

    # --- C. 阻尼 (Kd) ---
    kds: dict[str, float] = field(
        default_factory=lambda: {
            # Hip: Pitch=10, Roll=5, Yaw=5
            "left_hip_pitch_joint": 10.0,
            "left_hip_roll_joint": 5.0,
            "left_hip_yaw_joint": 5.0,
            "right_hip_pitch_joint": 10.0,
            "right_hip_roll_joint": 5.0,
            "right_hip_yaw_joint": 5.0,
            # Knee: 10
            "left_knee_joint": 10.0,
            "right_knee_joint": 10.0,
            # Ankle: Pitch=2.5, Roll=2.0
            "left_ankle_pitch_joint": 2.5,
            "left_ankle_roll_joint": 0.2,
            "right_ankle_pitch_joint": 2.5,
            "right_ankle_roll_joint": 0.2,
            # Torso: Waist=5, Head=1
            "waist_joint": 5.0,
            "head_joint": 1.0,
            # Arm: Pitch=2, Roll=2, Yaw=1, Elbow=1
            "left_arm_pitch_joint": 2.0,
            "left_arm_roll_joint": 1.0,
            "left_arm_yaw_joint": 1.0,
            "left_elbow_joint": 1.0,
            "right_arm_pitch_joint": 2.0,
            "right_arm_roll_joint": 1.0,
            "right_arm_yaw_joint": 1.0,
            "right_elbow_joint": 1.0,
        }
    )

    # --- D. 动作缩放 (Action Scale) ---
    action_scales: dict[str, float] = field(
        default_factory=lambda: {
            name: 0.25
            for name in [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_joint",
                "head_joint",
                "left_arm_pitch_joint",
                "left_arm_roll_joint",
                "left_arm_yaw_joint",
                "left_elbow_joint",
                "right_arm_pitch_joint",
                "right_arm_roll_joint",
                "right_arm_yaw_joint",
                "right_elbow_joint",
            ]
        }
    )

    # --- E. 扭矩限制 (Torque Limits) ---
    torque_limits: dict[str, float] = field(
        default_factory=lambda: {
            # A10020 (150 Nm)
            "left_hip_pitch_joint": 150,
            "right_hip_pitch_joint": 150,
            "left_knee_joint": 150,
            "right_knee_joint": 150,
            # A8112 (90 Nm)
            "left_hip_roll_joint": 90,
            "right_hip_roll_joint": 90,
            "left_hip_yaw_joint": 90,
            "right_hip_yaw_joint": 90,
            "waist_joint": 90,
            # A6408 (60 Nm)
            "left_arm_pitch_joint": 60,
            "right_arm_pitch_joint": 60,
            "left_arm_roll_joint": 60,
            "right_arm_roll_joint": 60,
            # A4310 (36 Nm)
            "left_ankle_pitch_joint": 36,
            "right_ankle_pitch_joint": 36,
            "left_ankle_roll_joint": 36,
            "right_ankle_roll_joint": 36,
            "head_joint": 36,
            "left_arm_yaw_joint": 36,
            "right_arm_yaw_joint": 36,
            "left_elbow_joint": 36,
            "right_elbow_joint": 36,
        }
    )

    # --- F. 动作截断 ---
    clip_actions: dict[str, float] = field(
        default_factory=lambda: {
            name: 100.0
            for name in [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_joint",
                "head_joint",
                "left_arm_pitch_joint",
                "left_arm_roll_joint",
                "left_arm_yaw_joint",
                "left_elbow_joint",
                "right_arm_pitch_joint",
                "right_arm_roll_joint",
                "right_arm_yaw_joint",
                "right_elbow_joint",
            ]
        }
    )
