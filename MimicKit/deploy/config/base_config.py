# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BaseRobotConfig:
    """
    机器人配置基类
    职责：定义必要参数，并提供将字典参数按 joint_names 顺序转换为数组的功能。
    """

    # ==========================================
    # 1. 基础信息
    # ==========================================
    name: str = "robot_base"
    xml_path: str = ""  # 必须是绝对路径
    policy_path: str = ""  # 必须是绝对路径

    # ==========================================
    # 2. 仿真与控制
    # ==========================================
    dt: float = 0.001
    decimation: int = 10

    lin_vel_scale: float = 1.0
    ang_vel_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 1.0
    gait_period: float = 0.0

    # 维度自检
    num_observations: int = 0
    num_actions: int = 0

    # ==========================================
    # 3. 关节定义 (核心真理)
    # ==========================================
    # 必须是完整全名 (e.g. "Left_knee_joint")
    # 且顺序必须与 Isaac Lab 训练时的顺序严格一致
    joint_names: list[str] = field(default_factory=list)
    foot_names: list[str] = field(default_factory=list)
    policy_joint_names: list[str] = field(default_factory=list)
    # ==========================================
    # 4. 物理参数 (字典存储)
    # ==========================================
    # 必须使用与 joint_names 一致的全名作为 Key
    default_dof_pos: dict[str, float] = field(default_factory=dict)
    kps: dict[str, float] = field(default_factory=dict)
    kds: dict[str, float] = field(default_factory=dict)

    action_scales: dict[str, float] = field(default_factory=dict)
    clip_actions: dict[str, float] = field(default_factory=dict)
    torque_limits: dict[str, float] = field(default_factory=dict)

    # ==========================================
    # 5. 核心工具
    # ==========================================
    def __post_init__(self):
        if self.xml_path and not os.path.isabs(self.xml_path):
            print(f"[Warning] xml_path 建议使用绝对路径: {self.xml_path}")

    def parse_params_to_array(self, param_dict: dict[str, float], default_value: float = 0.0) -> np.ndarray:
        """
        [严格排序]
        依据 self.joint_names 的顺序，从 param_dict 中提取对应的值。
        不进行任何模糊匹配，Key 必须完全相等。
        """
        ordered_list = []
        missing_keys = []

        for name in self.joint_names:
            if name in param_dict:
                ordered_list.append(param_dict[name])
            else:
                # 记录缺失，稍后报错，保证鲁棒性
                ordered_list.append(default_value)
                missing_keys.append(name)

        if missing_keys:
            # 这里抛出异常比打印警告更好，强制用户去 Config 里把名字写对
            raise ValueError(f"[Config Error] 参数字典中缺少以下关节定义: {missing_keys}")

        return np.array(ordered_list, dtype=np.float32)
