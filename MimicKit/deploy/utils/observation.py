# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import torch
from utils.math_utils import quat_rotate_inverse

# ==========================================
# 原子化观测函数库 (Atomic Observation Functions)
# 每个函数只负责计算一个独立的观测分量
# ==========================================


def get_obs_ang_vel(base_ang_vel: torch.Tensor, scale: float) -> torch.Tensor:
    """
    计算基座角速度观测。
    Args:
        base_ang_vel: [1, 3] raw sensor data
        scale: scalar
    """
    return base_ang_vel * scale


def get_obs_gravity(base_quat: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    计算投影重力观测 (Projected Gravity)。
    逻辑: 将世界坐标系下的 [0, 0, -1] 旋转到基座坐标系。
    """
    gravity = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0)  # [1, 3]
    return quat_rotate_inverse(base_quat, gravity)


def get_obs_cmd(cmd: torch.Tensor, scale: float) -> torch.Tensor:
    """
    计算控制指令观测。
    Args:
        cmd: [1, 3] (vx, vy, omega)
        scale: scalar
    """
    return cmd * scale


def get_obs_dof_pos(dof_pos: torch.Tensor, default_dof_pos: torch.Tensor, scale: float) -> torch.Tensor:
    """
    计算关节位置观测 (残差形式)。
    Formula: (Current - Default) * Scale
    """
    return (dof_pos - default_dof_pos) * scale


def get_obs_dof_vel(dof_vel: torch.Tensor, scale: float) -> torch.Tensor:
    """
    计算关节速度观测。
    Formula: Current * Scale
    """
    return dof_vel * scale


def get_obs_last_action(last_action: torch.Tensor) -> torch.Tensor:
    """
    获取上一帧动作观测。
    (通常不需要处理，但为了保持接口一致性保留此函数)
    """
    return last_action


def get_obs_gait_phase(current_time: float, gait_period: float, device: torch.device) -> torch.Tensor:
    """
    计算步态相位观测 [sin, cos]。
    Formula: phase = (time % period) / period
    """
    if gait_period <= 0:
        return torch.zeros(1, 2, device=device)

    phase_scalar = (current_time % gait_period) / gait_period

    # 构造 [1, 2] 的 Tensor
    phase_obs = torch.zeros(1, 2, device=device)
    theta = phase_scalar * 2 * 3.141592653589793

    phase_obs[0, 0] = torch.sin(torch.tensor(theta))
    phase_obs[0, 1] = torch.cos(torch.tensor(theta))

    return phase_obs
