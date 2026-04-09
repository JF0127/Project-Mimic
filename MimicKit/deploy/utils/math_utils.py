# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import torch

# ==========================================
# 1. PyTorch 数学工具 (用于策略推理)
# ==========================================


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    将向量 v 旋转到四元数 q 的逆方向 (即：将世界坐标系的向量转到基座坐标系)。
    公式: v' = q_inverse * v * q

    Args:
        q: [w, x, y, z], shape (..., 4)
        v: [x, y, z], shape (..., 3)

    Returns:
        rotated vector, shape (..., 3)
    """
    # 提取分量 (支持 Batch 或 单个)
    q_w = q[..., 0]
    q_vec = q[..., 1:]

    # 扩展维度以支持广播 (Broadcasting)
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.matmul(q_vec, v.unsqueeze(-1)).squeeze(-1) * 2.0

    return a - b + c


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """
    将角度限制在 [-Pi, Pi] 范围内。
    通常用于处理 Heading Error (航向误差)。
    """
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi


# ==========================================
# 2. Numpy/Math 数学工具 (用于日志与调试)
# ==========================================


def get_rpy_from_quat(q: np.ndarray | list) -> tuple[float, float, float]:
    """
    将四元数 [w, x, y, z] 转换为 (Roll, Pitch, Yaw)。
    用于在 Log 中记录直观的姿态信息。

    Args:
        q: numpy array or list [w, x, y, z]

    Returns:
        (roll, pitch, yaw) in radians
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # 使用 copysign 处理数值稳定性 (sinp > 1 时)
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
