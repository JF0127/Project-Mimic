# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause


import mujoco
import numpy as np


class MujocoEngine:
    """
    MuJoCo 硬件抽象层 (支持运动学下降与触地检测).
    """

    def __init__(
        self,
        xml_path: str,
        dt: float,
        joint_names: list,
        foot_geom_names: list[str] = None,
    ):
        print(f"[Engine] Loading MuJoCo model from: {xml_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise ValueError(f"Failed to load MuJoCo XML: {e}")

        # 设置物理步长
        self.model.opt.timestep = dt

        # ---------------------------------------------------------
        # 建立索引映射
        # ---------------------------------------------------------
        self.joint_names = joint_names
        self.joint_indices = []
        self.dof_vel_indices = []
        self.actuator_indices = []

        print(f"[Engine] Building index mapping for {len(joint_names)} joints...")
        for name in joint_names:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id == -1:
                raise ValueError(f"[Engine Error] XML 中找不到关节: '{name}'")

            self.joint_indices.append(self.model.jnt_qposadr[j_id])
            self.dof_vel_indices.append(self.model.jnt_dofadr[j_id])

            a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a_id == -1:
                raise ValueError(f"[Engine Error] XML 中找不到执行器: '{name}'")
            self.actuator_indices.append(a_id)

        self.joint_indices = np.array(self.joint_indices, dtype=np.int32)
        self.dof_vel_indices = np.array(self.dof_vel_indices, dtype=np.int32)
        self.actuator_indices = np.array(self.actuator_indices, dtype=np.int32)

        # 传感器
        self.imu_quat_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
        self.imu_gyro_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")

        # ---------------------------------------------------------
        # [关键] 注册脚部 Geom ID (用于触地检测)
        # ---------------------------------------------------------
        self.foot_geom_ids = []
        if foot_geom_names:
            print(f"[Engine] Registering foot geoms for contact: {foot_geom_names}")
            for name in foot_geom_names:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid == -1:
                    print(f"[Warning] Geom '{name}' not found in XML!")
                else:
                    self.foot_geom_ids.append(gid)

        # 运动学下降状态
        self.is_descent_mode = False
        self.descent_vel = 0.0
        self.locked_xy = np.zeros(2)

    # ========================================================
    # 触地检测与下降逻辑
    # ========================================================
    def check_feet_contact(self) -> bool:
        """检查所有注册的脚部 Geom 是否都接触到了物体"""
        if not self.foot_geom_ids:
            return True

        contacted_feet = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if g1 in self.foot_geom_ids:
                contacted_feet.add(g1)
            if g2 in self.foot_geom_ids:
                contacted_feet.add(g2)

        # 只有当所有注册的脚都产生接触时，才返回 True
        return len(contacted_feet) == len(self.foot_geom_ids)

    def start_kinematic_descent(self, height: float, descent_speed: float):
        """开启恒速下降模式"""
        self.is_descent_mode = True
        self.descent_vel = descent_speed

        # 瞬间移动 + 锁定状态
        self.data.qpos[2] = height
        self.locked_xy = self.data.qpos[0:2].copy()
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # 重置姿态
        self.data.qvel[:] = 0.0
        self.data.qvel[2] = descent_speed

        mujoco.mj_forward(self.model, self.data)

    def _process_kinematic_override(self):
        """在 Step 前调用，强制覆盖状态"""
        if not self.is_descent_mode:
            return

        # 1. 检查触地
        if self.check_feet_contact():
            print("\n[Engine] Ground Contact Detected! Releasing kinematic lock.")
            self.is_descent_mode = False
            self.data.qvel[2] = 0.0  # 触地瞬间速度清零
            return

        # 2. 状态覆写 (God Mode)
        self.data.qpos[0:2] = self.locked_xy
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[0:2] = 0.0
        self.data.qvel[3:6] = 0.0
        self.data.qvel[2] = self.descent_vel  # 强制 Z 轴速度

    # ========================================================
    # 核心接口
    # ========================================================
    def reset(self, initial_dof_pos: np.ndarray, base_pos=None):
        mujoco.mj_resetData(self.model, self.data)
        if base_pos is None:
            base_pos = np.array([0.0, 0.0, 0.6])

        self.data.qpos[0:3] = base_pos
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.joint_indices] = initial_dof_pos
        self.data.qvel[:] = 0.0
        self.is_descent_mode = False
        mujoco.mj_forward(self.model, self.data)

    def get_sensors(self):
        if self.imu_quat_idx != -1:
            q_adr = self.model.sensor_adr[self.imu_quat_idx]
            g_adr = self.model.sensor_adr[self.imu_gyro_idx]
            base_quat = self.data.sensordata[q_adr : q_adr + 4]
            base_ang_vel = self.data.sensordata[g_adr : g_adr + 3]
        else:
            base_quat = self.data.qpos[3:7].copy()
            base_ang_vel = self.data.qvel[3:6].copy()

        dof_pos = self.data.qpos[self.joint_indices]
        dof_vel = self.data.qvel[self.dof_vel_indices]
        return base_quat.copy(), base_ang_vel.copy(), dof_pos.copy(), dof_vel.copy()

    def step(self, target_dof_pos, kp, kd, torque_limits):
        # 1. 处理下降逻辑
        self._process_kinematic_override()

        # 2. 计算 PD
        current_pos = self.data.qpos[self.joint_indices]
        current_vel = self.data.qvel[self.dof_vel_indices]
        kp_torque = kp * (target_dof_pos - current_pos)
        kd_torque = -kd * current_vel
        raw_torque = kp_torque + kd_torque
        applied_torque = np.clip(raw_torque, -torque_limits, torque_limits)

        self.data.ctrl[self.actuator_indices] = applied_torque

        # 3. 物理步进
        mujoco.mj_step(self.model, self.data)
        return applied_torque, kp_torque, kd_torque
