# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import time
from abc import ABC, abstractmethod

import mujoco
import mujoco.viewer
import torch
from config.base_config import BaseRobotConfig
from utils.engine import MujocoEngine
from utils.logger import DataLogger


class BaseRunner(ABC):
    """
    所有 Runner 的基类。

    职责:
    1. 初始化 Engine (MuJoCo) 和 Logger。
    2. 管理 Viewer 的生命周期 (启动/关闭)。
    3. 统一处理时间同步 (Sync) 和 异常捕获 (Ctrl+C)。
    """

    def __init__(self, config: BaseRobotConfig, log_path: str = "logs/robot_log.csv"):
        self.cfg = config

        # 1. 初始化物理引擎
        print(f"[Runner] Initializing Engine for {config.name}...")
        self.engine = MujocoEngine(
            xml_path=config.xml_path,
            dt=config.dt,
            joint_names=config.joint_names,
            foot_geom_names=config.foot_names,
        )

        # 2. 初始化日志
        self.logger = DataLogger(log_path)

        # 3. 设备管理 (虽然 MuJoCo 是 CPU，但策略通常涉及 Torch)
        self.device = torch.device("cpu")  # 部署通常强制使用 CPU

        # 4. 运行时状态
        self.step_counter = 0
        self.time_start = 0.0

    def run(self):
        """
        入口函数：启动 Viewer 并运行控制循环。
        """
        print("\n" + "=" * 50)
        print(f"🚀 Starting Runner: {self.__class__.__name__}")
        print(f"   Robot: {self.cfg.name}")
        print("   Press Ctrl+C to stop.")
        print("=" * 50 + "\n")

        # 重置机器人到初始状态
        self.reset()

        try:
            # 启动 Passive Viewer
            with mujoco.viewer.launch_passive(self.engine.model, self.engine.data) as viewer:
                self.time_start = time.time()

                # 进入子类定义的具体控制循环
                self._run_loop(viewer)

        except KeyboardInterrupt:
            print("\n[Runner] Stopped by user (Ctrl+C).")
        except Exception as e:
            print(f"\n[Runner] ❌ Runtime Error: {e}")
            raise e  # 重新抛出以便调试
        finally:
            # 确保退出时保存日志
            self.logger.save()
            print("[Runner] Cleanup done.")

    def reset(self):
        """
        通用的重置逻辑。
        子类如果需要特殊重置 (e.g. 随机化)，可以重写此方法。
        """
        # 将 Config 中的字典转换为有序数组
        # 注意: 这使用了我们在 BaseConfig 中定义的严格排序函数
        init_pos = self.cfg.parse_params_to_array(self.cfg.default_dof_pos)

        self.engine.reset(initial_dof_pos=init_pos)
        self.step_counter = 0
        print("[Runner] Robot reset to default pose.")

    def sync_viewer(self, viewer, step_start_time):
        """
        时间同步：确保仿真速度不超过实时速度。
        """
        viewer.sync()

        # 计算物理步长耗时 (Decimation * dt)
        step_dt = self.cfg.decimation * self.cfg.dt

        elapsed = time.time() - step_start_time
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

    @abstractmethod
    def _run_loop(self, viewer):
        """
        [抽象方法] 必须由子类实现。
        这里编写具体的 while viewer.is_running(): 循环。
        """
        pass
