# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

# 1. 导入配置
from config.mvr_10dof import Mvr10DofConfig
from config.mvr_22dof import Mvr22DofConfig

# 2. 导入运行器
from runners.pd_runner import PDRunner
from runners.sim2sim_runner import Sim2SimRunner

# ==========================================
# 注册表 (Registry) - 新增机器人只需改这里
# ==========================================
ROBOT_MAP = {
    "Mvr_10dof": Mvr10DofConfig,
    "Mvr_22dof": Mvr22DofConfig,
}


def main():
    # ==========================================
    # 1. 命令行参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="MuJoCo Deployment Framework Entry Point")

    # 必选参数: 机器人名称
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        choices=ROBOT_MAP.keys(),
        help="选择机器人配置 (e.g., Mvr_10dof, Mvr_22dof)",
    )

    # 必选参数: 任务类型
    parser.add_argument(
        "--task",
        type=str,
        default="sim2sim",
        choices=["sim2sim", "pd", "single"],
        help="选择运行模式: sim2sim (推理), pd (站立测试), single (单关节测试)",
    )

    args = parser.parse_args()

    # ==========================================
    # 2. 实例化配置 (Config)
    # ==========================================
    print(f"[*] Loading Configuration for: {args.robot}")
    ConfigClass = ROBOT_MAP[args.robot]
    cfg = ConfigClass()

    # ==========================================
    # 3. 实例化运行器 (Runner)
    # ==========================================
    runner = None

    if args.task == "sim2sim":
        # --- Sim2Sim 模式 ---
        runner = Sim2SimRunner(config=cfg)

    elif args.task == "pd":
        # --- PD 站立测试模式 ---
        runner = PDRunner(config=cfg, mode="stand")

    # ==========================================
    # 4. 启动运行
    # ==========================================
    if runner is not None:
        runner.run()
    else:
        print("Error: Failed to initialize runner.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# ==========================================
# 💡 使用示例 (Usage Examples)
# ==========================================
#
# 1. 运行 Mvr10Dof 的 Sim2Sim 推理:
#    python main.py --robot Mvr_10dof --task sim2sim
#
# 2. 运行 Mvr22Dof 的 PD 站立测试 (检查 Kp/Initial Pos):
#    python main.py --robot Mvr_22dof --task pd
#
#
# 4. 查看帮助:
#    python main.py --help
# ==========================================
