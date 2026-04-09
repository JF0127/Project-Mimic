# 🤖 MuJoCo Sim2Sim Deployment Framework

这是一个用于将 **Isaac Lab** 训练的强化学习 (RL) 策略部署到 **MuJoCo** 物理引擎中的轻量级、模块化仿真框架。

该框架旨在填补 "训练 (Training)" 与 "实机部署 (Sim2Real)" 之间的空白，提供高精度的物理验证、电机 PD 参数调试以及关节映射自检功能。

## ✨ 核心特性

* **⚡️ 双模架构 (Dual-Mode)**: 完美兼容 `PyTorch Tensor` (策略层) 与 `Numpy` (物理层) 的数据流。
* **🛡️ 运动学下降 (Kinematic Descent)**:
* **[NEW] 隐形电梯启动**: 摒弃了不稳定的“重力补偿”或“轨迹插值”。
* **上帝模式 (God Mode)**: 启动时将机器人基座 (Base) 的 XY 位置与姿态 (IMU) 强行锁定，只允许 Z 轴以恒定速度（如 -0.2m/s）匀速下降。
* **智能触地检测**: 引擎实时监测脚部 Geom 的接触力，**双脚触地瞬间**自动解锁，无缝切换至物理引擎接管。这允许机器人在空中就将关节刚度 (KP) 拉满，以最佳姿态落地。


* **🔧 硬件自检 (Hardware Check)**:
* **PD Stand**: 验证满额 KP/KD 下的稳态性能，包含详细的关节力矩分解 (KP项 vs KD项)。


* **📊 原子化观测 (Atomic Observations)**: 观测构造函数解耦，支持模块化组装，严格对齐 Isaac Lab 的观测顺序。
* **📈 深度数据分析**: 内置内存级 `DataLogger` 和 `viz.py` 可视化工具，新增对 **Ankle Roll/Pitch** 等关键关节的 `Pos`, `Torque`, `KP_term`, `KD_term` 的详细波形记录。

---

## 📂 项目结构

```text
project_root/
│
├── main.py                     # [入口] 统一命令行入口
│
├── config/                     # [配置层] 机器人参数定义
│   ├── base_config.py          # 基类 (处理数组解析、类型转换)
│   ├── mvr_10dof.py            # Mvr10Dof 机器人配置
│   └── mvr_22dof.py            # Mvr22Dof 机器人配置
│
├── runners/                    # [控制层] 核心运行逻辑
│   ├── base_runner.py          # 基类 (自动初始化带触地检测的 Engine)
│   ├── pd_runner.py            # [NEW] 集成运动学下降的 PD 测试
│   └── sim2sim_runner.py       # RL策略推理 & 状态机管理
│
├── engines/                    # [核心] 物理引擎封装
│   └── mujoco_engine.py        # [NEW] 实现 State Override 和 Contact Detection
│
├── utils/                      # [工具层] 底层驱动与数学库
│   ├── observation.py          # 原子化观测构造函数
│   ├── math_utils.py           # 四元数、欧拉角转换
│   ├── logger.py               # CSV 自动记录器
│   └── viz.py                  # 日志可视化与绘图脚本
│
├── logs/                       # [数据] 运行时产生的 CSV 日志
└── results/                    # [输出] 可视化图片保存目录

```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境依赖

确保已安装以下库：

```bash
pip install mujoco torch numpy pandas matplotlib

```

### 2. 运行 Sim2Sim 推理 (Sim2Sim Task)

加载训练好的 Policy，让机器人执行动作。

```bash
python main.py --robot mvr22 --task sim2sim

```

### 3. 运行 PD 站立测试 (PD Stand Task)

**[更新]** 使用运动学下降法测试站立稳定性。

```bash
python main.py --robot mvr22 --task pd

```

* **启动现象**: 机器人出现在空中 (0.8m)，**双腿瞬间伸直** (Full KP)，像坐电梯一样匀速下降。
* **落地逻辑**: 脚尖触地瞬间，终端显示 `[LANDED]`，机器人由物理引擎完全接管。
* **用途**: 检查 PD 参数是否过硬（导致触地反弹）或过软（导致跪地）。

### 4. 运行单关节调试 (Single Joint Task)

验证电机顺序和方向。机器人会被完全“钉”在空中（位置和姿态均锁定），只有指定的关节会按正弦波摆动。

```bash
# 测试左脚踝 Roll 关节
python main.py --robot mvr22 --task single --joint left_ankle_roll_joint

```

---

## 📊 数据可视化 (Visualization)

仿真结束后，使用 `utils/viz.py` 分析日志。这对于 Sim2Real 极其重要。

1. 修改 `utils/viz.py` 中的 `csv_path`。
2. 运行脚本：

```bash
python utils/viz.py

```

**新增分析功能：**

* **Torque Breakdown**: 针对脚踝等关键关节，可以分别查看 **KP项 (位置误差贡献)** 和 **KD项 (速度阻尼贡献)** 的曲线。这有助于判断是否是因为 D 增益过大导致了震荡。

---

## ⚙️ 配置指南 (How to Configure)

### 添加新机器人

1. 在 `config/` 下新建文件，继承 `BaseRobotConfig`。
2. **[关键]** 填写 `joint_names`：必须与 Isaac Lab 训练时的顺序一致。
3. **[NEW]** 填写 `foot_names`：
* 这是用于触地检测的 **Geom Name** (不是 Joint Name!)。
* 请在 XML 文件中找到脚底的 `<geom name="L_foot_1" ...>`。
* 建议左右脚各选一个主要的接触点即可。



```python
# 示例配置
self.foot_names = ["L_foot_1", "R_foot_1"]

```

4. 定义 `kps`, `kds`, `default_dof_pos` 等参数。

---

## ⚠️ 常见问题

1. **机器人一直悬在空中下降，穿过地面也不停**:
* **原因**: 触地检测失效。
* **解决**: 检查 Config 中的 `foot_names` 是否与 XML 中的 `<geom name="...">` 完全一致。注意 XML 中有些 geom 可能没有定义 name，需要手动加上。


2. **机器人落地瞬间剧烈弹跳或炸飞**:
* **原因**: PD 参数 (`kps`) 过大，或者物理仿真步长 (`dt`) 不够小。
* **解决**: 减小 KP，或在 `viz.py` 中查看触地瞬间的 `Torque` 峰值。


3. **机器人落地后“软腿”直接跪下**:
* **原因**: 下降时我们使用了“上帝模式”锁定姿态，掩盖了 KP 不足的问题。触地解锁后，真实的支撑力不足。
* **解决**: 增大 KP，或检查 `default_dof_pos` 是否合理。


4. **报错 `ValueError: ... not found in XML**`:
* 检查 Joint Name 或 Geom Name 拼写。



---

### Maintainer

* **Author**: JF
* **Date**: 2026-01-27 (Updated: Kinematic Descent)
* **Status**: Active Development
