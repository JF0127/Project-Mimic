# CLAUDE.md — MimicKit: MVR 22DOF AMP 训练全流程

本文件描述以 **MVR_22DOF 机器人** 为目标，基于 **GMR 生成的动作数据**，使用 **MimicKit + IsaacLab** 训练 AMP（Adversarial Motion Priors）控制器的完整流程。使用 `uv` 作为包管理器。

---

## 项目结构

```
Project-Mimic/
├── MimicKit/                         # 本目录：RL 训练框架
│   ├── mimickit/                     # 核心库（AMP/ASE/AWR 等算法）
│   ├── data/
│   │   ├── assets/mvr_22dof/         # MVR 22DOF MuJoCo 模型
│   │   ├── envs/amp_mvr_22dof_env.yaml   # AMP 环境配置
│   │   ├── agents/amp_humanoid_agent.yaml # AMP 智能体配置（可复用）
│   │   ├── engines/isaac_lab_engine.yaml  # IsaacLab 引擎配置
│   │   └── motions/mvr_22dof/        # 放置转换后的动作文件（需手动创建）
│   ├── tools/gmr_to_mimickit/        # GMR → MimicKit 格式转换工具
│   └── output/                       # 训练输出（模型、日志）
├── GMR/
│   └── output/jhl_marktime.pkl       # GMR 已生成的动作文件（起点）
└── thirdparty/IsaacLab/              # IsaacLab 子模块
```

---

## 第一步：环境配置（uv + IsaacLab）

### 1.1 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 1.2 创建虚拟环境（Python 3.10）

```bash
cd /home/jhl/projects/Project-Mimic/MimicKit
uv venv .venv --python 3.10
source .venv/bin/activate
```

### 1.3 安装 IsaacSim（通过 pip wheel）

IsaacLab 需要先安装 IsaacSim。使用 NGC pip 源：

```bash
uv pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121

uv pip install \
  isaacsim-rl \
  isaacsim-replicator \
  isaacsim-extscache-physics \
  isaacsim-extscache-kit \
  isaacsim-extscache-kit-sdk \
  --extra-index-url https://pypi.nvidia.com
```

> **注意**：若使用的是 IsaacSim 4.x，包名可能为 `isaacsim==4.x.x`，请以 [NVIDIA IsaacLab 安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 为准。本项目测试的 IsaacLab commit 为 `2ed331acfcbb1b96c47b190564476511836c3754`。

### 1.4 安装 IsaacLab

```bash
cd /home/jhl/projects/Project-Mimic/thirdparty/IsaacLab
# 以 editable 模式安装所有 IsaacLab 扩展
./isaaclab.sh --install  # 或手动：uv pip install -e .
```

若 `isaaclab.sh` 不适用，可手动安装：

```bash
uv pip install -e /home/jhl/projects/Project-Mimic/thirdparty/IsaacLab
```

### 1.5 安装 MimicKit 依赖

```bash
cd /home/jhl/projects/Project-Mimic/MimicKit
uv pip install -r requirements.txt
```

---

## 第二步：动作数据转换（GMR → MimicKit）

GMR 已生成 `GMR/output/jhl_marktime.pkl`，需将其转换为 MimicKit 格式并放置到正确目录。

### 2.1 创建动作目录

```bash
mkdir -p /home/jhl/projects/Project-Mimic/MimicKit/data/motions/mvr_22dof
```

### 2.2 运行格式转换脚本

从 `MimicKit/` 根目录运行：

```bash
cd /home/jhl/projects/Project-Mimic/MimicKit

python tools/gmr_to_mimickit/gmr_to_mimickit.py \
  --input_file  ../GMR/output/jhl_marktime.pkl \
  --output_file data/motions/mvr_22dof/mvr_marktime.pkl \
  --loop wrap
```

**参数说明：**
- `--loop wrap`：动作循环播放（适合 marktime 踏步类动作）
- `--loop clamp`：动作播放到末帧后停止
- `--start_frame` / `--end_frame`：可选，裁剪动作片段
- `--output_fps`：可选，重采样帧率（默认保持原帧率）

转换成功后会打印动作帧数、FPS 等信息。

### 2.3 （可选）可视化验证动作

用 `view_motion` 环境验证转换结果：

```bash
python mimickit/run.py \
  --mode test \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/view_motion_mvr_env.yaml \
  --visualize true
```

> 若无 `view_motion_mvr_env.yaml`，可参考 `data/envs/view_motion_humanoid_env.yaml` 复制并修改 `char_file` 和 `motion_file` 字段。

---

## 第三步：确认配置文件

### 3.1 环境配置（已存在）

`data/envs/amp_mvr_22dof_env.yaml`：

```yaml
env_name: "amp"
char_file: "data/assets/mvr_22dof/Mvr_22dof.xml"
episode_length: 10.0
motion_file: "data/motions/mvr_22dof/mvr_marktime.pkl"   # ← 第二步生成的文件

key_bodies: ["head_link", "left_ankle_roll_link", "right_ankle_roll_link",
             "left_hand_link", "right_hand_link"]
contact_bodies: ["left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
                 "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link"]

reward_pose_w: 0.5        # 关节姿态奖励权重
reward_vel_w: 0.1         # 速度奖励权重
reward_root_pose_w: 0.15  # 根节点姿态奖励权重
reward_root_vel_w: 0.1    # 根节点速度奖励权重
reward_key_pos_w: 0.15    # 关键点位置奖励权重
```

> 若将动作文件放在不同路径，修改 `motion_file` 字段。

### 3.2 机器人模型（已存在）

确认 MVR 22DOF 机器人模型已就绪：

```bash
ls data/assets/mvr_22dof/
# 应包含：Mvr_22dof.xml、Mvr_22dof.urdf、meshes/
```

若 `data/assets/mvr_22dof/` 不存在，需将 GMR 的机器人模型链接过来：

```bash
ln -s /home/jhl/projects/Project-Mimic/GMR/assets/mvr_22dof \
      /home/jhl/projects/Project-Mimic/MimicKit/data/assets/mvr_22dof
```

### 3.3 智能体配置

使用通用 AMP 智能体配置（`data/agents/amp_humanoid_agent.yaml`）即可，MVR 22DOF 无需单独 agent yaml。

---

## 第四步：AMP 训练

从 `MimicKit/` 根目录运行，确保 `.venv` 已激活：

```bash
cd /home/jhl/projects/Project-Mimic/MimicKit
source .venv/bin/activate

python mimickit/run.py \
  --mode train \
  --num_envs 4096 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_mvr_22dof_env.yaml \
  --agent_config data/agents/amp_humanoid_agent.yaml \
  --visualize false \
  --out_dir output/amp_mvr_22dof_marktime/ \
  --logger tb
```

**关键参数：**
- `--num_envs 4096`：并行环境数，显存不足时可降低（如 1024/2048）
- `--visualize false`：训练时关闭渲染以加速
- `--out_dir`：模型和日志保存目录
- `--logger tb`：使用 TensorBoard 记录训练曲线（也支持 `wandb`、`txt`）

### 监控训练

```bash
tensorboard --logdir=output/amp_mvr_22dof_marktime/ --port=6006 --samples_per_plugin scalars=999999
```

浏览器打开 `http://localhost:6006`。

---

## 第五步：测试训练结果

```bash
python mimickit/run.py \
  --mode test \
  --num_envs 4 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_mvr_22dof_env.yaml \
  --agent_config data/agents/amp_humanoid_agent.yaml \
  --visualize true \
  --model_file output/amp_mvr_22dof_marktime/<model_file>.pt
```

---

## 分布式训练（多 GPU）

```bash
python mimickit/run.py \
  --mode train \
  --num_envs 4096 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_mvr_22dof_env.yaml \
  --agent_config data/agents/amp_humanoid_agent.yaml \
  --visualize false \
  --out_dir output/amp_mvr_22dof_marktime/ \
  --devices cuda:0 cuda:1
```

---

## 完整流程速查

```
GVHMR（视频）→ GMR（动作重定向）→ GMR/output/jhl_marktime.pkl
                                          ↓
                        gmr_to_mimickit.py（格式转换）
                                          ↓
                  data/motions/mvr_22dof/mvr_marktime.pkl
                                          ↓
          MimicKit AMP 训练（IsaacLab 引擎 + amp_mvr_22dof_env.yaml）
                                          ↓
                     output/amp_mvr_22dof_marktime/*.pt
```

---

## 关键文件索引

| 文件 | 用途 |
|------|------|
| `data/envs/amp_mvr_22dof_env.yaml` | AMP 环境配置（奖励权重、机器人、动作文件路径） |
| `data/agents/amp_humanoid_agent.yaml` | AMP 智能体超参数（网络结构、学习率等） |
| `data/engines/isaac_lab_engine.yaml` | IsaacLab 引擎配置（控制频率 30Hz，仿真频率 120Hz） |
| `data/assets/mvr_22dof/Mvr_22dof.xml` | MVR 22DOF 机器人 MuJoCo 模型 |
| `tools/gmr_to_mimickit/gmr_to_mimickit.py` | GMR pkl → MimicKit pkl 转换脚本 |
| `mimickit/run.py` | 训练/测试入口 |

---

## 常见问题

**Q: 运行时报 `IsaacSim not found`**  
A: 确认 IsaacSim pip 包已安装，或设置 `ISAACSIM_PATH` 环境变量指向本地 IsaacSim 安装目录。

**Q: `char_file` 路径找不到**  
A: 所有路径均相对于 `MimicKit/` 根目录。务必从该目录运行命令，或使用绝对路径。

**Q: 显存不足 OOM**  
A: 降低 `--num_envs`（如 1024），或在 `amp_mvr_22dof_env.yaml` 中减小 `num_disc_obs_steps`。

**Q: 动作转换后关节数不匹配**  
A: 检查 GMR 使用的 IK 配置是否为 `smplx_to_mvr_22dof.json`，确保 `dof_pos` 维度为 22。
