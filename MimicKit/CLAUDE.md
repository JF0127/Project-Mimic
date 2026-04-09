## V47 22DOF 行走训练

包含 RMA、非对称观测、域随机化等完整特性的 AMP 行走控制方案。

### 关键特性

- **非对称观测**：Actor 187维（无特权），Critic 207维（全量）
- **域随机化**：Push robots，80N 外力，8s 间隔，作用于 `waist_link`
- **状态初始化**：`state_init_mode: "Real"`（真实站立姿态 + 逐关节噪声）
- **动作平滑**：`action_smooth_weight: 0.02` 限制时序跳变
- **物理边界惩罚**：`use_physical_bound_loss: True` 梯度逼迫遵守关节限位
- **统一优化器**：disc loss 集成进 PPO combined loss（`disc_loss_weight: 5`）

### 观测维度说明

机器人共 27 个 body（含 4 个虚拟末端 toe/hand），kinematic model 保留全部 26 个非根关节旋转：

| 分量 | 维度 | 说明 |
|------|------|------|
| root_rot (tan_norm) | 6 | 全局朝向 |
| root_ang_vel | 3 | 全局角速度 |
| joint_rot_obs | 156 | 26 joints × 6 |
| dof_vel | 22 | 关节速度 |
| **Actor 合计** | **187** | prop_obs，无特权 |
| root_height | 1 | 特权：高度 |
| root_vel | 3 | 特权：线速度 |
| key_pos | 15 | 特权：5关键体位置 |
| push_flag | 1 | 特权：外力扰动标志 |
| **Critic 合计** | **207** | full_obs，全量 |
| **Disc 合计** | **2080** | 208 × 10 步 |

---

## Play（测试已训练模型）

```bash
cd /home/jhl/projects/Project-Mimic/MimicKit
source .venv/bin/activate

# 方式一：arg_file（推荐）
python mimickit/run.py --arg_file args/amp_v47_walk_play_args.txt

# 方式二：完整命令
python mimickit/run.py \
  --mode test \
  --num_envs 1 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/amp_v47_walk_env.yaml \
  --agent_config data/agents/amp_v47_walk_agent.yaml \
  --model_file output/amp_mvr_22dof/2026-0402_211536/model.pt
```

---

## Train（继续训练）

```bash
cd /home/jhl/projects/Project-Mimic/MimicKit
source .venv/bin/activate

# 方式一：arg_file
python mimickit/run.py --arg_file args/amp_mvr_22dof_args.txt

# 方式二：完整命令
python mimickit/run.py \
  --mode train \
  --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/amp_v47_walk_env.yaml \
  --agent_config data/agents/amp_v47_walk_agent.yaml \
  --out_dir output/amp_v47_walk/ \
  --logger wandb
```

---

## 完整流程速查

```
GVHMR（视频）→ GMR（动作重定向）→ GMR/output/*.pkl
                                          ↓
                        gmr_to_mimickit.py（格式转换）
                                          ↓
                  data/motions/v47_self/GMR_VID_*.pkl
                                          ↓
          MimicKit AMP 训练（IsaacGym 引擎 + amp_v47_walk_env.yaml）
                                          ↓
                     output/amp_v47_walk/*.pt
```

---

## 关键文件索引

| 文件 | 用途 |
|------|------|
| `data/envs/amp_v47_walk_env.yaml` | AMP 行走训练环境配置（非对称观测、Real 初始化、Push 随机化） |
| `data/agents/amp_v47_walk_agent.yaml` | AMP 行走智能体超参数（SGD 5e-5、action_smooth、无 RMA） |
| `data/engines/isaac_gym_engine.yaml` | IsaacGym 引擎配置（控制频率 40Hz，仿真频率 200Hz） |
| `data/assets/v47/mjcf/v47_inertia_v3all_lmt_2.xml` | V47 22DOF 机器人 MuJoCo 模型（含惯性/限位优化） |
| `data/motions/v47_self/GMR_VID_20260330_204103.pkl` | V47 22DOF 行走动作文件（由 GMR 生成） |
| `args/amp_v47_walk_play_args.txt` | Play 启动参数文件 |
| `args/amp_mvr_22dof_args.txt` | Train 启动参数文件 |
| `output/amp_mvr_22dof/2026-0402_211536/model.pt` | 已训练的行走模型 checkpoint |
| `tools/gmr_to_mimickit/gmr_to_mimickit.py` | GMR pkl → MimicKit pkl 转换脚本 |
| `mimickit/run.py` | 训练/测试入口 |

---

## 核心代码改动说明

| 文件 | 主要变化 |
|------|---------|
| `mimickit/learning/base_agent.py` | 统一优化器、`_critic_obs_norm`、RMA/SE 标志 |
| `mimickit/learning/ppo_agent.py` | 集成 disc loss、action smooth/PD/physical bound 惩罚 |
| `mimickit/learning/amp_agent.py` | disc loss 合并到 combined loss，移除独立 disc optimizer |
| `mimickit/learning/amp_model.py` | RMA encoder 架构（含 Dropout） |
| `mimickit/learning/ppo_model.py` | critic 支持 `get_critic_obs_space()` fallback |
| `mimickit/envs/char_env.py` | `compute_char_obs` 返回4值、`init_pose_real`、action PD limit |
| `mimickit/envs/deepmimic_env.py` | 5种状态初始化、push robots 域随机化 |
| `mimickit/envs/amp_env.py` | `fix_root`/`joint_mask`、速度追踪奖励 |

---

## 常见问题

**Q: 运行时报 `IsaacSim not found`**  
A: 确认 IsaacSim pip 包已安装，或设置 `ISAACSIM_PATH` 环境变量指向本地 IsaacSim 安装目录。

**Q: `char_file` 路径找不到**  
A: 所有路径均相对于 `MimicKit/` 根目录，务必从该目录运行命令。

**Q: 显存不足 OOM**  
A: 降低 `--num_envs`（如 1024），或在 `amp_v47_walk_env.yaml` 中减小 `num_disc_obs_steps`。

**Q: 动作转换后关节数不匹配**  
A: 检查 GMR 使用的 IK 配置是否对应 V47 22DOF，确保 `dof_pos` 维度为 22。

**Q: play 时 critic 维度 warning**  
A: 正常现象。play 时仅用 actor，critic 维度差 1（push_robots 关闭）不影响推理。
