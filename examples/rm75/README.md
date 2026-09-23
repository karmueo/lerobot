# RM75 玩具机器人工作流：DexGraspVLA Zarr → LeRobot → SmolVLA 微调 → 离线验证

RM75 玩偶微调任务的端到端工作流：

1. **转换**：将 DexGraspVLA 夹爪 Zarr 演示数据（RM75 玩具机器人任务）转换为 LeRobot v3 数据集。
2. **微调**：在 2x RTX 5090 上微调 SmolVLA（纯 DDP，BF16）。
3. **验证**：在固定条件下离线验证所有检查点，并选出最佳的一个。

原始源数据（`/home/scl/work/UMI/DexGraspVLA/data/gripper_zarr` 及其图像缓存）**严格只读** ——
每个 zarr 存储都以 `mode="r"` 打开，任何内容都不会写入其中。所有派生数据均写入
`data/rm75_toy_robot_lerobot` 和 `outputs/`。

| | |
|---|---|
| 源类别 | `Target`、`Target2`、`Target3`、`Target4`、`Target5`（仅严格名称匹配） |
| 预期规模 | 600 个 episode · 267,847 帧 · 50 Hz · 单任务 |
| 任务指令 | `Pick up the toy robot and place it in the pink bowl.` |
| 特征 | `observation.images.gripper`（普通 RGB 视频，无掩码）· `observation.state` · `action` |
| 状态 / 动作 | 8 维 float32（7 关节 + 夹爪），保持原始 `[-1, 1]` 语义，**不做重新归一化** |
| 数据集元数据 | `robot_type=rm75`，`fps=50` |
| 划分 | 按类别确定的 90/10 划分 → 540 训练 / 60 评估，评估 episode 置于数据集末尾（与 `dataset.eval_split=0.1` 兼容） |

## 环境要求

- 可用的 LeRobot 开发环境（`uv sync --locked`），且 PATH 中有 `ffmpeg`（用于视频编码/解码）。
- `zarr>=2.18,<3` 是**一次性转换依赖** —— 由 `uv run --with` 注入，不会添加到 LeRobot 的核心依赖中。

## 步骤 1 — 转换 Zarr 源数据

```bash
uv run --with "zarr>=2.18,<3" examples/rm75/convert_dexgraspvla_to_lerobot.py \
    --source-zarr /home/scl/work/UMI/DexGraspVLA/data/gripper_zarr \
    --output-root data/rm75_toy_robot_lerobot
```

预期的源数据布局（实际的 DexGraspVLA 采集数据）：

```
DexGraspVLA/data/
├── gripper_zarr/                     # 一个只读的 Zarr 组
│   ├── data/state                    # [507671, 8]，float32
│   ├── data/action                   # [507671, 8]，float32
│   ├── data/gripper_image_paths      # 每个控制步一条相对 RGB 路径
│   ├── meta/episode_ends             # 1200 个 episode 边界
│   └── meta/episode_folders          # 例如 source_000/Target4/episode_0
└── gripper_frames/                   # 被引用的 JPEG 帧（只读）
```

转换器会选择倒数第二级文件夹名精确等于五个目标类别之一的 episode。它将每条图像路径解析为
相对于 `gripper_zarr` 的路径，用 PIL `convert("RGB")` 转换 JPEG，并有意忽略掩码路径。源数据
包含 200 个 `Target` episode 以及 `Target2` 到 `Target5` 各 100 个 episode；因此按每个类别留出
每第十个 episode，可得到 540 个训练和 60 个类别均衡的评估 episode。评估 episode 按类别交错
排列并放在数据集末尾，这样 `dataset.eval_split=0.1` 选中的恰好就是这批数据。

常用参数：

| 参数 | 作用 |
|---|---|
| `--dry-run` | 仅做发现 + 全量计数预检；不写入任何内容 |
| `--expected-episodes` / `--expected-frames` | 全量计数预检（默认 600 / 267847，设为 `0` 禁用） |
| `--resume` | 继续未完成的转换（见下文） |
| `--image-writer-threads` / `--image-writer-processes` | 视频编码并发度 |
| `--repo-id` | 写入元数据的数据集 id（默认 `rm75_toy_robot`） |

**清单与断点续转。** `manifest.json` 写在输出根目录，记录源根目录、图像缓存、划分规则、
每个 episode 的条目（源 episode、类别、帧数、train/eval）以及 `completed` 列表。不带
`--resume` 时，非空的输出目录会被**拒绝**（绝不静默覆盖）。带 `--resume` 时，只有当现有清单
与当前源数据**完全**一致（相同的 episode、帧数和划分）时才会继续转换 —— 否则中止，并要求
你删除输出目录后重新转换。

## 步骤 2 — 在 2x RTX 5090 上微调 SmolVLA

```bash
./examples/rm75/train_smolvla_rm75_5090x2.sh
```

关键设置（完整列表见脚本头部注释）：

- 通过 `torchrun` 使用纯 DDP（2 个进程），BF16 混合精度，针对 Blackwell/5090 的 NCCL 设置
  （`NCCL_P2P_DISABLE=1`、`NCCL_CUMEM_ENABLE=0` 等 —— 每一项均可通过环境变量覆盖）。
- 从 `lerobot/smolvla_base` 初始化。默认**全量微调**（`RM75_FREEZE_VISION=false`、
  `RM75_TRAIN_EXPERT_ONLY=false`）：视觉编码器可适配相机域偏移，但在 600 条演示下
  过拟合/遗忘风险更高。`SMOLVLA_BASE` 可指向本地目录，避免从 Hub 下载初始权重。
- **对照实验（SmolVLA 论文默认方案：冻结视觉编码器 + 仅训练 expert）**，其余配置完全一致，
  输出目录不要与全量微调相同：

  ```bash
  RM75_FREEZE_VISION=true RM75_TRAIN_EXPERT_ONLY=true \
      SMOLVLA_OUTPUT=outputs/rm75_smolvla_frozen \
      ./examples/rm75/train_smolvla_rm75_5090x2.sh
  ```

  两臂以留出集评估损失曲线 + 实机 rollout 对比定夺；冻结视觉保留预训练表征、显存更少，
  但无法适配明显的域偏移。
- **局部解冻实验**：冻结视觉编码器，同时训练 action expert、图像 connector 和最后 N 个
  VLM 文本层。建议先试 `N=2`，避免全量文本层更新带来的过拟合：

  ```bash
  RM75_FREEZE_VISION=true RM75_TRAIN_EXPERT_ONLY=false RM75_TRAIN_VLM_LAYERS=2 \
      SMOLVLA_OUTPUT=outputs/rm75_smolvla_vfrozen_last2 \
      ./examples/rm75/train_smolvla_rm75_5090x2.sh
  ```

  `RM75_TRAIN_VLM_LAYERS=0` 保持原有两种模式不变；该变量大于 0 时必须设置
  `RM75_TRAIN_EXPERT_ONLY=false`。
- `chunk_size=64`（训练目标长度），`n_action_steps=16`（仅影响部署：每次推理消费的动作数，
  50 Hz 下开环约 0.32 s 后重规划；不影响训练损失，训练 chunk 由 `chunk_size` 推导。
  最终值按控制延迟与实机表现调整，推理时可经 `--policy.n_action_steps` 覆盖而无需重训）。
- 每 GPU batch 32 x 2 GPU x 梯度累积 1 = **有效 batch 64**（与 SmolVLA 训练预设的假定批
  一致，学习率无需调整）；30,000 次迭代 = 30,000 次优化器更新 ≈ 8 个 epoch。实测单卡
  batch 32 峰值显存约 22 GB / 32 GB，稳态约 3.3 step/s，总墙钟约 2.5–3 小时。
- 学习率 1e-4，warmup 1,000，余弦衰减 30,000 —— 由 SmolVLA 训练预设提供
  （`use_policy_training_preset=true`，与 `SmolVLAConfig` 中的值完全一致）。
- 每 5,000 步保存一次检查点并做留出集评估（`save_freq=5000`、`eval_steps=5000`、
  `max_eval_samples=2048`；默认梯度累积为 1，步数即优化器更新次数）；评估使用
  `dataset.eval_split=0.1` 留出的 episode。
- 不做图像增广、不用 EMA、不上传 Hub；默认启用 W&B 记录训练指标，但不上传
  W&B artifact。检查点保存在本地
  `outputs/rm75_smolvla_5090x2/checkpoints/<step>/pretrained_model` 下。

环境变量覆盖：`RM75_LEROBOT_DATASET`、`SMOLVLA_BASE`、`SMOLVLA_OUTPUT`、`SMOLVLA_PORT`、
`RM75_PER_GPU_BATCH`、`RM75_GRAD_ACCUM`、`RM75_FREEZE_VISION`、`RM75_TRAIN_EXPERT_ONLY`、
`RM75_TRAIN_VLM_LAYERS`
（以及 `SMOLVLA_MASTER_ADDR`、`RM75_NUM_WORKERS`）；脚本末尾追加的 CLI 参数会原样传递给
`lerobot-train`，例如恢复一次中断的训练：

```bash
./examples/rm75/train_smolvla_rm75_5090x2.sh --resume=true \
    --config_path=outputs/rm75_smolvla_5090x2/checkpoints/000010000/pretrained_model
```

**OOM 备选方案。** 如果每 GPU batch 32 显存不足，可减半 batch、加倍梯度累积。注意
`--steps`、`--save_freq`、`--eval_steps` 都按训练循环迭代数计数（每迭代消费一个微批次，
优化器每 N 个迭代更新一次），因此要一同翻倍，才能维持 30,000 次优化器更新、完整余弦衰减
以及“每 5,000 次更新保存/评估”的节奏：

```bash
RM75_PER_GPU_BATCH=16 RM75_GRAD_ACCUM=2 ./examples/rm75/train_smolvla_rm75_5090x2.sh \
    --steps=60000 --save_freq=10000 --eval_steps=10000
```

（不翻倍的话只剩 15,000 次更新、衰减走不完，保存/评估也会密一倍。）

## 步骤 3 — 离线检查点验证

```bash
uv run examples/rm75/validate_checkpoints.py \
    --train-output outputs/rm75_smolvla_5090x2 \
    --dataset-root data/rm75_toy_robot_lerobot \
    --repo-id rm75_toy_robot
```

对 `checkpoints/` 下的每个检查点，脚本会计算：

- 在留出 episode 的**固定窗口**上计算的验证 flow-matching 损失，使用**固定**的
  flow-matching 噪声和时间步（只播种一次并在所有检查点间复用，因此各损失值可直接比较），
- 64x8 动作块的 MAE，含总体和逐维度（归一化动作空间，忽略尾部填充），
- 预测动作落在 `[-1, 1]` 之外的比例，以及 NaN / Inf 计数，
- 检查点加载时间以及每个动作块的平均推理时间。

固定条件下损失最低的检查点胜出；结果写入 `validation_report.json`，最佳检查点的路径写入
`best_checkpoint.txt`。检查点只被读取 —— 绝不删除任何内容。

```bash
# 变量覆盖 + 显式指定检查点子集
uv run examples/rm75/validate_checkpoints.py --train-output outputs/rm75_smolvla_5090x2 \
    --dataset-root data/rm75_toy_robot_lerobot --num-samples 512 --batch-size 8 --device cuda:1 \
    --checkpoints outputs/rm75_smolvla_5090x2/checkpoints/000015000 \
                  outputs/rm75_smolvla_5090x2/checkpoints/000030000
```

> **注意：** 离线验证衡量的是检查点对留出演示数据的预测能力，**并不**代表真实机器人上的
> 成功率。部署前请务必在真实的 RM75 机械臂上确认所选检查点的效果。
