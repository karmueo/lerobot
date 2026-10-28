# UMI 数据训练 Diffusion Policy（SE(3) 相对位姿）

本文说明如何用 FastUMI 转换得到的 LeRobot 数据集训练 Diffusion Policy（DP），并按 UMI 的做法开启 SE(3) 相对位姿。文中还整理了各参数的含义和调参建议。

## 1. 前提条件

数据集需要满足：

| 项目 | 要求 |
|---|---|
| 格式 | LeRobot v3 数据集（包含 `meta/info.json`、`data/`、`videos/`） |
| `observation.state` | 7 维 `[x, y, z, rx, ry, rz, gripper]`，位置单位为米，旋转用轴角表示（单位为弧度） |
| `action` | 布局和 `observation.state` 相同，都是绝对位姿 |
| 图像 | 例如 `observation.images.front`，224×224 |

可以用下面的命令确认数据集的维度和帧数：

```bash
cat <数据集root>/meta/info.json
```

状态向量里，每个位姿组占 6 维（xyz + 轴角）。`relative_pose_state_starts` 指定每个位姿组的起始下标。单臂、位姿在最前面时，写 `'[0]'`。

## 2. 训练命令

双卡，每张卡 batch 128，约 120 个 epoch：

```bash
uv run torchrun --nproc-per-node=2 $(uv run which lerobot-train) \
  --dataset.repo_id=scl/jingbao \
  --dataset.root=/home/scl/work/UMI/FastUMI_Data/dataset/processed/lerobot_jingbao/lerobot \
  --policy.type=diffusion \
  --policy.use_relative_pose=true \
  --policy.relative_pose_state_starts='[0]' \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.drop_n_last_frames=7 \
  --policy.resize_shape='[224,224]' \
  --policy.crop_ratio=0.95 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/dp_umi_v2 \
  --job_name=dp_umi_v2 \
  --batch_size=128 \
  --steps=18500 \
  --save_freq=0 \
  --wandb.disable_artifact=true
```

单卡训练时，把开头换成 `uv run lerobot-train`，其余参数不变。

### 2.1 使用配置文件调整 DP 模型

从仓库根目录执行训练命令时，可以加入以下参数，加载 [dp_fastumi_config.json](./dp_fastumi_config.json)：

```bash
--config_path=docs/umi/dp_fastumi_config.json \
```

该文件将 U-Net 通道数设为 `[256, 512, 1024]`，并使用 DDIM、50 个训练扩散时间步和 16 个推理去噪步。它只配置这些策略参数，上面的数据集、相对位姿、时序和训练参数仍需保留。`--config_path` 中的相对路径以执行命令时所在的目录为准；从其他目录启动时，可改用配置文件的绝对路径。

### 2.2 如何算出 `--steps`

```
总 batch  = batch_size × GPU 数 × 梯度累积步数
epoch 数  = steps × 总 batch / 数据集总帧数
steps     = 目标 epoch 数 × 数据集总帧数 / 总 batch
```

例如 38848 帧，双卡，每卡 batch 128，目标 120 个 epoch：120 × 38848 / 256 ≈ 18200，取 18500。

训练日志中的 `epch:` 字段就是当前已训练的 epoch 数，可以用来核对。

### 2.3 训练输出

```
outputs/train/<job>/
├── checkpoints/
│   ├── 018500/
│   │   ├── pretrained_model/   # 推理用：模型权重、config、前后处理器（含归一化统计量）
│   │   └── training_state/     # 续训用：优化器、调度器、随机数状态
│   └── last -> 018500          # 软链接，指向最新的 checkpoint
└── wandb/offline-run-*/        # wandb 离线日志
```

- 如果 `--output_dir` 已经存在，训练会直接报错。每次训练要换一个新目录；或者加 `--resume=true`，接着上次的进度继续训练。
- 本仓库已将 wandb 改为默认开启、离线模式。训练结束后用下面的命令上传：
  ```bash
  uv run wandb sync outputs/train/<job>/wandb/offline-run-*
  ```
  想实时在线查看，加 `--wandb.mode=online`；不想记录，加 `--wandb.enable=false`。

## 3. 参数说明与调参建议

### 3.1 数据集

| 参数 | 本例取值 | 含义 | 调参建议 |
|---|---|---|---|
| `--dataset.repo_id` | `scl/jingbao` | 数据集名称。数据在本地时只起标识作用 | 随意命名，但要和数据集内容对应 |
| `--dataset.root` | 本地路径 | 数据集根目录，也就是包含 `meta/` 的那一层 | — |
| `--dataset.eval_split` | 不设（0） | 每个任务留出的回合比例，取最后若干个回合，用作验证集 | 调参阶段建议设 `0.1`；确定参数后，最终模型去掉此项，用全部数据训练 |
| `--dataset.image_transforms.enable` | 不设（false） | 训练时随机调整亮度、对比度、饱和度、色调、锐度，并做小幅仿射变换 | 数据少，或实机光照和采集时不同时，建议设为 `true`，可以缓解过拟合 |

### 3.2 相对位姿（UMI 方式）

| 参数 | 本例取值 | 含义 | 调参建议 |
|---|---|---|---|
| `--policy.use_relative_pose` | `true` | 以当前帧为参考，把状态历史和动作块转换为 SE(3) 相对位姿，旋转用 rot6d 表示，因此 7 维变为 10 维。推理时后处理器会自动转回绝对位姿 | UMI 类数据（位姿在采集起点坐标系下）建议开启 |
| `--policy.relative_pose_state_starts` | `'[0]'` | `observation.state` 中各个位姿组的起始下标 | 单臂写 `'[0]'`；双臂例如 `'[0,7]'`，按实际布局填写 |
| `--policy.relative_pose_action_starts` | 不设 | `action` 中各个位姿组的起始下标。不设时和 state 相同 | 只有 action 布局和 state 不同时才需要设置 |
| `--policy.relative_pose_stats_stride` | 不设（1） | 训练前在相对空间重新计算归一化统计量时，每隔多少帧采样一次 | 数据量很大、统计耗时太长时，可以设为 `5`～`10` |

### 3.3 时序结构

| 参数 | 本例取值 | 含义 | 调参建议 |
|---|---|---|---|
| `--policy.n_obs_steps` | `2` | 输入几帧观测（当前帧加上前 1 帧） | UMI 和 DP 论文都用 2，一般不用改 |
| `--policy.horizon` | `16` | 每次预测的动作序列长度，从第一帧观测开始计 | 需要是 8 的倍数，因为 U-Net 会下采样 3 次。常用 16 |
| `--policy.n_action_steps` | `8` | 每次推理后实际执行多少步，必须满足 ≤ `horizon - n_obs_steps + 1` | 越大动作越连贯，但对扰动反应越慢。30 fps 时 8 步约 0.27 秒。机器人控制频率较低时，可以对应减小 |
| `--policy.drop_n_last_frames` | `7` | 每个回合末尾不作为训练起点的帧数，避免动作块大量补齐 | 取 `horizon - n_action_steps - n_obs_steps + 1`，本例为 7；改了前面三个参数时要同步修改 |

### 3.4 图像

| 参数 | 本例取值 | 含义 | 调参建议 |
|---|---|---|---|
| `--policy.resize_shape` | `'[224,224]'` | 输入图像先缩放到这个尺寸 | 和数据集图像尺寸一致时，相当于不缩放 |
| `--policy.crop_ratio` | `0.95` | 训练时随机裁剪（这里是 212×212），推理时中心裁剪 | 0.9～0.95。值越小增强越强，但边缘信息丢得越多 |
| `--policy.vision_backbone` | 默认 `resnet18` | 视觉编码器，使用 ImageNet 预训练权重 | 数据少时保持 resnet18 即可 |

### 3.5 优化与训练规模

| 参数 | 本例取值 | 含义 | 调参建议 |
|---|---|---|---|
| `--batch_size` | `128` | **每张卡**的 batch 大小 | 5090（32GB）上，每卡 128 实测约用 10.4GB 显存。每卡 192 也能放下，256 有 OOM 风险 |
| `--steps` | `18500` | 总优化步数。学习率按余弦曲线在这些步内衰减到 0 | 按 2.2 节的公式从目标 epoch 数换算。见第 4 节的经验数据 |
| `--policy.optimizer_lr` | 默认 `1e-4` | 学习率 | **注意**：默认会使用 policy 自带的训练预设，`--optimizer.lr` 会被覆盖、不起作用，所以改学习率要用这个参数。把总 batch 加大好几倍时，可以试 `2e-4` |
| `--policy.scheduler_warmup_steps` | 默认 `500` | 学习率预热步数 | 一般不用改 |
| `--save_freq` | `0` | 每隔多少步保存一次 checkpoint。设为 0 时只在最后保存一次 | 调参阶段建议设 `2000`，以便比较不同步数的效果；正式训练可以设为 0，或设得比较稀疏 |
| `--eval_steps` | 不设（0） | 每隔多少步在验证集上计算一次 `eval_loss`。需要先设 `eval_split` | 调参阶段建议设 `1000` |
| `--num_workers` | 默认 `4` | 每张卡的数据加载进程数 | 如果日志中 `data_s` 明显偏大，说明数据加载拖慢了训练，可以调大 |
| `--accelerator.mixed_precision` | 默认 `no` | 混合精度 | 可以试 `bf16` 来提速和省显存，但需要先确认 loss 曲线正常 |

### 3.6 其他

| 参数 | 本例取值 | 含义 |
|---|---|---|
| `--policy.device` | `cuda` | 训练设备。多卡训练时由 torchrun 自动分配 |
| `--policy.push_to_hub` | `false` | 不上传到 Hugging Face Hub |
| `--output_dir` / `--job_name` | — | 输出目录和任务名，`job_name` 同时也是 wandb 的 run 名 |
| `--wandb.disable_artifact` | `true` | 不把 checkpoint 作为 wandb artifact 保存，避免重复占用磁盘 |

## 4. 经验数据（jingbao 数据集，127 个回合，38848 帧）

两张 RTX 5090，每卡 batch 128，18500 步，吞吐约 2000 样本/秒，用时约 40 分钟。

`dp_umi_v2` 的设置为 `eval_split=0.1`，即用 114 个回合训练、留出 13 个回合验证：

| 步数（epoch） | 1K（7） | 2K（15） | 6K（45） | 10K（74） | 14K（104） | 18K（134） |
|---|---|---|---|---|---|---|
| 训练 loss | 0.022 | 0.013 | ~0.008 | ~0.006 | ~0.004 | 0.003 |
| eval_loss | 0.021 | **0.016** | 0.023 | 0.031 | 0.042 | 0.053 |

最终 checkpoint 的离线动作误差，取 8 步动作块的末步：

| | 训练回合 | 留出回合 | 原地不动作为对照 |
|---|---|---|---|
| 位置误差 | 2.7 mm | 10.1 mm | ~26 mm |
| 旋转误差 | 0.47° | 1.68° | ~2.5° |

结论：数据量较少时，训练 100 多个 epoch 会明显过拟合。eval_loss 在约 15 个 epoch 时最低，之后持续上升；留出回合上的误差约为训练回合的 4 倍。

需要说明的是，Diffusion Policy 的 eval_loss 和实机成功率的相关性比较弱，原论文中也有过验证 loss 上升、成功率仍在提高的情况。因此 eval_loss 只能作为参考，最终还是以实机测试为准。

## 5. 推荐的调参流程

1. **调参阶段**：设 `--dataset.eval_split=0.1 --eval_steps=1000 --save_freq=2000`，开启 `--dataset.image_transforms.enable=true`。训练完成后，在留出回合上逐个评估 checkpoint，找出误差最小的步数范围。
2. **正式训练**：去掉 `eval_split`，用全部回合训练，`--steps` 按上一步找到的 epoch 数换算。
3. **实机验证**：用 `checkpoints/last/pretrained_model` 部署。如果有多个候选 checkpoint，都在实机上对比一下。
4. **数据优先**：如果过拟合明显，增加采集回合数（UMI 类任务通常需要几百个回合），效果通常比调任何超参数都好。

## 6. 常见问题

- **`output_dir` 已存在而报错**：换一个新的输出目录，或者加 `--resume=true`，接着上次的进度继续训练。
- **改了 `--optimizer.lr` 但学习率没变**：要改用 `--policy.optimizer_lr`，见 3.5 节。
- **找不到 `wandb` 命令**：wandb 装在项目的虚拟环境里，要用 `uv run wandb ...` 运行。
- **多卡时学习率和步数会不会自动调整**：不会。总 batch 等于每卡 batch 乘以 GPU 数，`--steps` 需要按 2.1 节的公式重新计算。
