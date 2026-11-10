# UMI DP：相对回合起始 TCP 位姿训练

本文说明如何使用仓库中的 [训练脚本](../../scripts/train_dp_handumi_tcp.sh) 训练相对回合起始 TCP 的 Diffusion Policy（DP）模型。

核心约定：**数据导出时，已将每个回合的 TCP 位姿转换到该回合首帧 TCP 坐标系；训练时设置 `--policy.use_relative_pose=false`，保持状态和动作均为 7 维。**

## 1. 相对回合起始位是什么意思

每个 episode 单独选取首帧 TCP 作为固定参考系。对于切分后的 pick 回合，参考的是该 pick 回合的首帧，而不是整段原始录制的首帧。TCP 指夹爪抓取中心，不能直接用相机、追踪器或机器人法兰位姿代替。

设 `T_W_TCP(t)` 是时刻 `t` 的 TCP 在原始参考系 `W` 下的齐次变换，`T_W_TCP(0)` 是该回合首帧位姿，则存入数据集的位姿为：

```text
T_start_TCP(t) = inverse(T_W_TCP(0)) × T_W_TCP(t)

R_relative(t) = transpose(R_0) × R_t
p_relative(t) = transpose(R_0) × (p_t - p_0)
```

将 `R_relative` 转为旋转向量（轴角），与位置和夹爪值拼成：

```text
[x, y, z, rx, ry, rz, gripper]
```

位置单位为米，旋转向量单位为弧度。变换同时影响平移和旋转，不能仅减去起点 xyz 或直接相减轴角。首帧的前 6 维应接近 0；夹爪值独立保留，不参与 SE(3) 变换。

这里的 action 是固定起始坐标系中的**目标位姿**，不是逐帧位姿增量，不能把连续预测结果累加到当前 TCP 上。

### 与 `umi_dp训练.md` 的区别

| 项目 | 本文：相对回合起始 TCP | [原文档](./umi_dp训练.md)：相对当前帧 |
|---|---|---|
| 参考系 | 每个回合首帧 TCP，整回合固定 | 当前观测帧 TCP，随时刻变化 |
| 变换发生阶段 | 数据导出阶段 | 策略前处理阶段 |
| `policy.use_relative_pose` | `false` | `true` |
| 策略内部状态／动作表示 | 7 维：xyz＋轴角＋夹爪 | 10 维：xyz＋rot6d＋夹爪 |
| 输出后处理 | 反归一化后仍在起始 TCP 坐标系 | 相对位姿后处理器转回输入参考系 |

`use_relative_pose=false` 只表示不启用策略内部的“相对当前帧”变换，并不表示数据使用世界坐标系。若改为 `true`，就改变了模型的表示和任务定义，无法复现本次训练。

## 2. 数据与环境准备

在仓库根目录安装训练依赖：

```bash
cd /home/scl/work/lerobot
uv sync --locked --extra diffusion --extra training
```

需要可用的 CUDA/PyTorch 环境，以及视频解码所需的 FFmpeg、torchcodec；训练脚本默认使用两张 NVIDIA GPU。

本文主要使用已有的 `data/handumi_data/umi-demo-tcp-dp-240_134`。脚本也支持 `data/handumi_data/umi-demo-tcp-dp`，但本次对该版本只做了短程兼容验证，没有完成同等规模的训练对照。

| 项目 | `umi-demo-tcp-dp-240_134` | `umi-demo-tcp-dp` |
|---|---|---|
| 图像 `observation.images.front` | 宽 240 × 高 134 像素，RGB | 宽 224 × 高 224 像素，RGB |
| 回合、帧数及帧率 | 200 个回合、53597 帧，30 fps | 200 个回合、53597 帧，30 fps |
| state / action | 均为 7 维 `xyz_rotvec_gripper` | 均为 7 维 `xyz_rotvec_gripper` |
| TCP 和夹爪 | 抓取中心，夹爪归一化到 0～1 | 相同 |

两者使用相同的轨迹数据，图像尺寸及预处理方式不同。脚本根据目录名生成 `repo_id`，例如 `handumi/umi-demo-tcp-dp-240_134`，并使用数据集图像的原始分辨率。

训练前检查数据集元信息：

```bash
cat data/handumi_data/umi-demo-tcp-dp-240_134/meta/info.json
cat data/handumi_data/umi-demo-tcp-dp-240_134/meta/fastumi.json
```

`info.json` 描述维度、帧数和图像；`fastumi.json` 记录坐标系和导出约定，关键字段应为：

```json
{
  "pose_frame": "episode_start_tcp",
  "end_frame": "tcp",
  "tcp_definition": "gripper_grasp_center",
  "state_layout": "xyz_rotvec_gripper",
  "action_layout": "xyz_rotvec_gripper",
  "position_unit": "m",
  "rotation_unit": "rad"
}
```

换用其他数据集时，应在导出阶段按第 1 节公式转换 state 和 action，并重新生成统计量。仅修改 `pose_frame` 元数据不会转换数据；训练脚本本身不会执行坐标转换。

默认配置按任务随机划分回合，划分种子为 `1000`，`eval_split=0.1`。该单任务数据集分为 **180 个训练回合和 20 个验证回合**；实际回合编号保存在输出的 `split_manifest.json` 中。原始数据集和现有归一化统计量保持不变；归一化统计量来自整份数据，不是仅由训练集重新计算的统计量。

## 3. 启动训练与查看曲线

### 3.1 选择训练配方

如果本机已有第 4 节列出的最佳 EMA 检查点，优先核对部署坐标并进行真实任务对比，无需为了使用该模型重新从头训练。需要新训练时，在仓库根目录执行：

```bash
# 默认 regularized：小 U-Net、颜色增强和 EMA。
bash scripts/train_dp_handumi_tcp.sh data/handumi_data/umi-demo-tcp-dp-240_134

# 复现 baseline：大 U-Net、不使用图像增强和 EMA。
bash scripts/train_dp_handumi_tcp.sh data/handumi_data/umi-demo-tcp-dp-240_134 --recipe baseline
```

第一个参数也可以是绝对路径。需要使用 224×224 数据时，将数据目录替换为 `data/handumi_data/umi-demo-tcp-dp`。两种配方的输出目录分别为：

```text
outputs/train/dp_handumi_tcp_regularized_<启动时间>/
outputs/train/dp_handumi_tcp_baseline_<启动时间>/
```

也可以从 [训练 YAML](./dp_handumi_tcp_train.yaml) 启动双卡训练：

```bash
bash scripts/train_dp_handumi_tcp.sh --config_path docs/umi/dp_handumi_tcp_train.yaml
```

示例保留颜色增强和 EMA，并设置 `policy.n_obs_steps: 3`、`policy.down_dims: [512, 1024, 2048]` 和每卡 `batch_size: 192`。在 YAML 中修改数据集和训练参数，`dataset.root` 的相对路径以仓库根目录为基准。脚本为两个进程统一设置输出目录 `outputs/train/dp_handumi_tcp_yaml_<启动时间>/`。

训练参数以所用配方或 YAML 以及实际检查点的 `train_config.json` 为准；调整参数时应同步确认验证协议，避免把不同设置产生的 loss 直接比较。

### 3.2 关键参数

| 参数 | 默认 regularized 的取值与含义 |
|---|---|
| `use_relative_pose=false` | 保持数据集的起始 TCP 坐标和轴角表示；state/action 使用 MIN_MAX 归一化 |
| `n_obs_steps=2` | 输入当前帧及前一帧观测；回合起点缺少历史时重复首帧 |
| `horizon=16` | 预测 `t-1` 到 `t+14` 的 16 帧动作 |
| `n_action_steps=8` | 从当前时刻起执行其中 8 步，再重新预测；30 Hz 下约 0.27 秒 |
| `drop_n_last_frames=7` | 不从回合末尾 7 帧采样，减少动作块补齐；不代表所有动作窗口都无补齐 |
| `down_dims=[256,512,1024]` | 小 U-Net；baseline 为 `[512,1024,2048]` |
| 图像增强 | GPU 上对亮度、对比度使用 `[0.8,1.2]`，饱和度使用 `[0.9,1.1]` 的温和 `ColorJitter`；不做几何增强。baseline 关闭增强 |
| EMA | 使用 warmup EMA；`eval_use_ema=true`，以 EMA 验证损失选择 `best`。baseline 不启用 EMA |
| 扩散调度 | DDPM，`num_train_timesteps=100`；本次检查点推理沿用 100 步 |
| `batch_size=128` | 每卡 128、双卡总有效 batch 为 256，梯度累积为 1 |
| `steps=19466` | 总训练步数；Adam 学习率 `1e-4`、weight decay `1e-6`，cosine 调度、warmup 500 步 |
| `eval_steps=500`、`eval_seed=1000` | 每 500 步以固定种子计算留出回合损失；各数据并行进程使用种子加进程序号 |
| `save_freq=2000` | 每 2000 步及最终步更新 `last`；创新低时更新 `best`，仅保留受管理的最佳及最新检查点 |
| `env_eval_freq=0` | 关闭环境 rollout；留出数据集 loss 验证仍然启用 |
| `trackio.enable=true` | 记录训练与验证指标，默认项目为 `lerobot` |

两种配方均使用 ImageNet 预训练 ResNet18，不启用混合精度。脚本不指定额外的图像 resize 或 crop；本次 240×134 检查点的 `resize_shape=null`、`crop_shape=null`，`crop_ratio=1.0`。其他检查点应以各自的 `config.json` 和保存的处理器为准，不能假定所有模型都使用 224×224 输入。

颜色增强只用于训练，留出回合验证和独立离线评估均不使用该增强。

如需调整学习率，应使用 `--policy.optimizer_lr`；默认训练预设会覆盖单独设置的 `--optimizer.lr`。不要额外加载 [dp_fastumi_config.json](./dp_fastumi_config.json) 来复现本次训练：该文件使用不同的通道数、DDIM 和扩散步数。

选择其他数据集或硬件时，可用以下公式估算数据遍历次数；尾帧采样和 batch 取整会带来差异，仍应核对训练日志：

```text
总有效 batch = 每卡 batch × GPU 数 × 梯度累积步数
近似 epoch 数 = steps × 总有效 batch / 训练集帧数
```

`19466` 是当前脚本的配置值，并非适用于所有数据集的最佳步数。

### 3.3 查看训练曲线

```bash
uv run --extra training trackio show --project lerobot --host 0.0.0.0
```

打开命令输出的网址并选择对应的 `dp_handumi_tcp-<短 ID>`。指标默认保存在训练用户的 `~/.cache/huggingface/trackio`，不在训练输出目录中。

- `train/loss`：每 200 步记录的训练损失。
- `eval/live_eval_loss`：当前训练权重的验证损失。
- `eval/ema_eval_loss`：默认 regularized 配方的 EMA 验证损失。
- `eval/eval_loss`：用于选择 `best` 的验证损失；regularized 取 EMA loss，baseline 取 live loss，每 500 步记录一次。

验证固定种子为 1000。它衡量的是扩散噪声预测误差，不直接等于位姿误差或任务成功率。

## 4. 检查点、离线评估与恢复

### 4.1 找到最佳模型

默认配方的输出结构如下；`best` 指向验证损失最低的步数，`last` 指向最新定期或最终检查点：

```text
outputs/train/dp_handumi_tcp_regularized_<启动时间>/
├── split_manifest.json
├── evaluation_protocol.json
└── checkpoints/
    ├── <best 或 last 步数>/
    │   ├── split_manifest.json
    │   ├── evaluation_protocol.json
    │   ├── pretrained_model/           # 当前训练权重、配置及处理器
    │   ├── pretrained_model_ema/       # EMA 权重、配置及处理器
    │   └── training_state/              # 优化器、调度器、EMA 及随机状态等
    ├── best -> <最佳步数>
    └── last -> <最新步数>
```

两个 `pretrained_model*` 目录也分别携带划分清单和验证协议。best 与 last 可能指向同一步；脚本不会自动将终端输出写入 `train.log`。

- 默认 regularized 用于推理或导出的目录：`checkpoints/best/pretrained_model_ema`。
- baseline 用于推理或导出的目录：`checkpoints/best/pretrained_model`；该配方没有 EMA 目录。

本机本次实验已训练完成，最佳 EMA 目录为：

```text
outputs/train/dp_handumi_tcp_regularized_2026-11-06_10-32-15/checkpoints/best/pretrained_model_ema
```

该路径指向第 **4000** 步；它是本机训练产物，其他机器或新克隆的仓库不一定包含。加载时保留完整模型目录及处理器配置与统计量，并核对 `config.json` 中 `use_relative_pose=false`、state/action 的 shape 均为 `[7]`。

### 4.2 独立离线评估

对本机已有模型复评时，可在仓库根目录运行，输出目录请按需要更换：

```bash
uv run --locked --extra diffusion --extra training python scripts/eval_dp_handumi_tcp.py \
  --checkpoint outputs/train/dp_handumi_tcp_regularized_2026-11-06_10-32-15/checkpoints/best/pretrained_model_ema \
  --output-dir "outputs/eval/handumi_regularized_$(date +%Y-%m-%d_%H-%M-%S)" \
  --device cuda:0
```

也可传入 `checkpoints/best`：工具会根据保存的 `eval_use_ema` 自动选择相应权重。工具使用保存的验证回合清单，对全量有效验证帧分别以种子 `0、1、2` 计算噪声预测 loss；每个回合固定选取 8 个窗口，以相同三个种子执行 100 步 DDPM 采样，比较实际执行的 8 步（标签片段 `action[:, 1:9]`）。结果分别输出位置欧氏误差（米）、旋转测地角误差（弧度）和夹爪绝对误差，并包含逐步、逐回合及逐种子统计。输出目录中的 `results.json` 为结果，`manifest.json` 记录检查点、回合、窗口和采样配置。

本机完整对照的训练过程使用种子 1000 选择最佳步数：baseline 的最佳步数为 5000，regularized 为 4000。随后使用上面的**三个种子独立复评**：

| 指标 | baseline | regularized |
|---|---:|---:|
| 验证 loss，三种子均值 | 0.012763 | 0.010282，降低约 19.4% |
| 位置误差均值 | 0.01169 m | 0.01129 m |
| 旋转误差均值 | 0.02691 rad | 0.02636 rad |
| 夹爪绝对误差均值 | 0.00444 | 0.00433 |

优化后验证曲线在最佳步数之后仍有回升；总体均值改善不代表每个窗口都改善，夹爪误差的总体 p95 在本次对照中略有上升。详情见本机实验产物 [验收报告](../../outputs/experiments/handumi_overfit_20261106_9ka3l821/acceptance_report.json) 和 [验证曲线](../../outputs/experiments/handumi_overfit_20261106_9ka3l821/validation_curves.png)；这些输出不会随普通仓库克隆提供。

本次仅完成离线评估，**尚未验证真实机器人任务成功率**。下一步先按第 5 节核对起始坐标、TCP 标定和动作转换，再以同一任务条件比较已有最佳 EMA 与 baseline 的实机表现。

### 4.3 恢复中断的训练

从 `last/pretrained_model/train_config.json` 加载 live 权重，并保留同级的 `training_state/`；EMA 权重会从训练状态恢复。将下面目录替换为实际中断的训练目录：

```bash
RUN_DIR="outputs/train/dp_handumi_tcp_regularized_2026-11-06_10-32-15"
uv run --locked --extra diffusion --extra training torchrun --standalone --nproc-per-node=2 \
  "$(uv run --locked --extra diffusion --extra training which lerobot-train)" \
  --config_path="$RUN_DIR/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true
```

`pretrained_model_ema` 是默认配方的推理和导出目录，不应替代上述 live 检查点用于续训。划分、评估种子、验证权重或 batch／并行拓扑等评价协议发生变化时，不应沿用旧 `best` 损失比较；请使用原协议恢复或另起训练。达到配置总步数后，使用原配置恢复不会再增加训练步数；延长训练需要另行安排总步数和学习率调度。

## 5. 推理时必须保持同一坐标约定

每次开始新的机器人回合时，记录机器人控制参考系 `B` 下的起始 TCP 位姿 `T_B_start`，整回合保持不变：

```text
实时观测转为模型输入：
T_start_current = inverse(T_B_start) × T_B_current

模型输出经后处理器反归一化，得到起始坐标系中的目标 T_start_target。

目标转回机器人参考系：
T_B_target = T_B_start × T_start_target
```

输入旋转和输出旋转都按轴角与旋转矩阵互转；夹爪单独处理。机器人侧必须使用与训练一致的抓取中心 TCP 定义，并通过部署标定处理工具偏移。开始新回合时还需要重置策略的观测历史和动作队列。

本模型的后处理器**不会自动完成上面的起始坐标系到机器人参考系的转换**。现有 `dp_infer_lerobot` 不能直接复用来控制该模型；需要适配 7 维轴角表示、固定回合起点和 TCP 标定。旧的 10 维相对当前帧检查点也不能与本模型互换。

如需导出 ONNX／TensorRT，可参考 [UMI 模型部署](./模型部署.md) 并替换为本文的检查点路径；导出模型后，机器人侧仍需实现上述坐标转换。

`lerobot-export` 会读取检查点 `train_config.json` 中的 `dataset.root`，将训练数据的 `meta/fastumi.json` 原样复制为导出目录下的 `fastumi.json`，与 `encoder.onnx`、`denoiser.onnx` 同级，并纳入 `manifest.json` 的文件摘要校验。部署时一并保留该文件，以核对 `pose_frame=episode_start_tcp`、TCP 定义、状态／动作布局和单位。若训练配置未记录数据路径或源文件不存在，导出会警告并跳过复制；导出前应确保训练数据路径仍可访问。
