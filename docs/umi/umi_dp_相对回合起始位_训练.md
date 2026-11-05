# UMI DP：相对回合起始 TCP 位姿训练

本文说明如何训练 `outputs/train/dp_pick_episode_start_tcp` 这类 Diffusion Policy（DP）模型。参数依据仓库中的 [训练脚本](../../scripts/train_dp_pick_episode_start_tcp.sh) 和该次训练最终检查点的 `train_config.json`。

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

在仓库根目录执行：

```bash
cd /home/scl/work/lerobot
uv sync --locked --extra diffusion --extra training
```

需要可用的 CUDA/PyTorch 环境，以及视频解码所需的 FFmpeg、torchcodec。本文双卡命令要求两张可用的 NVIDIA GPU。

本次训练使用已经转换好的本地数据：

| 项目 | 本次取值 |
|---|---|
| 数据集路径 | `/home/scl/work/UMI/FastUMI_Data/dataset/pick_place/lerobot/pick` |
| `repo_id` | `fastumi/pick` |
| 格式 | LeRobot v3 |
| 数据规模 | 302 个回合、44212 帧，30 fps |
| `observation.state` | 7 维，`xyz_rotvec_gripper` |
| `action` | 7 维，`xyz_rotvec_gripper`，来源为示范轨迹 |
| 图像 | `observation.images.front`，224×224 RGB |
| TCP 定义 | `gripper_grasp_center` |
| 夹爪 | 本数据集为 binary 模式，使用 0～1 表示 |

训练前检查以下两个文件：

```bash
cat /home/scl/work/UMI/FastUMI_Data/dataset/pick_place/lerobot/pick/meta/info.json
head -n 28 /home/scl/work/UMI/FastUMI_Data/dataset/pick_place/lerobot/pick/meta/fastumi.json
```

`info.json` 描述维度、帧数和图像；`fastumi.json` 记录本数据集的坐标系和导出约定，关键字段应为：

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

换用其他数据集时，应在导出阶段按第 1 节公式转换 state 和 action，并重新生成统计量。仅修改 `pose_frame` 元数据不会转换数据。当前训练命令本身不会执行这一步坐标变换。

本次保留全部 302 个回合（包括 episode 57），未设置 `episodes` 或 `exclude_episodes`。`eval_split=0.1` 按任务留出最后 `ceil(302×0.1)=31` 个回合验证，剩余 271 个回合训练；训练日志记录训练集为 41528 帧。

## 3. 启动训练

### 3.1 仓库已有脚本

原始训练的启动入口为：

```bash
bash scripts/train_dp_pick_episode_start_tcp.sh
```

脚本固定写入 `outputs/train/dp_pick_episode_start_tcp`，保留了原始训练关闭 Trackio 的设置。**当前该目录已经有训练结果，再次执行会因目录存在而报错。重新训练请使用下一节默认开启 Trackio 的新目录命令。**

### 3.2 重新训练同类模型

以下命令沿用原始脚本的模型与训练参数，将输出目录和任务名改为 `_rerun`，并默认开启 Trackio 记录曲线，从头训练新模型。若该目录也已存在，请再次更换名称。第 2 节的 `training` 依赖组包含 Trackio 和训练所需依赖。

```bash
cd /home/scl/work/lerobot

uv run torchrun --standalone --nproc-per-node=2 "$(uv run which lerobot-train)" \
  --dataset.repo_id=fastumi/pick \
  --dataset.root=/home/scl/work/UMI/FastUMI_Data/dataset/pick_place/lerobot/pick \
  --dataset.eval_split=0.1 \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable=false \
  --policy.type=diffusion \
  --policy.use_relative_pose=false \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.drop_n_last_frames=7 \
  --policy.down_dims='[512,1024,2048]' \
  --policy.noise_scheduler_type=DDPM \
  --policy.num_train_timesteps=100 \
  --policy.resize_shape='[224,224]' \
  --policy.crop_ratio=0.95 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/dp_pick_episode_start_tcp_rerun \
  --job_name=dp_pick_episode_start_tcp_rerun \
  --seed=1000 \
  --batch_size=128 \
  --num_workers=4 \
  --steps=19466 \
  --eval_steps=1000 \
  --save_freq=2000 \
  --log_freq=200 \
  --env_eval_freq=0 \
  --trackio.enable=true \
  --trackio.project=lerobot \
  --trackio.disable_artifact=true
```

单卡时，将 `uv run torchrun ... "$(uv run which lerobot-train)"` 替换成 `uv run lerobot-train`。每卡 batch 不变时，总 batch 会从 256 降至 128；若希望保持近似的数据遍历次数，应相应增加总步数。

### 3.3 关键参数

| 参数 | 本次取值与含义 |
|---|---|
| `use_relative_pose=false` | 保持数据集的起始 TCP 坐标和轴角表示；state/action 使用 MIN_MAX 归一化 |
| `n_obs_steps=2` | 输入当前帧及前一帧观测；回合起点缺少历史时重复首帧 |
| `horizon=16` | 预测 16 帧动作序列，时间范围为 `t-1` 到 `t+14` |
| `n_action_steps=8` | 从当前时刻开始执行其中 8 步，再重新预测；30 Hz 下约 0.27 秒 |
| `drop_n_last_frames=7` | 回合末尾 7 帧不作为训练采样起点，减少动作块补齐；不代表所有动作窗口都无补齐 |
| `down_dims=[512,1024,2048]` | U-Net 通道配置 |
| `DDPM`、`num_train_timesteps=100` | 100 个训练扩散时间步；本次 `num_inference_steps=null`，推理沿用 100 步 |
| `crop_ratio=0.95` | 224×224 图像裁剪为 212×212；训练随机裁剪，推理中心裁剪 |
| `image_transforms.enable=false` | 关闭数据集级图像增强；策略内部的随机裁剪仍然启用 |
| `batch_size=128` | 每卡 batch 128，双卡总有效 batch 256，梯度累积为 1 |
| `eval_steps=1000` | 每 1000 步计算留出回合的 `eval_loss` |
| `save_freq=2000` | 每 2000 步保存，并保存最终步检查点 |
| `env_eval_freq=0` | 关闭环境 rollout 评估；留出数据集的 loss 验证仍然启用 |
| `trackio.enable=true` | 上述新训练命令默认记录 Trackio 曲线；原始训练为 `false` |
| `trackio.project=lerobot` | 将运行记录归入本地 `lerobot` 项目 |
| `trackio.disable_artifact=true` | 不向 Trackio 上传模型文件，仍记录训练与验证指标 |

检查点中的其他设置：ResNet18＋ImageNet 预训练权重、学习率 `1e-4`、cosine 调度、warmup 500 步、无混合精度、未启用 EMA。调整学习率应使用 `--policy.optimizer_lr`；默认训练预设会覆盖单独设置的 `--optimizer.lr`。

不要额外加载 [dp_fastumi_config.json](./dp_fastumi_config.json) 来复现本次训练：该文件使用不同的通道数、DDIM 和扩散步数。

### 3.4 换数据集时如何调整步数

```text
总有效 batch = 每卡 batch × GPU 数 × 梯度累积步数
近似 epoch 数 = steps × 总有效 batch / 训练集帧数
steps ≈ 目标 epoch 数 × 训练集帧数 / 总有效 batch
```

本次 `19466 × 256 / 41528 ≈ 120`，约为 120 个 epoch。应使用划分后的训练集帧数估算，并以训练日志的 `epch` 字段核对；尾帧采样规则和 batch 取整会影响实际遍历情况。19466 只是本次配置的总步数，不代表其他数据集也应使用相同步数。

### 3.5 查看训练曲线

训练过程中或结束后，在训练机器上另开终端，进入仓库并启动 Trackio 面板：

```bash
cd /home/scl/work/lerobot
uv run trackio show --project lerobot
```

打开命令输出的网址，在项目中选择 `dp_pick_episode_start_tcp_rerun-<短 ID>` 对应的运行记录。运行名称默认由 `job_name` 加随机短 ID 生成；可在训练命令中用 `--trackio.run_name=<名称>` 显式指定。

- `train/loss`：训练损失，每 `log_freq=200` 步记录一次。
- `eval/eval_loss`：留出回合的验证损失，每 `eval_steps=1000` 步记录一次，用于观察泛化和过拟合趋势。
- 横轴选择训练步数（step），可结合平滑曲线观察整体趋势。

未设置 `trackio.space_id` 时，指标默认保存在训练用户的 `~/.cache/huggingface/trackio`，不在 `output_dir` 中；训练结束后仍可查看。若通过 SSH 训练，需将面板实际使用的端口转发到本机，再用本机浏览器访问。

原始训练未开启 Trackio，无法在面板中补回历史曲线；若保留了 `train.log`，可从日志中的损失数值另行绘图。恢复训练时再开启 Trackio，只能记录后续步骤。

## 4. 检查训练结果与恢复中断训练

原始训练已完成 19466 步，结果结构为：

```text
outputs/train/dp_pick_episode_start_tcp/
├── train.log
└── checkpoints/
    ├── 002000/
    ├── ...
    ├── 018000/
    ├── 019466/
    │   ├── pretrained_model/
    │   │   ├── model.safetensors
    │   │   ├── config.json
    │   │   ├── train_config.json
    │   │   ├── policy_preprocessor.json
    │   │   ├── policy_postprocessor.json
    │   │   └── *.safetensors（处理器归一化统计量）
    │   └── training_state/
    └── last -> 019466
```

`train.log` 是本次已有运行的日志文件；上面的直接启动命令不会自动将终端输出写到这个文件。

用于加载模型的最终路径为：

```text
/home/scl/work/lerobot/outputs/train/dp_pick_episode_start_tcp/checkpoints/019466/pretrained_model
```

核对 `config.json` 中 `use_relative_pose=false`，且 state/action 的 shape 均为 `[7]`。本次前处理器包含归一化，后处理器包含反归一化，均没有相对位姿转换步骤；只复制模型权重会丢失这些配套配置和统计量。

若新训练在达到目标步数前中断，可使用它的最新检查点恢复，例如 `_rerun`：

```bash
uv run torchrun --standalone --nproc-per-node=2 "$(uv run which lerobot-train)" \
  --config_path=outputs/train/dp_pick_episode_start_tcp_rerun/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

恢复需要保留配套的 `training_state`。原始 `019466` 已达到配置的总步数，用原配置恢复不会再训练额外步数；延长训练涉及总步数与学习率调度安排，不等同于恢复中断。

训练完成后，结合留出回合的动作误差、轨迹表现和实机成功率选择检查点。`eval_loss` 是扩散模型的验证损失，不直接等同于毫米级位姿误差或任务成功率。

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
