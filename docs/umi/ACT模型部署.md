# UMI pick ACT：ONNX 和 TensorRT 转换

本文针对 `scripts/train_act_pick_episode_start_tcp.sh` 训练出的 ACT 检查点。示例选择最后的 030000 步；选择其他检查点时同时修改两个路径。训练损失和转换精度不能证明机器人任务成功率。

## 准备与导出

在仓库根目录安装依赖（如果已经安装可跳过），然后使用 `--no-sync` 保留可选依赖：

```bash
uv sync --locked --inexact --extra dataset --extra export --extra export-tensorrt

MODEL_PATH=outputs/train/act_pick_episode_start_tcp/checkpoints/030000/pretrained_model
EXPORT_DIR=outputs/export/act_pick_episode_start_tcp/030000
EXPORT_DEVICE=cuda:0

uv run --no-sync lerobot-export \
  --policy_path="$MODEL_PATH" \
  --output_dir="$EXPORT_DIR" \
  --format=both --device="$EXPORT_DEVICE" --trt_precision=both
```

输出目录默认不可覆盖；重新导出请改用新目录，或显式指定 `--overwrite=true` 替换已有 ACT 导出包。`--format=onnx --device=cpu` 只生成 ONNX；TensorRT 可用 `--trt_precision=fp32` 或 `fp16` 选择精度。ONNX 在 CPU 追踪，TensorRT 在指定的 GPU 上构建。ACT 的 FP16 engine 采用保守混合精度：视觉残差块、Transformer 和动作头保持 FP32，其他适用层可使用 FP16；不能将它理解为整个网络均为 FP16。TensorRT 需要 Linux x86-64、CUDA GPU 和项目锁定的 TensorRT 10.13.3；更换 GPU 型号或 TensorRT 版本后重新构建。

模型包含 `policy.onnx`、`policy_fp32.engine`、`policy_fp16.engine`、`assets/` 和 `manifest.json`。按选择的格式和精度生成对应文件；复制时保留整个目录，包括可能的 ONNX 外部权重。模型包不包含原始模型权重，也不需要原检查点即可运行，但需要 Python、PyTorch、LeRobot 前后处理器及对应后端。

## 精度和性能验证

验证命令需要原始检查点和本地训练数据；每个窗口比较原生模型与导出模型的**全部 50 步**，包括 episode 首帧。ACT 推理确定性使用零 VAE latent，因此不重复随机种子采样。以下完整验收默认选择 4 段、每段 8 个窗口以及每段首帧，并测量 GPU 性能：

```bash
OMP_NUM_THREADS=4 uv run --no-sync lerobot-validate-export \
  --policy_path="$MODEL_PATH" \
  --export_dir="$EXPORT_DIR" \
  --device="$EXPORT_DEVICE" \
  --warmup=10 --iterations=100
```

结果保存在 `validation_report.json`。输出位置误差上限 1 mm、旋转误差上限 0.5°、夹爪误差上限 0.01；比较的是转换误差而不是对示教轨迹的误差。FP16 只有通过后才能视为满足这些转换精度要求。

快速检查一段一个窗口，且不做性能测量：

```bash
uv run --no-sync lerobot-validate-export \
  --policy_path="$MODEL_PATH" \
  --export_dir="$EXPORT_DIR" \
  --device="$EXPORT_DEVICE" \
  --episodes=1 --windows_per_episode=1 --benchmark=false
```

仅验证 ONNX 时使用 `--device=cpu --backends='[onnx]'`；数据迁移时用 `--dataset_root=/path/to/dataset`。ACT 的物理误差验证要求数据集 `meta/fastumi.json` 声明 `action_layout=xyz_rotvec_gripper` 及米、弧度、归一化夹爪单位。性能计时排除文件读取、视频解码、模型加载和 engine 构建，包含前后处理；显存是整卡采样值。

### 030000 检查点的本机结果

2026-10-30 在 RTX 5090 上验证 4 个 episode 的 32 个常规窗口及 4 个起点窗口，每个窗口比较完整 50 步：

| 后端 | 最大位置差 | 最大旋转差 | 最大夹爪差 | GPU 完整调用 P50 |
| --- | ---: | ---: | ---: | ---: |
| ONNX FP32（CPU） | 0.000553 mm | 0.0000577° | 0.0000249 | 不参与 GPU 比较 |
| TensorRT FP32 | 0.000590 mm | 0.0000610° | 0.0000299 | 1.286 ms |
| TensorRT 混合 FP16 | 0.325 mm | 0.0304° | 0.00638 | 1.238 ms |
| PyTorch FP32 | 基准 | 基准 | 基准 | 3.909 ms |

这些数值仅适用于该检查点、当前 GPU 和软件环境；重新构建可能选择不同计算方式，应重新验证。混合 FP16 的夹爪误差接近门槛时，优先使用通过验证的 FP32。

## 输入输出与部署边界

网络的固定 batch=1 接口记录在 `manifest.json`。本检查点的输入是经过保存的处理器归一化后的 `state` `[1,7]` 和 `image_0` `[1,3,224,224]`，图像是 RGB、浮点 0～1 的训练语义，并按训练数据的 224×224 等比例缩放加黑边处理；输出是归一化的 `actions` `[1,50,7]`。必须使用包内统计量进行前处理和反归一化，不能将原始图像或网络原始输出直接当作机器人动作。

`lerobot.export.runtime.ExportedACT` 提供 `predict_chunk(raw)`（完整动作块）和 `predict(raw)`（前 10 步），返回元组的第一个元素为反归一化动作。`raw` 的键使用模型的原始输入名，例如 `observation.state` 和 `observation.images.front`；数据形状分别为 `[1,7]` 和 `[1,3,224,224]`。调用方负责每 10 步重新观测、维护执行队列及 episode 切换时调用 `reset()`。该运行器不实现时间集成，启用了时间集成的 ACT 检查点会被导出时拒绝。

例如，已有按上述语义准备的 `state` 和 `image` PyTorch 张量时，可这样加载 FP32 engine 并获取目标：

```python
from pathlib import Path
from lerobot.export.backends import TensorRTSession
from lerobot.export.package import load_manifest
from lerobot.export.runtime import ExportedACT

root = Path("outputs/export/act_pick_episode_start_tcp/030000")
manifest = load_manifest(root)
session = TensorRTSession(root, manifest, "cuda:0", "fp32")
runner = ExportedACT(root, session, "cuda:0", manifest)

# state: [1, 7] 起始 TCP 坐标系位姿；image: [1, 3, 224, 224] RGB 浮点图像。
runner.reset()  # 开始新的 episode 时调用。
raw = {"observation.state": state, "observation.images.front": image}
targets = runner.predict(raw)[0]  # [1, 10, 7]，反归一化后的目标。
```

若通过精度验证后使用 FP16，将上面 `TensorRTSession` 的 `"fp32"` 改为 `"fp16"`。ONNX 可替换为 `OnnxSession(root, manifest, "cpu")`，并将运行器设备设为 `"cpu"`。

这些动作是**当前 pick episode 起始 TCP 坐标系中的目标位姿**，不是相对当前 TCP 的增量。实际机器人推理需要记录当前 episode 的起始 TCP，将实时观测转换到相同坐标系，并将预测目标转换回机器人控制坐标系；夹爪输出的阈值和安全执行需要单独设计。现有 `dp_infer_lerobot` 不能直接加载此 ACT 模型包。

Diffusion 模型的转换和部署说明见 [模型部署.md](./模型部署.md)。
