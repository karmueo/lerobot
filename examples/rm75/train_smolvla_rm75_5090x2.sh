#!/usr/bin/env bash
#
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 在 2x RTX 5090 上用 RM75 玩具机器人数据集微调 SmolVLA(纯 DDP,BF16)。
#
# 可训练范围两种方案(环境变量切换,其余配置完全一致,便于对照):
#   ① 默认:全量微调(不冻结视觉编码器,并非仅训练 expert)。视觉编码器可适配相机
#      域偏移(合成渲染 → 真实末端相机),但 600 条演示下过拟合/遗忘风险与显存占用
#      更高(单卡 32 峰值约 22GB)。
#   ② 对照:SmolVLA 论文默认方案(冻结视觉编码器 + 仅训练 expert,见
#      configuration_smolvla.py 的默认值):
#      RM75_FREEZE_VISION=true RM75_TRAIN_EXPERT_ONLY=true \
#          SMOLVLA_OUTPUT=outputs/rm75_smolvla_frozen \
#          ./examples/rm75/train_smolvla_rm75_5090x2.sh
#      保留预训练视觉表征,可训练参数与显存更少(有效批 64 无需变动);代价是冻结的
#      视觉编码器无法适配明显域偏移。两条曲线以留出集评估损失 + 实机 rollout 对比定夺。
#   ③ 局部解冻:冻结视觉编码器,训练 expert、图像 connector 和最后 N 个 VLM 文本层。
#      例如 RM75_TRAIN_VLM_LAYERS=2 可在保留大部分预训练表征的同时提供少量任务适配能力；
#      该模式要求 RM75_TRAIN_EXPERT_ONLY=false。
#
# 默认配置:chunk_size=64(训练目标长度,不随部署变化),n_action_steps=16(仅影响部署:每次
# 推理消费的动作数,50Hz 下开环 0.32s 后重规划;不影响训练损失,训练 chunk 由
# chunk_size 推导。10-20 之间的最终值应根据控制延迟与实机表现调整,推理时可经
# --policy.n_action_steps 覆盖而无需重训),学习率 1e-4,带 1,000 步预热和 30,000 步余弦衰减
# (SmolVLA 训练预设),每 5,000 步保存一次检查点并在留出集上评估(最多 2,048 个评估样本),
# 检查点仅保存到本地(不上传 Hub 或 W&B artifact),启用 W&B 指标记录,
# 不启用 EMA,不做图像增强。
#
# 注意 --steps 的语义:它计的是训练循环迭代数,每个迭代消费一个微批次;优化器更新
# 由 accelerate 按累积窗口推进(LR 调度器同样按更新推进)。--steps/--save_freq/
# --eval_steps 都按迭代计数。默认 accum=1 时迭代数恰等于更新数:--steps=30000 即
# 30,000 次更新,余弦衰减恰好在训练结束时走完,每 5,000 步保存/评估即每 5,000 次更新。
# 若 OOM 改用 batch=16 + 累积 2(每 2 个迭代才更新一次),为维持 30,000 次更新、完整
# 衰减以及"每 5,000 次优化器更新保存/评估"的节奏,迭代数参数全部翻倍:
#   RM75_PER_GPU_BATCH=16 RM75_GRAD_ACCUM=2 ./examples/rm75/train_smolvla_rm75_5090x2.sh \
#       --steps=60000 --save_freq=10000 --eval_steps=10000
# (不改的话只剩 15,000 次更新、衰减走不完,保存/评估也密了一倍。)
#
# 有效批大小 = 单卡批大小 x GPU 数 x 梯度累积 = 32 x 2 x 1 = 64,与 SmolVLA
# 训练预设的假定批大小一致(学习率 1e-4 无需调整)。实测 2x RTX 5090(BF16):
# 单卡 32 峰值显存约 22GB/32GB,稳态约 3.3 step/s;30,000 步 x 64 ≈ 8 个 epoch
# (约 24.1 万训练帧),总墙钟约 2.5-3 小时。
#
# 数据集相机名为 observation.images.gripper,通过 --rename_map 映射到基座模型的
# observation.images.camera1;状态/动作维度(8)与基座(6)不同由模型内部 pad 自适应。
#
# 任何额外的尾部参数都会原样附加到 lerobot-train 命令后,例如:
#   ./examples/rm75/train_smolvla_rm75_5090x2.sh --resume=true \
#       --config_path=outputs/rm75_smolvla_5090x2/checkpoints/000010000/pretrained_model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 可覆盖的配置(环境变量)────────────────────────────────────────────────────────
#   RM75_LEROBOT_DATASET  转换后的 LeRobot 数据集本地根目录
#   SMOLVLA_BASE          SmolVLA 初始权重的 Hub id 或本地目录
#   SMOLVLA_OUTPUT        训练输出目录
#   SMOLVLA_PORT          torchrun rendezvous 端口(如有冲突请修改)
#   SMOLVLA_MASTER_ADDR   torchrun rendezvous 地址
#   RM75_PER_GPU_BATCH    单卡批大小(默认 32;OOM 时改 16 并把梯度累积改为 2,
#                         同时将 --steps/--save_freq/--eval_steps 翻倍,见上方说明)
#   RM75_GRAD_ACCUM       梯度累积步数(默认 1;与单卡批大小配合维持有效批 64)
#   RM75_FREEZE_VISION    冻结视觉编码器(默认 false;对照实验设 true)
#   RM75_TRAIN_EXPERT_ONLY 仅训练动作 expert(默认 false;对照实验设 true)
#   RM75_TRAIN_VLM_LAYERS  局部解冻最后 N 个 VLM 文本层及图像 connector(默认 0;需 expert_only=false)
RM75_LEROBOT_DATASET="${RM75_LEROBOT_DATASET:-${LEROBOT_ROOT}/data/rm75_toy_robot_lerobot}"
SMOLVLA_BASE="${SMOLVLA_BASE:-lerobot/smolvla_base}"
SMOLVLA_OUTPUT="${SMOLVLA_OUTPUT:-${LEROBOT_ROOT}/outputs/rm75_smolvla_5090x2}"
SMOLVLA_PORT="${SMOLVLA_PORT:-29517}"
SMOLVLA_MASTER_ADDR="${SMOLVLA_MASTER_ADDR:-127.0.0.1}"
RM75_PER_GPU_BATCH="${RM75_PER_GPU_BATCH:-32}"
RM75_GRAD_ACCUM="${RM75_GRAD_ACCUM:-1}"
RM75_FREEZE_VISION="${RM75_FREEZE_VISION:-false}"
RM75_TRAIN_EXPERT_ONLY="${RM75_TRAIN_EXPERT_ONLY:-false}"
RM75_TRAIN_VLM_LAYERS="${RM75_TRAIN_VLM_LAYERS:-0}"
RM75_NUM_WORKERS="${RM75_NUM_WORKERS:-8}"

# ── RTX 5090(Blackwell)的 NCCL 设置,继承自原来的 2x5090 脚本。
# 已预先 export,用户仍可通过环境变量覆盖其中任意一项。
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NTHREADS="${NCCL_NTHREADS:-256}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${LEROBOT_ROOT}"

echo "[rm75] dataset   : ${RM75_LEROBOT_DATASET}"
echo "[rm75] base model: ${SMOLVLA_BASE}"
echo "[rm75] output    : ${SMOLVLA_OUTPUT}"
echo "[rm75] effective batch: ${RM75_PER_GPU_BATCH} x 2 x ${RM75_GRAD_ACCUM} = $((RM75_PER_GPU_BATCH * 2 * RM75_GRAD_ACCUM))"

uv run torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_addr="${SMOLVLA_MASTER_ADDR}" \
  --master_port="${SMOLVLA_PORT}" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=rm75_toy_robot \
  --dataset.root="${RM75_LEROBOT_DATASET}" \
  --dataset.eval_split=0.1 \
  --rename_map='{"observation.images.gripper": "observation.images.camera1"}' \
  --dataset.image_transforms.enable=false \
  --policy.path="${SMOLVLA_BASE}" \
  --policy.freeze_vision_encoder="${RM75_FREEZE_VISION}" \
  --policy.train_expert_only="${RM75_TRAIN_EXPERT_ONLY}" \
  --policy.train_vlm_layers="${RM75_TRAIN_VLM_LAYERS}" \
  --policy.chunk_size=64 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --output_dir="${SMOLVLA_OUTPUT}" \
  --job_name=rm75_smolvla_5090x2 \
  --seed=1000 \
  --steps=30000 \
  --batch_size="${RM75_PER_GPU_BATCH}" \
  --num_workers="${RM75_NUM_WORKERS}" \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps="${RM75_GRAD_ACCUM}" \
  --use_policy_training_preset=true \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_steps=5000 \
  --max_eval_samples=2048 \
  --policy.push_to_hub=false \
  --save_checkpoint_to_hub=false \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --ema.enable=false \
  "$@"
