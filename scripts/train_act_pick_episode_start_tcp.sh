#!/usr/bin/env bash
# 使用双卡从头训练 ACT，视觉骨干沿用 ImageNet 预训练 ResNet18。
# 保留全部 302 段（包括 episode 57）：271 段训练，最后 31 段验证。
# state/action 为 7 维 xyz + 轴角 + 夹爪，使用 ACT 默认 MEAN_STD 归一化。
# 反归一化后的动作是 episode 首帧 TCP 坐标系中的目标，不能作为当前位姿增量累加。
# 现有 dp_infer_lerobot 不支持此 ACT 模型，机器人部署需要另行适配。
set -euo pipefail

# 从任意工作目录调用时，均在仓库根目录解析 uv 环境和相对输出路径。
training_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$training_repo_root"

# 输入当前一帧，预测 50 步；默认执行前 10 步后重新观测，不启用时间集成。
# 原图已为 224×224；尾部补齐动作由 ACT 的 padding mask 屏蔽，无需丢弃尾帧。
# 每卡 batch 32，总有效 batch 64；AdamW 使用默认梯度裁剪，无学习率调度器。
# 每 1000 步验证并保存，最终步骤也保存；不自动选择最佳模型或提前停止。
# eval_loss 为归一化 L1，训练总损失还含 KL，不能直接比较二者绝对值。
# 训练入口检查输出目录冲突；本脚本不自动覆盖或续训已有模型。
exec uv run torchrun --standalone --nproc-per-node=2 "$(uv run which lerobot-train)" \
  --dataset.repo_id=fastumi/pick \
  --dataset.root=/home/scl/work/UMI/FastUMI_Data/dataset/pick_place/lerobot/pick \
  --dataset.eval_split=0.1 \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable=false \
  --policy.type=act \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.use_vae=true \
  --policy.kl_weight=10.0 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_pick_episode_start_tcp \
  --job_name=act_pick_episode_start_tcp \
  --seed=1000 \
  --batch_size=32 \
  --num_workers=4 \
  --steps=30000 \
  --eval_steps=1000 \
  --max_eval_samples=0 \
  --save_freq=1000 \
  --log_freq=200 \
  --env_eval_freq=0 \
  --wandb.enable=false
