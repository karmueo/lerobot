#!/usr/bin/env bash
# 使用双卡训练 DP，目标为每个 pick episode 首帧 TCP 坐标系中的位姿。
# 保留全部 302 段（包括 episode 57），与 pick_baseline 使用相同样本。
# 关闭相对转换后，state/action 为 7 维 xyz + 轴角 + 夹爪，沿用 MIN_MAX 归一化。
# 本配置从头训练新策略，不兼容旧的 10 维相对位姿检查点或现有 dp_infer_lerobot。
set -euo pipefail

# 从任意工作目录调用时，均在仓库根目录解析 uv 环境和相对输出路径。
training_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$training_repo_root"

# 每卡 batch 128，总有效 batch 256；最后 31 段验证，其余 271 段训练。
# 新输出目录由训练入口检查，已有目录不会被覆盖；每 2000 步及最终步骤保存。
exec uv run torchrun --standalone --nproc-per-node=2 "$(uv run which lerobot-train)" \
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
  --output_dir=outputs/train/dp_pick_episode_start_tcp \
  --job_name=dp_pick_episode_start_tcp \
  --seed=1000 \
  --batch_size=128 \
  --num_workers=4 \
  --steps=19466 \
  --eval_steps=1000 \
  --save_freq=2000 \
  --log_freq=200 \
  --env_eval_freq=0 \
  --trackio.enable=false
