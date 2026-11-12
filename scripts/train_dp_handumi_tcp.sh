#!/usr/bin/env bash
# 训练 HandUMI TCP 策略；YAML 可选择 GPU、位姿参考系、图像缩放与串行观测。
set -euo pipefail
training_timestamp="$(date +%Y-%m-%d_%H-%M-%S_%N)"

dataset_path=""
config_path=""
# regularized 推理使用 checkpoints/best/pretrained_model_ema；baseline 使用 checkpoints/best/pretrained_model。
recipe="regularized"
recipe_specified=false
recipe_variant="${TRAINING_RECIPE_VARIANT:-}"

usage() {
  printf '用法: %s <LeRobot 数据集目录> [--recipe baseline|regularized]\n       %s --config_path <训练 YAML 文件>\n' "$0" "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --recipe)
      if [[ $# -lt 2 ]]; then
        echo "--recipe 需要 baseline 或 regularized" >&2
        exit 2
      fi
      recipe="$2"
      recipe_specified=true
      shift 2
      ;;
    --recipe=*)
      recipe="${1#--recipe=}"
      recipe_specified=true
      shift
      ;;
    --config_path)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "--config_path 需要训练 YAML 文件路径" >&2
        exit 2
      fi
      config_path="$2"
      shift 2
      ;;
    --config_path=*)
      config_path="${1#--config_path=}"
      if [[ -z "$config_path" ]]; then
        echo "--config_path 需要训练 YAML 文件路径" >&2
        exit 2
      fi
      shift
      ;;
    -*)
      echo "未知选项: $1" >&2
      exit 2
      ;;
    *)
      if [[ -n "$dataset_path" ]]; then
        echo "只能指定一个数据集目录" >&2
        exit 2
      fi
      dataset_path="$1"
      shift
      ;;
  esac
done

# YAML 模式在初始化 CUDA 前选卡；原有数据集目录模式保持双进程行为。
launch_training() {
  local training_entry
  training_entry="$(uv run --locked --extra diffusion --extra training which lerobot-train)"
  if [[ -n "$config_path" ]]; then
    exec uv run --locked --extra diffusion --extra training python scripts/launch_handumi_training.py \
      --config-path "$config_path" --training-entry "$training_entry" --timestamp "$training_timestamp"
  fi
  exec uv run --locked --extra diffusion --extra training torchrun --standalone --nproc-per-node=2 "$training_entry" "$@"
}

training_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "$config_path" ]]; then
  if [[ -n "$dataset_path" || "$recipe_specified" == true || -n "$recipe_variant" ]]; then
    echo "--config_path 模式的数据集和训练参数由 YAML 指定，请单独使用该选项" >&2
    exit 2
  fi
  if [[ ! -f "$config_path" || ( "$config_path" != *.yaml && "$config_path" != *.yml ) ]]; then
    echo "训练 YAML 文件不存在或扩展名不是 .yaml/.yml: $config_path" >&2
    exit 2
  fi
  # 配置文件路径相对于调用目录，YAML 内的相对路径则从仓库根目录解析。
  if [[ "$config_path" != /* ]]; then
    config_path="$PWD/$config_path"
  fi
  cd -- "$training_repo_root"
  # 启动器为本次任务生成一次输出路径，供所有训练进程共用。
  launch_training
fi

if [[ -z "$dataset_path" ]] || [[ "$recipe" != "baseline" && "$recipe" != "regularized" ]]; then
  usage >&2
  exit 2
fi

# 先按调用时的工作目录解析路径，再切换到仓库以解析 uv 环境与输出路径。
if [[ ! -d "$dataset_path" ]] || [[ ! -f "$dataset_path/meta/info.json" ]]; then
  echo "数据集目录不存在或缺少 meta/info.json: $dataset_path" >&2
  exit 2
fi
dataset_root="$(cd -- "$dataset_path" && pwd -P)"
dataset_name="$(basename -- "$dataset_root")"
cd -- "$training_repo_root"

# 实验对照可单独启用小模型或颜色增强，其他训练参数与基线一致。
if [[ -n "$recipe_variant" && "$recipe" != "baseline" ]]; then
  echo "TRAINING_RECIPE_VARIANT 仅适用于 baseline 配方" >&2
  exit 2
fi
case "$recipe_variant" in
  "") ;;
  small_only|color_only) recipe="$recipe_variant" ;;
  *) echo "未知 TRAINING_RECIPE_VARIANT: $recipe_variant" >&2; exit 2 ;;
esac

down_dims='[512,1024,2048]'
recipe_options=(--dataset.image_transforms.enable=false)
if [[ "$recipe" == "regularized" || "$recipe" == "small_only" ]]; then
  down_dims='[256,512,1024]'
fi
if [[ "$recipe" == "regularized" || "$recipe" == "color_only" ]]; then
  # 显式覆盖变换字典，避免默认 affine 改变图像与 TCP 标签的几何对应。
  recipe_options=(
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.backend=gpu
    --dataset.image_transforms.max_num_transforms=1
    '--dataset.image_transforms.tfs={"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2],"contrast":[0.8,1.2],"saturation":[0.9,1.1]}}}'
  )
fi
if [[ "$recipe" == "regularized" ]]; then
  recipe_options+=(--ema.type=warmup --eval_use_ema=true)
fi

# Trackio 可通过 uv run --extra training trackio show --project lerobot 查看。
# 每 500 步比较验证损失，每 2000 步和最后一步更新 last。
launch_training \
  --dataset.repo_id="handumi/$dataset_name" \
  --dataset.root="$dataset_root" \
  --dataset.eval_split=0.1 \
  --dataset.eval_split_strategy=random \
  --dataset.eval_split_seed=1000 \
  --dataset.video_backend=torchcodec \
  "${recipe_options[@]}" \
  --policy.type=diffusion \
  --policy.use_relative_pose=false \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.drop_n_last_frames=7 \
  --policy.down_dims="$down_dims" \
  --policy.noise_scheduler_type=DDPM \
  --policy.num_train_timesteps=100 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="outputs/train/dp_handumi_tcp_${recipe}_${training_timestamp}" \
  --job_name=dp_handumi_tcp \
  --seed=1000 \
  --batch_size=128 \
  --num_workers=4 \
  --steps=19466 \
  --eval_steps=500 \
  --eval_seed=1000 \
  --save_freq=2000 \
  --log_freq=200 \
  --env_eval_freq=0 \
  --keep_best_and_last=true \
  --trackio.enable=true
