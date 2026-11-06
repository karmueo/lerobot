#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""在保存的验证 episode 上评估 HandUMI 去噪损失和实际执行动作。

示例::

    uv run --locked --extra diffusion --extra training python scripts/eval_dp_handumi_tcp.py \
        --checkpoint outputs/train/EXPERIMENT/checkpoints/best \
        --output-dir outputs/eval/EXPERIMENT --device cuda:0

传入 step 目录时依据训练配置中的 ``eval_use_ema`` 选择权重，并在输出中记录；
也可直接传入 ``pretrained_model`` 或 ``pretrained_model_ema`` 明确选择。
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.export.package import file_hash, load_policy, load_processors, read_json, write_json
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.relative_pose_processor import axis_angle_to_matrix
from lerobot.processor.rename_processor import rename_batch_keys
from lerobot.utils.random_utils import seeded_context

EVAL_SEEDS = (0, 1, 2)
WINDOWS_PER_EPISODE = 8
ACTION_BATCH_SIZE = 16
LOSS_BATCH_SIZE = 128


def resolve_checkpoint(checkpoint: Path) -> tuple[Path, Path]:
    """返回策略目录和划分清单；step 目录按当时的 best 权重类型选择。"""
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise NotADirectoryError(checkpoint)
    if (checkpoint / "config.json").is_file():
        model_dir = checkpoint
    else:
        live_dir = checkpoint / "pretrained_model"
        training = read_json(live_dir / "train_config.json")
        model_dir = checkpoint / (
            "pretrained_model_ema" if training.get("eval_use_ema") else "pretrained_model"
        )
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Selected model directory is missing: {model_dir}")
    manifest = next(
        (
            candidate
            for candidate in (model_dir / "split_manifest.json", model_dir.parent / "split_manifest.json")
            if candidate.is_file()
        ),
        None,
    )
    if manifest is None:
        raise FileNotFoundError(f"Checkpoint has no split_manifest.json: {model_dir}")
    return model_dir, manifest


def validate_split(manifest: dict[str, Any], training: dict[str, Any]) -> tuple[Path, str, list[int]]:
    """拒绝来源不一致、重复或与训练集重叠的验证 episode。"""
    if manifest.get("version") != 1:
        raise ValueError("Unsupported split manifest version")
    dataset = manifest.get("dataset", {})
    saved_dataset = training.get("dataset", {})
    root_value = dataset.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("Split manifest must identify a local dataset root")
    root = Path(root_value).expanduser().resolve(strict=True)
    if not root.is_dir() or not saved_dataset.get("root"):
        raise ValueError("A local dataset root is required")
    if root != Path(saved_dataset["root"]).expanduser().resolve(strict=True):
        raise ValueError("Checkpoint and split manifest dataset roots differ")
    repo_id = dataset.get("repo_id")
    if not isinstance(repo_id, str) or repo_id != saved_dataset.get("repo_id"):
        raise ValueError("Checkpoint and split manifest dataset identifiers differ")
    train, evaluation = manifest.get("train_episodes"), manifest.get("eval_episodes")
    if not isinstance(train, list) or not isinstance(evaluation, list):
        raise ValueError("Split manifest must contain train and eval episode lists")
    for label, episodes in (("train", train), ("eval", evaluation)):
        if not episodes or any(type(ep) is not int or ep < 0 for ep in episodes):
            raise ValueError(f"Invalid {label} episode list")
        if len(set(episodes)) != len(episodes):
            raise ValueError(f"Duplicate {label} episode")
    if set(train) & set(evaluation):
        raise ValueError("Training and evaluation episode lists overlap")
    return root, repo_id, sorted(evaluation)


def ensure_local_metadata(root: Path) -> None:
    """先验证本地元数据，禁止底层数据集缺文件时写回或从 Hub 下载。"""
    required = [root / "meta/info.json", root / "meta/stats.json", root / "meta/tasks.parquet"]
    if not list((root / "meta/episodes").rglob("*.parquet")):
        required.append(root / "meta/episodes")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Local metadata is incomplete: {missing}")


def build_windows(
    episode_rows: Sequence[Mapping[str, Any]],
    eval_episodes: Sequence[int],
    absolute_to_relative: Mapping[int, int],
    *,
    windows_per_episode: int = WINDOWS_PER_EPISODE,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
) -> list[dict[str, int]]:
    """按原始帧索引等间隔选窗，保证历史观测和执行窗口均无补帧。"""
    if windows_per_episode < 1 or n_obs_steps < 1 or n_action_steps < 1:
        raise ValueError("Window counts and horizons must be positive")
    result = []
    for episode in sorted(eval_episodes):
        if episode < 0 or episode >= len(episode_rows):
            raise ValueError(f"Invalid evaluation episode: {episode}")
        row = episode_rows[episode]
        start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        low, high = start + n_obs_steps - 1, end - n_action_steps
        if high - low + 1 < windows_per_episode:
            raise ValueError(f"Episode {episode} is too short for {windows_per_episode} distinct windows")
        for index in np.linspace(low, high, windows_per_episode, dtype=np.int64):
            absolute_index = int(index)
            if absolute_index not in absolute_to_relative:
                raise ValueError(f"Missing frame {absolute_index} from episode {episode}")
            result.append(
                {
                    "episode": episode,
                    "anchor": absolute_index,
                    "relative_index": absolute_to_relative[absolute_index],
                }
            )
    return result


def action_errors(reference: Tensor, actual: Tensor) -> dict[str, Tensor]:
    """计算 7D 物理动作的平移欧氏距离、SO(3) 测地角和夹爪绝对误差。"""
    if reference.shape != actual.shape or reference.ndim != 3 or reference.shape[-1] != 7:
        raise ValueError(f"Expected equal [batch, horizon, 7] actions, got {reference.shape}, {actual.shape}")
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError("Non-finite action value")
    target = reference.detach().to(device="cpu", dtype=torch.float64)
    prediction = actual.detach().to(device="cpu", dtype=torch.float64)
    target_rotation = axis_angle_to_matrix(target[..., 3:6])
    prediction_rotation = axis_angle_to_matrix(prediction[..., 3:6])
    relative_rotation = target_rotation.transpose(-1, -2) @ prediction_rotation
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return {
        "position_m": torch.linalg.vector_norm(target[..., :3] - prediction[..., :3], dim=-1),
        "rotation_rad": torch.acos(cosine),
        "gripper": (target[..., 6] - prediction[..., 6]).abs(),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    """同时给均值和尾部误差，避免少量异常被平均值掩盖。"""
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot summarize empty or non-finite errors")
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}


def summarize_actions(errors: dict[str, np.ndarray], windows: list[dict[str, int]]) -> dict[str, Any]:
    """按种子、episode 和执行步汇总全部样本，不按表现挑选采样。"""
    count = len(windows)
    expected = (len(EVAL_SEEDS), count, 8)
    if any(value.shape != expected for value in errors.values()):
        raise ValueError(f"Expected metric arrays shaped {expected}")
    episodes = sorted({item["episode"] for item in windows})

    def metrics(selector: tuple[Any, ...]) -> dict[str, Any]:
        return {name: summarize(value[selector]) for name, value in errors.items()}

    return {
        "overall": metrics((slice(None), slice(None), slice(None))),
        "first_step": metrics((slice(None), slice(None), 0)),
        "per_step": {str(step): metrics((slice(None), slice(None), step)) for step in range(8)},
        "per_seed": {
            str(seed): metrics((index, slice(None), slice(None))) for index, seed in enumerate(EVAL_SEEDS)
        },
        "per_episode": {
            str(episode): metrics(
                (
                    slice(None),
                    [i for i, item in enumerate(windows) if item["episode"] == episode],
                    slice(None),
                )
            )
            for episode in episodes
        },
    }


def prepare_batch(
    raw: dict[str, Any], camera_keys: Sequence[str], rename_map: dict[str, str], preprocessor: Any
) -> Any:
    """与训练相同地缩放图像、应用字段重命名和保存的预处理器。"""
    batch = dict(raw)
    for key in camera_keys:
        if key in batch and batch[key].dtype == torch.uint8:
            batch[key] = batch[key].to(dtype=torch.float32) / 255.0
    return preprocessor(rename_batch_keys(batch, rename_map))


def make_loader(
    dataset: Any, *, batch_size: int, num_workers: int, device: str, sampler: Any = None
) -> DataLoader:
    """使用独立生成器，数据读取不消费扩散模型的随机数。"""
    options: dict[str, Any] = {}
    if num_workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(0),
        pin_memory=torch.device(device).type == "cuda",
        drop_last=False,
        **options,
    )


@torch.inference_mode()
def evaluate_losses(
    policy: DiffusionPolicy, loader: DataLoader, camera_keys: Sequence[str], rename: dict[str, str], pre: Any
) -> dict[str, Any]:
    """逐种子评估整个验证集；按样本加权，避免末批放大权重。"""
    results = {}
    counts = []
    for seed in EVAL_SEEDS:
        total, count = 0.0, 0
        with seeded_context(seed):
            for batch in loader:
                size = len(batch["episode_index"])
                loss, _ = policy(prepare_batch(batch, camera_keys, rename, pre))
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite validation loss for seed {seed}")
                total += float(loss.item()) * size
                count += size
        if not count:
            raise ValueError("Validation data loader is empty")
        results[str(seed)] = total / count
        counts.append(count)
        print(f"Validation loss seed={seed}: {results[str(seed)]:.8f} ({count} frames)", flush=True)
    if len(set(counts)) != 1:
        raise ValueError("Validation sample counts differ between seeds")
    return {
        "mean": float(np.mean(list(results.values()))),
        "per_seed": results,
        "samples_per_seed": counts[0],
    }


@torch.inference_mode()
def evaluate_actions(
    policy: DiffusionPolicy,
    loader: DataLoader,
    windows: list[dict[str, int]],
    camera_keys: Sequence[str],
    rename: dict[str, str],
    pre: Any,
    post: Any,
) -> dict[str, Any]:
    """固定完整 DDPM 过程后评估实际执行的 8 个预测动作。"""
    collected: dict[str, list[np.ndarray]] = {name: [] for name in ("position_m", "rotation_rad", "gripper")}
    for seed in EVAL_SEEDS:
        seed_errors: dict[str, list[np.ndarray]] = {name: [] for name in collected}
        offset = 0
        with seeded_context(seed):
            for batch in loader:
                size = len(batch["episode_index"])
                expected = windows[offset : offset + size]
                actual_indices = batch["index"].tolist()
                actual_episodes = batch["episode_index"].tolist()
                if actual_indices != [item["anchor"] for item in expected] or actual_episodes != [
                    item["episode"] for item in expected
                ]:
                    raise ValueError("Dataloader returned windows outside the saved evaluation manifest")
                if batch["action_is_pad"][:, 1:9].any():
                    raise ValueError("Execution window unexpectedly contains padded actions")
                for key in ("observation.state_is_pad", *(f"{key}_is_pad" for key in camera_keys)):
                    if key in batch and batch[key].any():
                        raise ValueError(f"Observation window unexpectedly contains padding: {key}")
                # 预处理器可能就地修改数据，必须先复制物理单位标签。
                target = batch["action"][:, 1:9].clone()
                prepared = prepare_batch(batch, camera_keys, rename, pre)
                policy.reset()
                predicted = post(policy.predict_action_chunk(prepared))
                for name, value in action_errors(target, predicted).items():
                    seed_errors[name].append(value.numpy())
                offset += size
        if offset != len(windows):
            raise ValueError(f"Expected {len(windows)} windows, evaluated {offset}")
        for name, values in seed_errors.items():
            collected[name].append(np.concatenate(values, axis=0))
        print(f"Action sampling seed={seed}: {offset} windows", flush=True)
    return summarize_actions({name: np.stack(values) for name, values in collected.items()}, windows)


def evaluate(checkpoint: Path, output_dir: Path, device: str, num_workers: int) -> dict[str, Any]:
    """从本地检查点和固定划分计算可复核的验证结果。"""
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    model_dir, split_path = resolve_checkpoint(checkpoint)
    training = read_json(model_dir / "train_config.json")
    split = read_json(split_path)
    root, repo_id, episodes = validate_split(split, training)
    output = output_dir.expanduser().resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output directory must not overlap the dataset")
    if output.is_relative_to(model_dir) or model_dir.is_relative_to(output):
        raise ValueError("Output directory must not overlap the checkpoint")
    ensure_local_metadata(root)
    units = read_json(root / "meta/fastumi.json")
    expected_units = {
        "position_unit": "m",
        "rotation_unit": "rad",
        "gripper_representation": "normalized_0_1",
        "action_layout": "xyz_rotvec_gripper",
    }
    if any(units.get(key) != value for key, value in expected_units.items()):
        raise ValueError(f"Dataset does not specify required physical units/layout: {expected_units}")

    policy = load_policy(model_dir, device)
    if not isinstance(policy, DiffusionPolicy):
        raise ValueError("This evaluator supports Diffusion Policy only")
    cfg = policy.config
    if (cfg.n_obs_steps, cfg.horizon, cfg.n_action_steps, cfg.use_relative_pose) != (2, 16, 8, False):
        raise ValueError("Expected n_obs_steps=2, horizon=16, n_action_steps=8 and absolute TCP action")
    action_feature = cfg.action_feature
    if action_feature is None or action_feature.shape != (7,):
        raise ValueError("Expected 7-dimensional TCP actions")
    if (
        cfg.noise_scheduler_type != "DDPM"
        or cfg.num_train_timesteps != 100
        or cfg.num_inference_steps not in (None, 100)
    ):
        raise ValueError("Expected DDPM with 100 train and inference steps")
    pre, post = load_processors(model_dir, cfg, device)
    rename = training.get("rename_map", {})
    if not isinstance(rename, dict):
        raise ValueError("Invalid checkpoint rename_map")

    # 本地读取失败必须立即报错，禁止 LeRobotDataset 自动从 Hub 下载到数据目录。
    with (
        patch.object(
            LeRobotDatasetMetadata,
            "_pull_from_repo",
            side_effect=FileNotFoundError("Offline metadata missing"),
        ),
        patch.object(
            LeRobotDataset, "_download", side_effect=FileNotFoundError("Offline dataset files missing")
        ),
    ):
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        if any(ep >= meta.total_episodes for ep in episodes):
            raise ValueError("Manifest references an episode outside the dataset")
        delta = resolve_delta_timestamps(cfg, meta, rename_map=rename)
        dataset_cfg = training["dataset"]
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta,
            image_transforms=None,
            video_backend=dataset_cfg.get("video_backend", "torchcodec"),
            return_uint8=True,
            depth_output_unit=dataset_cfg.get("depth_output_unit", "mm"),
            tolerance_s=training.get("tolerance_s", 1e-4),
        )
    absolute_to_relative = dataset.absolute_to_relative_idx
    if absolute_to_relative is None:
        raise ValueError("Selected evaluation dataset lacks an index mapping")
    windows = build_windows(meta.episodes, episodes, absolute_to_relative)
    valid_sampler = EpisodeAwareSampler(
        meta.episodes["dataset_from_index"],
        meta.episodes["dataset_to_index"],
        episode_indices_to_use=episodes,
        drop_n_first_frames=getattr(cfg, "drop_n_first_frames", 0),
        drop_n_last_frames=getattr(cfg, "drop_n_last_frames", 0),
        absolute_to_relative_idx=absolute_to_relative,
    )
    loss_loader = make_loader(
        dataset, batch_size=LOSS_BATCH_SIZE, num_workers=num_workers, device=device, sampler=valid_sampler
    )
    action_loader = make_loader(
        Subset(dataset, [item["relative_index"] for item in windows]),
        batch_size=ACTION_BATCH_SIZE,
        num_workers=num_workers,
        device=device,
    )
    print(f"Evaluating {model_dir} on {len(episodes)} held-out episodes", flush=True)
    loss = evaluate_losses(policy, loss_loader, meta.camera_keys, rename, pre)
    actions = evaluate_actions(policy, action_loader, windows, meta.camera_keys, rename, pre, post)
    assets = sorted(
        path for path in model_dir.iterdir() if path.is_file() and path.suffix in {".json", ".safetensors"}
    )
    identity = {
        "checkpoint": str(model_dir),
        "checkpoint_file_sha256": {path.name: file_hash(path) for path in assets},
        "split_manifest_sha256": file_hash(split_path),
        "dataset_root": str(root),
        "dataset_repo_id": repo_id,
        "eval_episodes": episodes,
        "windows": windows,
        "seeds": list(EVAL_SEEDS),
        "action_batch_size": ACTION_BATCH_SIZE,
        "loss_batch_size": LOSS_BATCH_SIZE,
        "scheduler": dict(policy.diffusion.noise_scheduler.config),
        "inference_steps": policy.diffusion.num_inference_steps,
        "device": device,
    }
    result = {"eval_loss": loss, "action_errors": actions}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", identity)
    write_json(output / "results.json", result)
    print(
        json.dumps({"checkpoint": str(model_dir), "eval_loss": loss, "actions": actions["overall"]}, indent=2)
    )
    return result


def main() -> None:
    """解析命令行；完整结果和采样身份分别保存为 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output_dir, args.device, args.num_workers)


if __name__ == "__main__":
    main()
