#!/usr/bin/env python

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

"""Offline checkpoint validation for the RM75 SmolVLA fine-tune.

Compares every checkpoint in a training output directory under *identical* fixed
conditions — the same held-out episode window, the same flow-matching noise and
timesteps (seeded once, cached, and reused for every checkpoint) — and selects the
checkpoint with the lowest fixed-condition flow-matching loss.

Per checkpoint it reports:
  - validation flow-matching loss (padding-masked),
  - 64x8 action-chunk MAE, overall and per dimension (normalized action space),
  - out-of-[-1,1]-bounds ratio of the predicted chunk,
  - NaN / Inf counts of the predicted chunk,
  - checkpoint load time and mean inference time per action chunk.

Outputs ``validation_report.json`` and ``best_checkpoint.txt`` in the training output
directory. Checkpoints are only read, never deleted.

NOTE: Offline metrics measure how well a checkpoint predicts held-out demonstrations.
They do NOT measure real-robot task success. Always confirm the selected checkpoint on
the actual RM75 arm before trusting it.

Example::

    uv run examples/rm75/validate_checkpoints.py \\
        --train-output outputs/rm75_smolvla_5090x2 \\
        --dataset-root data/rm75_toy_robot_lerobot \\
        --repo-id rm75_toy_robot
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.common.flow_matching import sample_time_beta
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedConfig

# The validation window and the flow-matching randomness are drawn once with fixed seeds
# and cached for the whole run, so every checkpoint sees byte-identical conditions.
WINDOW_SEED = 1000

# Mirrors SmolVLAModel.sample_time (alpha=1.5, beta=1.0, scale=0.999, offset=0.001).
TIME_BETA_PARAMS = {"alpha": 1.5, "beta": 1.0, "scale": 0.999, "offset": 0.001}


def find_checkpoints(train_output: Path, explicit: list[str] | None) -> list[Path]:
    """Resolve checkpoint directories (each containing a pretrained_model/ subtree)."""
    if explicit:
        candidates = [Path(p) for p in explicit]
    else:
        ckpt_root = train_output / "checkpoints"
        if not ckpt_root.is_dir():
            raise SystemExit(f"No checkpoints directory under {train_output}")
        candidates = sorted(p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit())
    resolved = []
    for cand in candidates:
        model_dir = cand / "pretrained_model"
        model_dir = model_dir if (model_dir / "config.json").is_file() else cand
        if not (model_dir / "config.json").is_file():
            print(f"[warn] Skipping {cand}: no config.json (not a pretrained_model directory)")
            continue
        resolved.append(model_dir)
    if not resolved:
        raise SystemExit("No valid checkpoints found.")
    return resolved


def load_rename_map(model_dir: Path) -> dict[str, str]:
    """Load the dataset-to-policy feature mapping saved with a checkpoint."""
    train_config_path = model_dir / "train_config.json"
    if not train_config_path.is_file():
        return {}

    train_config = json.loads(train_config_path.read_text())
    rename_map = train_config.get("rename_map", {})
    if not isinstance(rename_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in rename_map.items()
    ):
        raise ValueError(f"Invalid rename_map in {train_config_path}: {rename_map!r}")
    return rename_map


def eval_episode_indices(meta: LeRobotDatasetMetadata, eval_split: float) -> list[int]:
    """Same held-out episodes the trainer uses: the last ceil(n * split) episodes."""
    n_eval = math.ceil(meta.total_episodes * eval_split)
    return list(range(meta.total_episodes - n_eval, meta.total_episodes))


def build_fixed_batches(
    args,
    policy_cfg: PreTrainedConfig,
    device: torch.device,
) -> tuple[list[dict], dict, LeRobotDatasetMetadata]:
    """Build the fixed validation window: preprocessed batches + fixed noise/time.

    The window, noise and timesteps are drawn once here with fixed seeds; the returned
    tensors are reused verbatim for every checkpoint.
    """
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    episodes = eval_episode_indices(meta, args.eval_split)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(policy_cfg, meta),
    )

    chunk = policy_cfg.chunk_size
    episode_ranges = []
    for ep_idx in episodes:
        ep_from = int(meta.episodes["dataset_from_index"][ep_idx])
        ep_to = int(meta.episodes["dataset_to_index"][ep_idx])
        last_start = ep_to - chunk  # window must fit a full 64-step chunk (no tail padding)
        if last_start < ep_from:
            print(f"[warn] Episode {ep_idx} shorter than chunk_size={chunk}; skipped")
            continue
        episode_ranges.append((ep_idx, ep_from, last_start))
    if not episode_ranges:
        raise SystemExit("The fixed validation window is empty.")

    # Allocate the requested window explicitly across episodes. Building a long list
    # and truncating it would silently overweight the first episodes in the split.
    sample_counts = {}
    starts = []
    base_count, remainder = divmod(args.num_samples, len(episode_ranges))
    for episode_number, (ep_idx, ep_from, last_start) in enumerate(episode_ranges):
        count = base_count + int(episode_number < remainder)
        sample_counts[str(ep_idx)] = count
        if count == 0:
            continue
        if count == 1:
            episode_starts = [(ep_from + last_start) // 2]
        else:
            episode_starts = np.linspace(ep_from, last_start, count, dtype=np.int64).tolist()
        starts.extend(episode_starts)

    print(
        f"[info] Fixed validation window: {len(starts)} samples "
        f"balanced across {len(episode_ranges)} held-out episodes "
        f"{episodes[0]}..{episodes[-1]}"
    )

    # Dataset indices are absolute in metadata but relative when an episode subset is loaded.
    starts = [dataset.absolute_to_relative_idx[index] for index in starts]

    max_action_dim = policy_cfg.max_action_dim
    batches = []
    for offset in range(0, len(starts), args.batch_size):
        window = starts[offset : offset + args.batch_size]
        samples = [dataset[i] for i in window]
        batch = default_collate(samples)

        # Draw on CPU with isolated RNG state, then move. These tensors are reused unchanged.
        with torch.random.fork_rng():
            torch.manual_seed(WINDOW_SEED + offset)
            noise = torch.randn(len(samples), chunk, max_action_dim, dtype=torch.float32).to(device)
            flow_time = sample_time_beta(len(samples), device, **TIME_BETA_PARAMS)
        batches.append({"batch": batch, "noise": noise, "flow_time": flow_time})

    window_info = {
        "num_samples": len(starts),
        "num_batches": len(batches),
        "episodes": episodes,
        "samples_per_episode": sample_counts,
        "window_seed": WINDOW_SEED,
        "time_beta": TIME_BETA_PARAMS,
    }
    return batches, window_info, meta


def validate_checkpoint(
    model_dir: Path,
    batches: list[dict],
    device: torch.device,
    meta: LeRobotDatasetMetadata,
    rename_map: dict[str, str],
) -> dict:
    """Evaluate one checkpoint under the fixed conditions."""
    t0 = time.perf_counter()
    policy_cfg = PreTrainedConfig.from_pretrained(str(model_dir))
    policy_cfg.pretrained_path = str(model_dir)
    policy_cfg.device = str(device)
    policy = make_policy(policy_cfg, ds_meta=meta, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(model_dir),
        dataset_stats=meta.stats,
        dataset_meta=meta,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    load_time_s = time.perf_counter() - t0

    state_dim = None
    loss_sum, valid_steps = 0.0, 0
    abs_err_sum = torch.zeros(policy_cfg.max_action_dim, dtype=torch.float64)
    per_dim_valid = 0
    oob_count, valid_count = 0, 0
    nan_count, inf_count = 0, 0
    inference_times = []

    with torch.no_grad():
        for item in batches:
            raw_batch = item["batch"]
            raw_action = raw_batch["action"].float()
            raw_is_pad = raw_batch.get("action_is_pad")
            batch = preprocessor(
                {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in raw_batch.items()
                }
            )
            noise, flow_time = item["noise"], item["flow_time"]

            loss, _ = policy.forward(batch, noise=noise, time=flow_time)
            is_pad = batch.get("action_is_pad")
            if is_pad is None:
                is_pad = torch.zeros(
                    batch["action"].shape[:2], dtype=torch.bool, device=batch["action"].device
                )
            steps_valid = int((~is_pad).sum())
            loss_sum += float(loss) * steps_valid
            valid_steps += steps_valid

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            pred = policy.predict_action_chunk(batch, noise=noise)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append((time.perf_counter() - t1) / batch["action"].shape[0])

            pred = postprocessor(pred).float().cpu()
            nan_count += int(torch.isnan(pred).sum())
            inf_count += int(torch.isinf(pred).sum())

            gt = raw_action.cpu()
            is_pad = (
                raw_is_pad.cpu() if raw_is_pad is not None else torch.zeros(gt.shape[:2], dtype=torch.bool)
            )
            state_dim = gt.shape[-1]
            pred = pred[:, :, :state_dim]
            mask = ~is_pad
            mask_expanded = mask.unsqueeze(-1)

            valid_count += int(mask.sum()) * state_dim
            oob_count += int((pred.abs() > 1.0)[mask_expanded.expand_as(pred)].sum())

            abs_err = (pred - gt).abs() * mask_expanded
            abs_err_sum[:state_dim] += abs_err.sum(dim=(0, 1)).double().cpu()
            per_dim_valid += int(mask.sum())

    report = {
        "checkpoint": str(model_dir),
        "step": int(model_dir.parent.name) if model_dir.parent.name.isdigit() else None,
        "val_loss": loss_sum / max(valid_steps, 1),
        "chunk_mae_overall": float(abs_err_sum[:state_dim].sum() / max(valid_count, 1)),
        "chunk_mae_per_dim": [float(abs_err_sum[d] / max(per_dim_valid, 1)) for d in range(state_dim)],
        "out_of_bounds_ratio": oob_count / max(valid_count, 1),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "load_time_s": load_time_s,
        "inference_ms_per_chunk": 1000.0 * float(np.mean(inference_times)) if inference_times else None,
        "num_action_steps_evaluated": valid_steps,
        "action_dim": state_dim,
    }
    del policy, preprocessor, postprocessor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train-output", type=Path, required=True, help="Training output directory holding checkpoints/."
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="Local root of the converted RM75 LeRobot dataset."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="rm75_toy_robot",
        help="Dataset repo id recorded in the metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="Explicit checkpoint dirs (default: every checkpoint under --train-output).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Fixed window size in observation samples (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Evaluation batch size (default: %(default)s)."
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Held-out fraction; the last ceil(n*split) episodes are used (default: %(default)s).",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (default: cuda if available, else cpu)."
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] Device: {device}")

    checkpoints = find_checkpoints(args.train_output, args.checkpoints)
    print(f"[info] {len(checkpoints)} checkpoints to compare.")

    rename_map = load_rename_map(checkpoints[0])
    if rename_map:
        print(f"[info] Feature rename map: {rename_map}")

    # The first checkpoint only provides the config used to build the fixed window.
    first_cfg = PreTrainedConfig.from_pretrained(str(checkpoints[0]))
    batches, window_info, meta = build_fixed_batches(args, first_cfg, device)

    reports = []
    for i, model_dir in enumerate(checkpoints):
        print(f"[validate] ({i + 1}/{len(checkpoints)}) {model_dir}")
        report = validate_checkpoint(model_dir, batches, device, meta, rename_map)
        print(
            f"[validate]   val_loss={report['val_loss']:.6f} mae={report['chunk_mae_overall']:.6f} "
            f"oob={report['out_of_bounds_ratio']:.2%} load={report['load_time_s']:.1f}s "
            f"infer={report['inference_ms_per_chunk']:.1f}ms/chunk"
        )
        reports.append(report)

    best = min(reports, key=lambda r: (r["val_loss"], r["step"] if r["step"] is not None else 0))
    train_output: Path = args.train_output
    (train_output / "validation_report.json").write_text(
        json.dumps(
            {
                "note": (
                    "Offline metrics measure held-out-demo prediction quality only; "
                    "they do not measure real-robot success rate."
                ),
                "dataset_root": str(args.dataset_root),
                "repo_id": args.repo_id,
                "device": str(device),
                "conditions": window_info,
                "checkpoints": reports,
                "best": {"checkpoint": best["checkpoint"], "val_loss": best["val_loss"]},
            },
            indent=2,
        )
    )
    best_txt = (
        f"{best['checkpoint']}\n"
        f"step: {best['step']}\n"
        f"val_loss: {best['val_loss']:.6f}\n"
        f"chunk_mae_overall: {best['chunk_mae_overall']:.6f}\n"
    )
    (train_output / "best_checkpoint.txt").write_text(best_txt)

    print(f"[best] {best['checkpoint']} (val_loss={best['val_loss']:.6f})")
    print(
        f"[done] Wrote {train_output / 'validation_report.json'} and {train_output / 'best_checkpoint.txt'}"
    )
    print(
        "[disclaimer] Offline validation does NOT equal real-robot success rate. "
        "Confirm the selected checkpoint on the actual RM75 arm."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
