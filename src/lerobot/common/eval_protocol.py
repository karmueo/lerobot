#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""离线验证口径及其续训一致性检查。"""

from pathlib import Path
from typing import Any

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.io_utils import load_json

SPLIT_MANIFEST = "split_manifest.json"
EVAL_PROTOCOL = "evaluation_protocol.json"


def build_eval_protocol(
    cfg: TrainPipelineConfig, manifest: dict[str, Any], dp_world_size: int
) -> dict[str, Any]:
    """记录影响 best 损失可比性的划分、噪声、采样和权重类型。"""
    active_cfg = cfg.trainable_config
    return {
        "version": 1,
        "dataset_repo_id": manifest["dataset"]["repo_id"],
        "dataset_revision": cfg.dataset.revision,
        "train_episodes": manifest["train_episodes"],
        "eval_episodes": manifest["eval_episodes"],
        "split_strategy": manifest["split_strategy"],
        "split_seed": manifest["split_seed"],
        "eval_seed": cfg.eval_seed,
        "weight_type": "ema" if cfg.eval_use_ema else "live",
        "batch_size": cfg.batch_size,
        "dp_world_size": dp_world_size,
        "max_eval_samples": cfg.max_eval_samples,
        "drop_n_first_frames": getattr(active_cfg, "drop_n_first_frames", 0),
        "drop_n_last_frames": getattr(active_cfg, "drop_n_last_frames", 0),
        "rename_map": cfg.rename_map,
    }


def validate_eval_protocol(
    checkpoint_dir: Path,
    expected: dict[str, Any],
    *,
    allow_legacy: bool,
) -> None:
    """续训时拒绝将不同验证口径的损失与历史 best 比较。"""
    path = checkpoint_dir / EVAL_PROTOCOL
    if not path.is_file():
        if allow_legacy:
            return
        raise ValueError(
            f"Checkpoint {checkpoint_dir} has no evaluation protocol; only the legacy tail split "
            "with an unset eval_seed and live weights may resume it. Start a new run instead."
        )
    saved = load_json(path)
    if saved != expected:
        differing = (
            sorted(key for key in set(expected) | set(saved) if expected.get(key) != saved.get(key))
            if isinstance(saved, dict)
            else ["format"]
        )
        raise ValueError(
            f"Evaluation protocol mismatch in {path}: {', '.join(differing)}. "
            "Resume with the recorded protocol or start a new run."
        )
