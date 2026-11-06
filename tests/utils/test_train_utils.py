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

import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from lerobot.common import train_utils
from lerobot.common.train_utils import (
    commit_best_and_last_checkpoint,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_best_checkpoint_state,
    load_training_metadata,
    prepare_best_and_last_checkpoint,
    push_checkpoint_to_hub,
    restore_best_checkpoint_link,
    save_training_metadata,
    save_training_state,
    should_save_checkpoint,
    update_last_checkpoint,
)
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    RNG_STATE,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.import_utils import _datasets_available

if TYPE_CHECKING or _datasets_available:
    from lerobot.scripts.lerobot_train import make_dataloaders


def test_get_step_identifier():
    assert get_step_identifier(5, 1000) == "000005"
    assert get_step_identifier(123, 100_000) == "000123"
    assert get_step_identifier(456789, 1_000_000) == "0456789"


def test_should_save_checkpoint():
    # Periodic checkpoints land on multiples of save_freq.
    assert should_save_checkpoint(10, save_freq=10, total_steps=100) is True
    assert should_save_checkpoint(5, save_freq=10, total_steps=100) is False
    # The final step always saves, even when it is not a multiple of save_freq.
    assert should_save_checkpoint(100, save_freq=30, total_steps=100) is True
    # save_freq <= 0 disables periodic saving without raising ZeroDivisionError.
    assert should_save_checkpoint(1, save_freq=0, total_steps=100) is False
    assert should_save_checkpoint(100, save_freq=0, total_steps=100) is True
    assert should_save_checkpoint(1, save_freq=-1, total_steps=100) is False


def test_get_step_checkpoint_dir():
    output_dir = Path("/checkpoints")
    step_dir = get_step_checkpoint_dir(output_dir, 1000, 5)
    assert step_dir == output_dir / CHECKPOINTS_DIR / "000005"


def make_cfg(batch_size: int = 32) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"), batch_size=batch_size)
    cfg.parallelism.resolve(1)
    return cfg


def test_save_training_metadata_writes_the_step_file(tmp_path):
    save_training_metadata(5000, tmp_path, make_cfg())
    assert (tmp_path / TRAINING_STEP).is_file()


def test_save_training_state_records_topology(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, make_cfg(batch_size=32), optimizer, scheduler)
    metadata = load_training_metadata(tmp_path / TRAINING_STATE_DIR)
    assert metadata["step"] == 10
    assert metadata["dp_world_size"] == 1
    assert metadata["batch_size"] == 32


def test_update_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == checkpoint


def _save_managed_checkpoint(output_dir, step, best_loss, best_step, best_dir, *, best, last):
    # 用轻量文件区分每次保存，检查点链接仍走真实文件系统操作。
    checkpoint = get_step_checkpoint_dir(output_dir, 5000, step)
    pending = prepare_best_and_last_checkpoint(checkpoint)
    (pending / "model.txt").write_text(str(step))
    return commit_best_and_last_checkpoint(
        pending, checkpoint, best_loss, best_step, best_dir, update_best=best, update_last=last
    )


def test_best_and_last_keep_current_snapshots_and_resume_history(tmp_path):
    first = _save_managed_checkpoint(tmp_path, 1000, 0.8, 1000, None, best=True, last=False)
    best_link = first.parent / "best"
    last_link = first.parent / "last"
    assert best_link.resolve() == first
    assert not last_link.exists()

    second = _save_managed_checkpoint(tmp_path, 2000, 0.8, 1000, first.name, best=False, last=True)
    assert best_link.resolve() == first
    assert last_link.resolve() == second
    assert load_best_checkpoint_state(tmp_path, second) == (0.8, 1000, first.name)

    third = _save_managed_checkpoint(tmp_path, 3000, 0.7, 3000, first.name, best=True, last=False)
    assert best_link.resolve() == third
    assert last_link.resolve() == second
    assert not first.exists()

    fourth = _save_managed_checkpoint(tmp_path, 4000, 0.6, 4000, third.name, best=True, last=True)
    assert best_link.resolve() == last_link.resolve() == fourth
    assert (best_link / "model.txt").read_text() == "4000"
    assert not second.exists()
    assert not third.exists()
    assert load_best_checkpoint_state(tmp_path, fourth) == (0.6, 4000, fourth.name)


def test_best_and_last_resume_and_unowned_backups(tmp_path):
    first = _save_managed_checkpoint(tmp_path, 1000, 0.8, 1000, None, best=True, last=False)
    second = _save_managed_checkpoint(tmp_path, 2000, 0.8, 1000, first.name, best=False, last=True)
    best_link = first.parent / "best"
    best_link.unlink()
    best_loss, best_step, best_dir = load_best_checkpoint_state(tmp_path, second)
    assert (best_loss, best_step, best_dir) == (0.8, 1000, first.name)
    restore_best_checkpoint_link(tmp_path, best_dir)
    assert best_link.resolve() == first

    # 即使复制了受管状态，名称不匹配的备份仍归用户所有。
    backup = first.parent / "000999"
    shutil.copytree(first, backup)
    named_backup = first.parent / "manual-backup"
    shutil.copytree(first, named_backup)
    # 从旧的 best 续训再次保存同一步时，不能覆盖现有 last。
    replacement = _save_managed_checkpoint(tmp_path, 2000, 0.7, 2000, first.name, best=True, last=False)
    assert replacement != second
    assert (first.parent / "last").resolve() == second
    assert backup.is_dir()
    assert named_backup.is_dir()


def test_best_and_last_recover_interrupted_save(tmp_path, monkeypatch):
    first = _save_managed_checkpoint(tmp_path, 1000, 0.8, 1000, None, best=True, last=True)
    checkpoint = get_step_checkpoint_dir(tmp_path, 5000, 2000)
    worker_pending = prepare_best_and_last_checkpoint(checkpoint, is_main=False)
    assert not worker_pending.exists()
    pending = prepare_best_and_last_checkpoint(checkpoint)
    assert pending == worker_pending
    (pending / "partial").touch()
    next_checkpoint = get_step_checkpoint_dir(tmp_path, 5000, 3000)
    assert prepare_best_and_last_checkpoint(next_checkpoint) == pending
    assert not (pending / "partial").exists()
    assert (first.parent / "best").resolve() == first
    assert (first.parent / "last").resolve() == first

    original_replace = train_utils.os.replace

    def fail_best_link(src, dst):
        if Path(dst) == first.parent / "best":
            raise OSError("模拟链接切换失败")
        original_replace(src, dst)

    monkeypatch.setattr(train_utils.os, "replace", fail_best_link)
    with pytest.raises(OSError, match="模拟链接切换失败"):
        commit_best_and_last_checkpoint(
            pending, next_checkpoint, 0.7, 3000, first.name, update_best=True, update_last=True
        )
    assert (first.parent / "best").resolve() == first
    assert (first.parent / "last").resolve() == first
    monkeypatch.setattr(train_utils.os, "replace", original_replace)
    latest = _save_managed_checkpoint(tmp_path, 4000, 0.6, 4000, first.name, best=True, last=True)
    assert not next_checkpoint.exists()
    assert (first.parent / "best").resolve() == latest
    assert (first.parent / "last").resolve() == latest


def test_best_and_last_no_evaluation_yet(tmp_path):
    checkpoint = _save_managed_checkpoint(tmp_path, 1000, math.inf, None, None, best=False, last=True)
    assert load_best_checkpoint_state(tmp_path, checkpoint) == (math.inf, None, None)
    assert (checkpoint.parent / "last").resolve() == checkpoint


# save_checkpoint round-trips (all formats, real policies) live in
# tests/common/test_checkpoint_save_resume.py.


def test_save_training_state_layout(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, make_cfg(), optimizer, scheduler)
    assert (tmp_path / TRAINING_STATE_DIR).is_dir()
    assert (tmp_path / TRAINING_STATE_DIR / TRAINING_STEP).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / RNG_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_PARAM_GROUPS).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / SCHEDULER_STATE).is_file()


# The two-phase resume (resume_before_prepare / resume_after_prepare) is covered in
# tests/common/test_checkpoint_save_resume.py with real policies and optimizer state.


def test_push_checkpoint_to_hub_creates_repo_and_uploads(tmp_path, monkeypatch):
    ckpt = tmp_path / "010000"
    (ckpt / "pretrained_model").mkdir(parents=True)
    api = MagicMock()
    monkeypatch.setattr("lerobot.common.train_utils.HfApi", lambda *a, **k: api)
    push_checkpoint_to_hub(ckpt, "user/run", private=True)
    api.create_repo.assert_called_once()
    assert api.create_repo.call_args.kwargs["private"] is True
    assert api.create_repo.call_args.kwargs["repo_type"] == "model"
    api.upload_folder.assert_called_once()
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "user/run"
    assert kwargs["repo_type"] == "model"
    assert kwargs["path_in_repo"] == "checkpoints/010000"
    assert kwargs["folder_path"] == str(ckpt)
    assert kwargs["commit_message"] == "checkpoint 010000"
    # A tag named after the checkpoint step is created so the checkpoint can be
    # recovered with --policy.pretrained_revision instead of a commit sha.
    api.create_tag.assert_called_once()
    tag_kwargs = api.create_tag.call_args.kwargs
    assert tag_kwargs["tag"] == "010000"
    assert tag_kwargs["revision"] == api.upload_folder.return_value.oid
    assert tag_kwargs["repo_type"] == "model"
    assert tag_kwargs["exist_ok"] is True


def test_push_checkpoint_to_hub_defaults_to_hub_default_visibility(tmp_path, monkeypatch):
    ckpt = tmp_path / "010000"
    (ckpt / "pretrained_model").mkdir(parents=True)
    api = MagicMock()
    monkeypatch.setattr("lerobot.common.train_utils.HfApi", lambda *a, **k: api)
    push_checkpoint_to_hub(ckpt, "user/run")
    api.create_repo.assert_called_once()
    assert api.create_repo.call_args.kwargs["private"] is None


def test_resolve_resume_checkpoint_downloads_latest_and_links(tmp_path, monkeypatch):
    out = tmp_path / "run"

    def fake_snapshot_download(repo_id, repo_type, allow_patterns, local_dir):
        # Mimic the Hub layout the real download materializes locally.
        assert allow_patterns == "checkpoints/020000/*"
        (Path(local_dir) / "checkpoints" / "020000" / "pretrained_model").mkdir(parents=True)
        return local_dir

    monkeypatch.setattr("lerobot.common.train_utils.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        "lerobot.common.train_utils.find_latest_hub_checkpoint", lambda repo_id: "checkpoints/020000"
    )

    checkpoint_dir = train_utils.resolve_resume_checkpoint("u/run", out)

    assert checkpoint_dir == out / CHECKPOINTS_DIR / "020000"
    last = out / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK
    assert last.is_symlink()
    # `last` points at the downloaded step dir.
    assert (last.parent / last.readlink()).resolve() == checkpoint_dir.resolve()


def test_resolve_resume_checkpoint_raises_without_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.common.train_utils.find_latest_hub_checkpoint", lambda repo_id: None)
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        train_utils.resolve_resume_checkpoint("u/run", tmp_path / "run")


@pytest.mark.parametrize("best_name", ["001000", "001000-1", "002000", None])
def test_hub_resume_restores_best_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, best_name: str | None
) -> None:
    """模拟 Hub 只按请求下载目录，覆盖 best 与 last 不同步及尚未验证的情况。"""
    remote, local = tmp_path / "remote", tmp_path / "local"
    latest = remote / CHECKPOINTS_DIR / "002000"
    best = remote / CHECKPOINTS_DIR / best_name if best_name is not None else None
    # 先保存 best，再保存 last，模拟真实训练中的引用和清理顺序。
    checkpoints = [best, latest] if best is not None and best != latest else [latest]
    for checkpoint in checkpoints:
        pending = prepare_best_and_last_checkpoint(checkpoint)
        train_utils.write_json({"eval_seed": 1000}, pending / train_utils.EVAL_PROTOCOL)
        commit_best_and_last_checkpoint(
            pending,
            checkpoint,
            0.5 if best else math.inf,
            1000 if best else None,
            best_name,
            update_best=checkpoint == best,
            update_last=checkpoint == latest,
        )
    downloads: list[str] = []

    def download(repo_id: str, repo_type: str, allow_patterns: str, local_dir: str) -> str:
        downloads.append(allow_patterns)
        relative = allow_patterns.removesuffix("/*")
        shutil.copytree(remote / relative, Path(local_dir) / relative)
        return local_dir

    monkeypatch.setattr(train_utils, "find_latest_hub_checkpoint", lambda _: "checkpoints/002000")
    monkeypatch.setattr(train_utils, "snapshot_download", download)
    resumed = train_utils.resolve_resume_checkpoint("user/run", local)
    assert resumed == local / CHECKPOINTS_DIR / "002000"
    assert (local / CHECKPOINTS_DIR / "last").resolve() == resumed
    expected = ["checkpoints/002000/*"]
    if best_name is not None:
        assert (local / CHECKPOINTS_DIR / "best").resolve() == local / CHECKPOINTS_DIR / best_name
        assert load_best_checkpoint_state(local, resumed) == (0.5, 1000, best_name)
        if best_name != "002000":
            expected.append(f"checkpoints/{best_name}/*")
    else:
        assert not (local / CHECKPOINTS_DIR / "best").exists()
        assert load_best_checkpoint_state(local, resumed) == (math.inf, None, None)
    assert downloads == expected


@pytest.mark.parametrize("failure", ["missing", "protocol", "invalid_name"])
def test_hub_resume_rejects_invalid_best_before_linking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """缺失、协议不一致或非法路径的历史 best 不得发布为可续训链接。"""
    downloads: list[str] = []

    def download(repo_id: str, repo_type: str, allow_patterns: str, local_dir: str) -> str:
        downloads.append(allow_patterns)
        checkpoint = Path(local_dir) / allow_patterns.removesuffix("/*")
        if checkpoint.name == "002000":
            train_utils.write_json(
                {
                    "best_eval_loss": 0.5,
                    "best_step": 1000,
                    "best_dir": "../outside" if failure == "invalid_name" else "001000",
                },
                checkpoint / TRAINING_STATE_DIR / "best_checkpoint.json",
            )
            train_utils.write_json({"eval_seed": 1000}, checkpoint / train_utils.EVAL_PROTOCOL)
        elif failure == "protocol":
            train_utils.write_json({"eval_seed": 2000}, checkpoint / train_utils.EVAL_PROTOCOL)
        return local_dir

    monkeypatch.setattr(train_utils, "find_latest_hub_checkpoint", lambda _: "checkpoints/002000")
    monkeypatch.setattr(train_utils, "snapshot_download", download)
    message = {
        "missing": "no evaluation protocol",
        "protocol": "Evaluation protocol mismatch",
        "invalid_name": "Invalid best checkpoint state",
    }[failure]
    with pytest.raises(ValueError, match=message):
        train_utils.resolve_resume_checkpoint("user/run", tmp_path)
    for name in ("best", "last"):
        assert not (tmp_path / CHECKPOINTS_DIR / name).is_symlink()
    assert len(downloads) == (1 if failure == "invalid_name" else 2)


@pytest.mark.dataset
@pytest.mark.skipif(not _datasets_available, reason="requires datasets")
@pytest.mark.parametrize("frames,drop_first,drop_last,max_eval_samples", [(12, 2, 3, 3), (64, 0, 32, 0)])
def test_dataloaders_filter_boundaries_without_consuming_policy_rng(
    frames, drop_first, drop_last, max_eval_samples
):
    policy = SimpleNamespace(drop_n_first_frames=drop_first, drop_n_last_frames=drop_last)
    expected = list(range(drop_first, frames - drop_last))

    class Dataset(torch.utils.data.Dataset):
        episodes = [0]
        absolute_to_relative_idx = None
        meta = SimpleNamespace(
            episodes={"dataset_from_index": [0], "dataset_to_index": [frames]},
            has_language_columns=False,
        )
        hf_dataset = SimpleNamespace(
            data=SimpleNamespace(column=lambda name: SimpleNamespace(to_numpy=lambda: np.zeros(frames)))
        )

        def __len__(self):
            return frames

        def __getitem__(self, index):
            return index

    cfg = SimpleNamespace(
        trainable_config=policy,
        dataset=SimpleNamespace(streaming=False),
        resume=False,
        seed=42,
        num_workers=0,
        batch_size=2,
        prefetch_factor=None,
        persistent_workers=False,
        dataloader_multiprocessing_context=None,
        max_eval_samples=max_eval_samples,
    )
    train, evaluation = make_dataloaders(cfg, Dataset(), Dataset(), 0, SimpleNamespace(device_type="cpu"))
    assert sorted(train.sampler) == expected
    rng = torch.get_rng_state().clone()
    next(iter(train))
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.cat(list(evaluation)).tolist() == (
        expected[:max_eval_samples] if max_eval_samples else expected
    )
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.dataset
@pytest.mark.skipif(not _datasets_available, reason="requires datasets")
@pytest.mark.parametrize(("num_workers", "expected"), [(0, 0), (1, 1), (4, 4), (16, 8)])
def test_streaming_dataloader_uses_num_workers_capped_by_rank_episodes(num_workers, expected):
    class Dataset(torch.utils.data.IterableDataset):
        meta = SimpleNamespace(has_language_columns=False)

        def set_data_parallel_shard(self, rank, world_size):
            pass

        def num_episodes_for_rank(self, rank, world_size):
            return 8

        def num_frames_for_rank(self, rank, world_size, num_workers):
            assert num_workers <= 8
            return 80

        def __iter__(self):
            return iter(())

    cfg = SimpleNamespace(
        trainable_config=None,
        dataset=SimpleNamespace(streaming=True),
        resume=False,
        seed=42,
        num_workers=num_workers,
        batch_size=2,
        prefetch_factor=2,
        persistent_workers=False,
        dataloader_multiprocessing_context=None,
        max_eval_samples=0,
    )
    parallel_dims = SimpleNamespace(device_type="cpu", dp_rank=0, dp_world_size=1)
    train, _ = make_dataloaders(cfg, Dataset(), None, 0, parallel_dims)

    assert train.num_workers == expected


@pytest.mark.dataset
@pytest.mark.skipif(not _datasets_available, reason="requires datasets")
def test_streaming_dataloader_receives_the_trainers_data_parallel_rank():
    class StreamingDataset(torch.utils.data.IterableDataset):
        meta = SimpleNamespace(has_language_columns=False)
        set_data_parallel_shard = MagicMock()
        num_episodes_for_rank = MagicMock(return_value=1)
        num_frames_for_rank = MagicMock(return_value=1)

        def __iter__(self):
            return iter([])

    cfg = SimpleNamespace(
        trainable_config=SimpleNamespace(),
        dataset=SimpleNamespace(streaming=True),
        resume=False,
        seed=42,
        num_workers=0,
        batch_size=2,
        prefetch_factor=None,
        persistent_workers=False,
        dataloader_multiprocessing_context=None,
    )
    parallel_dims = SimpleNamespace(device_type="cpu", dp_rank=3, dp_world_size=4)

    make_dataloaders(cfg, StreamingDataset(), None, 0, parallel_dims)

    StreamingDataset.set_data_parallel_shard.assert_called_once_with(3, 4)
