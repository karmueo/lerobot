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
"""验证 HandUMI 离线评估的切分、动作对齐和物理误差口径。"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# 脚本导入数据集工厂；收集阶段先跳过没有 dataset extra 的环境。
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from scripts.eval_dp_handumi_tcp import (
    action_errors,
    build_windows,
    evaluate_actions,
    evaluate_losses,
    resolve_checkpoint,
    summarize_actions,
    validate_split,
)

pytestmark = pytest.mark.dataset


@pytest.fixture(autouse=True)
def cpu_only_random_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    """这些测试只使用 CPU，避免固定随机数时触碰其他设备的状态。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: None)


def test_resolve_checkpoint_uses_saved_weight_type_and_nearby_manifest(tmp_path: Path) -> None:
    """step 自动选择原 best 权重；复制出来的模型也能读取自身清单。"""
    step = tmp_path / "003000"
    live, ema = step / "pretrained_model", step / "pretrained_model_ema"
    live.mkdir(parents=True)
    ema.mkdir()
    for path in (live, ema):
        (path / "config.json").write_text("{}")
    (live / "train_config.json").write_text(json.dumps({"eval_use_ema": True}))
    (step / "split_manifest.json").write_text("{}")
    best = tmp_path / "best"
    best.symlink_to(step, target_is_directory=True)
    assert resolve_checkpoint(best) == (ema.resolve(), (step / "split_manifest.json").resolve())
    (ema / "split_manifest.json").write_text("{}")
    assert resolve_checkpoint(ema) == (ema.resolve(), (ema / "split_manifest.json").resolve())
    # 显式指定 live 时不得因 train_config 的 EMA 标记悄悄切换权重。
    assert resolve_checkpoint(live)[0] == live.resolve()


def test_validate_split_rejects_overlap_and_different_dataset(tmp_path: Path) -> None:
    manifest = {
        "version": 1,
        "dataset": {"repo_id": "local/demo", "root": str(tmp_path)},
        "train_episodes": [0, 1],
        "eval_episodes": [2, 3],
    }
    training = {"dataset": {"repo_id": "local/demo", "root": str(tmp_path)}}
    assert validate_split(manifest, training) == (tmp_path.resolve(), "local/demo", [2, 3])
    with pytest.raises(ValueError, match="overlap"):
        validate_split({**manifest, "eval_episodes": [1, 3]}, training)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_split({**manifest, "eval_episodes": [2, 2]}, training)
    with pytest.raises(ValueError, match="identifiers differ"):
        validate_split(manifest, {"dataset": {"repo_id": "other", "root": str(tmp_path)}})


@pytest.mark.parametrize("n_obs_steps", [2, 3])
def test_windows_are_distinct_inside_episode_and_map_subset_indices(n_obs_steps: int) -> None:
    """最早采样点必须有完整观测历史，最晚采样点必须有完整动作标签。"""
    rows = [
        {"dataset_from_index": 0, "dataset_to_index": 30},
        {"dataset_from_index": 30, "dataset_to_index": 50},
        {"dataset_from_index": 50, "dataset_to_index": 80},
    ]
    mapping = {absolute: absolute - 30 for absolute in range(30, 50)}
    windows = build_windows(rows, [1], mapping, n_obs_steps=n_obs_steps)
    anchors = [item["anchor"] for item in windows]
    assert len(anchors) == len(set(anchors)) == 8
    assert anchors[0] == 30 + n_obs_steps - 1 and anchors[-1] == 42
    assert [item["relative_index"] for item in windows] == [index - 30 for index in anchors]
    assert all(item["episode"] == 1 for item in windows)
    with pytest.raises(ValueError, match="too short"):
        build_windows(rows, [1], mapping, n_obs_steps=n_obs_steps, windows_per_episode=13)
    with pytest.raises(ValueError, match="Missing frame"):
        build_windows(rows, [1], {}, n_obs_steps=n_obs_steps)


def test_action_errors_use_physical_units_and_rotation_geometry() -> None:
    target = torch.zeros(1, 2, 7, dtype=torch.float64)
    prediction = target.clone()
    prediction[0, 0, 0] = 0.12
    prediction[0, 0, 5] = math.pi / 2
    prediction[0, 0, 6] = 0.25
    # 不同轴角可代表相同旋转，不能直接计算 rotvec 的欧氏差。
    prediction[0, 1, 3] = 2 * math.pi
    errors = action_errors(target, prediction)
    assert errors["position_m"][0].tolist() == pytest.approx([0.12, 0.0])
    assert errors["rotation_rad"][0, 0].item() == pytest.approx(math.pi / 2)
    assert errors["rotation_rad"][0, 1].item() == pytest.approx(0.0, abs=1e-7)
    assert errors["gripper"][0, 0].item() == pytest.approx(0.25)
    with pytest.raises(ValueError, match="Non-finite"):
        action_errors(target, prediction * float("nan"))


class _FakePolicy:
    def __init__(self, n_obs_steps: int = 2) -> None:
        self.config = SimpleNamespace(n_obs_steps=n_obs_steps, n_action_steps=8)

    def __call__(self, batch):
        return batch["action"].mean(), None

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        # 消耗随机数以核验 seeded_context 能保留调用方随机流。
        action = torch.zeros(len(batch["action"]), 8, 7) + torch.rand(()) * 0
        # 为每个锚点生成不同的夹爪值，核对批次内的预测与索引配对。
        action[:, :, 6] = batch["index"][:, None] / 1000
        return action


def _items(n_obs_steps: int = 2) -> list[dict]:
    result = []
    for index in range(8):
        action = torch.zeros(16, 7)
        action[:, 0] = torch.arange(16)
        result.append(
            {
                "episode_index": torch.tensor(4),
                "index": torch.tensor(100 + index),
                "observation.state_is_pad": torch.zeros(n_obs_steps, dtype=torch.bool),
                "action_is_pad": torch.zeros(16, dtype=torch.bool),
                "action": action,
            }
        )
    return result


@pytest.mark.parametrize("n_obs_steps", [2, 3])
def test_action_execution_window_is_copied_before_preprocessing(n_obs_steps: int) -> None:
    items = _items(n_obs_steps)
    policy = _FakePolicy(n_obs_steps)
    windows = [{"episode": 4, "anchor": 100 + i, "relative_index": i} for i in range(8)]

    def mutate_action(batch):
        batch["action"].fill_(999)
        return batch

    def postprocess(action: torch.Tensor) -> torch.Tensor:
        """模拟反归一化，验证导出使用后处理结果。"""
        action[:, :, 6] += 0.25
        return action

    torch.manual_seed(12)
    before = torch.random.get_rng_state().clone()
    result, predictions = evaluate_actions(
        policy, DataLoader(items, batch_size=3), windows, [], {}, mutate_action, postprocess
    )
    assert torch.equal(torch.random.get_rng_state(), before)
    # 两帧模型执行索引 1..8，三帧模型执行索引 2..9。
    first, last = n_obs_steps - 1, n_obs_steps + 6
    assert result["overall"]["position_m"]["mean"] == pytest.approx((first + last) / 2)
    assert result["first_step"]["position_m"]["mean"] == pytest.approx(first)
    assert result["per_step"]["7"]["position_m"]["mean"] == pytest.approx(last)
    assert result["per_episode"]["4"]["position_m"]["mean"] == pytest.approx((first + last) / 2)
    assert len(result["per_seed"]) == 3
    assert [(row["seed"], row["episode"], row["anchor"]) for row in predictions] == [
        (seed, 4, anchor) for seed in (0, 1, 2) for anchor in range(100, 108)
    ]
    for row in predictions:
        assert np.asarray(row["action"]).shape == (8, 7)
        assert [step[6] for step in row["action"]] == pytest.approx([row["anchor"] / 1000 + 0.25] * 8)

    items[0]["action_is_pad"][last] = True
    with pytest.raises(ValueError, match="padded actions"):
        evaluate_actions(policy, DataLoader(items, batch_size=3), windows, [], {}, lambda x: x, lambda x: x)


def test_loss_covers_all_frames_and_weights_short_batch() -> None:
    items = _items()[:3]
    for item, value in zip(items, (1.0, 2.0, 6.0), strict=True):
        item["action"] = torch.full((16, 7), value)
    result = evaluate_losses(_FakePolicy(), DataLoader(items, batch_size=2), [], {}, lambda x: x)
    assert result["mean"] == pytest.approx(3.0)
    assert result["samples_per_seed"] == 3
    assert result["per_seed"] == {"0": 3.0, "1": 3.0, "2": 3.0}


def test_all_seed_samples_contribute_to_summary() -> None:
    windows = [{"episode": 4, "anchor": 100 + i, "relative_index": i} for i in range(8)]
    values = np.stack([np.full((8, 8), seed) for seed in (0, 1, 2)])
    result = summarize_actions({"position_m": values}, windows)
    assert result["overall"]["position_m"]["mean"] == 1.0
    assert result["per_seed"]["2"]["position_m"]["mean"] == 2.0
