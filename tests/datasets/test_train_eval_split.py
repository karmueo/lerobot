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

import random
from types import SimpleNamespace

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.datasets import factory
from lerobot.datasets.utils import resolve_episode_indices

pytestmark = pytest.mark.dataset


def test_tail_split_preserves_previous_task_group_order() -> None:
    # 两个任务交错出现时，保留旧实现按首次出现顺序分组的结果。
    tasks = [["a"], ["b"], ["a"], ["b"], ["a"], ["b"]]
    train, evaluation = factory.split_episodes_by_task(list(range(6)), tasks, 0.34)
    assert train == [0, 1]
    assert evaluation == [2, 4, 3, 5]


def test_random_split_is_seeded_stratified_disjoint_and_does_not_consume_global_rng() -> None:
    tasks = [["a"] if episode < 100 else ["b"] for episode in range(200)]
    episodes = list(range(200))
    random.seed(51)
    before = random.getstate()
    train, evaluation = factory.split_episodes_by_task(episodes, tasks, 0.1, "random", 1000)
    assert random.getstate() == before
    assert (train, evaluation) == factory.split_episodes_by_task(episodes, tasks, 0.1, "random", 1000)
    assert evaluation != factory.split_episodes_by_task(episodes, tasks, 0.1, "random", 1001)[1]
    assert len(train) == 180 and len(evaluation) == 20
    assert train == sorted(train) and evaluation == sorted(evaluation)
    assert set(train).isdisjoint(evaluation)
    assert set(train) | set(evaluation) == set(episodes)
    assert sum(episode < 100 for episode in evaluation) == 10


def test_split_rejects_no_training_episode() -> None:
    with pytest.raises(ValueError, match="0 training episodes"):
        factory.split_episodes_by_task([0, 1], [["a"], ["b"]], 0.1, "random", 1000)


def test_factory_splits_only_filtered_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [["task"] for _ in range(10)]
    metadata = SimpleNamespace(episodes={"tasks": tasks}, total_episodes=10)
    selected = resolve_episode_indices([0, 1, 2, 4, 5, 7, 9], 10, [1, 7])
    full_dataset = SimpleNamespace(episodes=selected, meta=metadata)
    created: list[SimpleNamespace] = []

    def create_dataset(*args: object, **kwargs: object) -> SimpleNamespace:
        dataset = SimpleNamespace(
            episodes=kwargs["episodes"], meta=metadata, transforms=kwargs["image_transforms"]
        )
        created.append(dataset)
        return dataset

    monkeypatch.setattr(factory, "make_dataset", lambda cfg: full_dataset)
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args: None)
    monkeypatch.setattr(factory, "LeRobotDataset", create_dataset)
    config = DatasetConfig(
        repo_id="owner/data",
        episodes=[0, 1, 2, 4, 5, 7, 9],
        exclude_episodes=[1, 7],
        eval_split=0.2,
        eval_split_strategy="random",
        use_imagenet_stats=False,
    )
    cfg = SimpleNamespace(dataset=config, trainable_config=None, rename_map={}, tolerance_s=1e-4)
    train, evaluation = factory.make_train_eval_datasets(cfg)

    assert evaluation is not None
    assert len(train.episodes) == 4 and len(evaluation.episodes) == 1
    assert set(train.episodes) | set(evaluation.episodes) == {0, 2, 4, 5, 9}
    assert set(train.episodes).isdisjoint(evaluation.episodes)
    assert created[1].transforms is None
    manifest = factory.build_split_manifest(cfg, train, evaluation)
    assert manifest == {
        "version": 1,
        "dataset": {"repo_id": "owner/data", "root": None},
        "split_strategy": "random",
        "split_seed": 1000,
        "train_episodes": train.episodes,
        "eval_episodes": evaluation.episodes,
    }


def test_manifest_expands_unrestricted_dataset() -> None:
    dataset = SimpleNamespace(episodes=None, meta=SimpleNamespace(total_episodes=3))
    cfg = SimpleNamespace(dataset=DatasetConfig(repo_id="owner/data"))
    manifest = factory.build_split_manifest(cfg, dataset, None)
    assert manifest["train_episodes"] == [0, 1, 2]
    assert manifest["eval_episodes"] == []
