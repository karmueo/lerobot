#!/usr/bin/env python

"""离线验证的随机状态、EMA 权重和续训口径。"""

import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# 训练入口在模块加载时导入数据集工厂，故须在收集阶段先检查可选依赖。
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.common.eval_protocol import EVAL_PROTOCOL, build_eval_protocol, validate_eval_protocol
from lerobot.optim.ema import EMA, WarmupEMAConfig
from lerobot.scripts.lerobot_train import compute_offline_eval_loss
from lerobot.utils import random_utils
from lerobot.utils.io_utils import write_json
from lerobot.utils.random_utils import get_rng_state, set_rng_state

pytestmark = pytest.mark.dataset


@pytest.fixture(autouse=True)
def cpu_only_random_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认不访问真实 GPU；跨设备状态测试使用单独的模拟对象。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: None)


class FakeAccelerator:
    device = torch.device("cpu")

    def autocast(self):
        return nullcontext()

    def reduce(self, tensor, reduction):
        return tensor

    def unwrap_model(self, policy):
        return policy


class RandomLossPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.child = torch.nn.Identity()
        self.fail = False

    def forward(self, batch):
        if self.fail:
            raise RuntimeError("test failure")
        loss = ((self.weight * batch - torch.randn_like(batch)) ** 2).mean()
        return loss, {}


@pytest.mark.parametrize("failure", [False, True])
def test_seeded_context_restores_all_cuda_devices_and_mps_without_hardware(monkeypatch, failure):
    # 以模拟设备状态验证跨卡保存，避免该 CPU 单测占用训练 GPU。
    states = {"cuda": [torch.tensor([1]), torch.tensor([2])], "mps": torch.tensor([3])}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.random, "get_rng_state_all", lambda: [s.clone() for s in states["cuda"]])
    monkeypatch.setattr(torch.cuda.random, "set_rng_state_all", lambda s: states.update(cuda=s))
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: states["mps"].clone())
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda s: states.update(mps=s))

    def fake_seed(seed):
        states.update(cuda=[torch.tensor([seed]), torch.tensor([seed])], mps=torch.tensor([seed]))

    monkeypatch.setattr(random_utils, "set_seed", fake_seed)
    try:
        with random_utils.seeded_context(123):
            assert [s.item() for s in states["cuda"]] == [123, 123]
            if failure:
                raise RuntimeError("test failure")
    except RuntimeError:
        if not failure:
            raise
    assert [s.item() for s in states["cuda"]] == [1, 2]
    assert states["mps"].item() == 3


def test_fixed_eval_repeats_without_consuming_training_rng():
    policy = RandomLossPolicy()
    accelerator = FakeAccelerator()
    batches = [torch.ones(4), torch.ones(3)]
    state = get_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(2))
    set_rng_state(state)

    first, count = compute_offline_eval_loss(policy, batches, lambda batch: batch, accelerator, seed=123)
    second, _ = compute_offline_eval_loss(policy, batches, lambda batch: batch, accelerator, seed=123)

    assert first == second
    assert count == 2
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("failure", [False, True])
def test_ema_eval_restores_parameters_and_each_module_mode(failure):
    policy = RandomLossPolicy()
    policy.train()
    policy.child.eval()
    ema = EMA(policy, WarmupEMAConfig())
    ema.averages["weight"].fill_(3.0)
    policy.fail = failure
    state = get_rng_state()
    expected = torch.rand(3)
    set_rng_state(state)
    if failure:
        with pytest.raises(RuntimeError, match="test failure"):
            compute_offline_eval_loss(
                policy, [torch.ones(4)], lambda batch: batch, FakeAccelerator(), seed=4, ema=ema
            )
    else:
        ema_loss, _ = compute_offline_eval_loss(
            policy, [torch.ones(4)], lambda batch: batch, FakeAccelerator(), seed=4, ema=ema
        )
        live_loss, _ = compute_offline_eval_loss(
            policy, [torch.ones(4)], lambda batch: batch, FakeAccelerator(), seed=4
        )
        assert ema_loss != live_loss
    assert policy.weight.item() == 1.0
    assert policy.training
    assert not policy.child.training
    torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)


def test_eval_protocol_rejects_mismatch_and_new_modes_for_legacy(tmp_path):
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(revision=None),
        trainable_config=SimpleNamespace(drop_n_last_frames=7),
        eval_seed=1000,
        eval_use_ema=True,
        batch_size=128,
        max_eval_samples=0,
        rename_map={},
    )
    manifest = {
        "dataset": {"repo_id": "handumi/test"},
        "train_episodes": [0, 1],
        "eval_episodes": [2],
        "split_strategy": "random",
        "split_seed": 1000,
    }
    protocol = build_eval_protocol(cfg, manifest, 2)
    with pytest.raises(ValueError, match="no evaluation protocol"):
        validate_eval_protocol(tmp_path, protocol, allow_legacy=False)
    validate_eval_protocol(tmp_path, protocol, allow_legacy=True)
    write_json(protocol, tmp_path / EVAL_PROTOCOL)
    validate_eval_protocol(tmp_path, protocol, allow_legacy=False)
    for field, value in [("eval_seed", 4), ("weight_type", "live"), ("dp_world_size", 1)]:
        changed = {**protocol, field: value}
        with pytest.raises(ValueError, match=field):
            validate_eval_protocol(tmp_path, changed, allow_legacy=False)
