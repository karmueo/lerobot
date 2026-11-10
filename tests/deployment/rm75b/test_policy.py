"""验证动作段和 IPC 隔离，完整 GPU 模型测试需显式启用."""

import os
import time
from multiprocessing import active_children
from multiprocessing.queues import Queue
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.deployment.rm75b import policy as policy_module
from lerobot.deployment.rm75b.config import PolicyConfig, load_assets, load_config
from lerobot.deployment.rm75b.policy import InferenceWorker, PolicyRunner, benchmark, replay_windows
from lerobot.deployment.rm75b.types import ActionChunk, ObservationWindow


def _window(request: int) -> ObservationWindow:
    """创建固定的三帧窗口，以请求号区分迟到结果."""
    return ObservationWindow(
        4,
        request,
        (0, 1, 2),
        (0.0, 1 / 30, 2 / 30),
        np.zeros((3, 3, 134, 240), dtype=np.float32),
        np.zeros((3, 7), dtype=np.float32),
        np.eye(4),
        np.eye(4),
    )


class _NativePolicy:
    """返回带明确路点编号的原生八步动作，用于发现重复切片."""

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assert batch["observation.images.front"].shape == (1, 3, 3, 134, 240)
        assert batch["observation.state"].shape == (1, 3, 7)
        return torch.arange(56, dtype=torch.float32).reshape(1, 8, 7)


def test_native_chunk_is_not_sliced_again() -> None:
    """执行完整的在线调用边界，保留后处理的全部八步."""
    runner = PolicyRunner.__new__(PolicyRunner)
    runner._torch = torch
    runner._device = torch.device("cpu")
    runner._policy = _NativePolicy()
    runner._pre = lambda batch: batch
    runner._post = lambda actions: actions + 100
    runner._seed = 0
    chunk = runner.predict(_window(1))
    np.testing.assert_array_equal(chunk.actions, np.arange(56).reshape(8, 7) + 100)
    assert chunk.request_id == 1


def _fake_worker(cfg: PolicyConfig, incoming: Queue, outgoing: Queue) -> None:
    """由真实 spawn 启动的轻量推理子进程替身."""
    outgoing.put(("ready", None))
    while (window := incoming.get()) is not None:
        assert window.archive is None
        if window.request_id == 3:
            outgoing.put(("error", "mock worker failure"))
            return
        outgoing.put(
            (
                "action",
                ActionChunk(
                    window.episode_id,
                    window.request_id,
                    window.timestamps[-1],
                    np.zeros((8, 7), dtype=np.float32),
                ),
            )
        )


def _drain(worker: InferenceWorker, timeout_s: float = 5.0) -> ActionChunk | None:
    """限时轮询单请求，测试不以 sleep 猜测进程结束时间."""
    deadline = time.monotonic() + timeout_s
    while worker.busy:
        result = worker.poll()
        if not worker.busy:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError("等待测试子进程结果超时")
        time.sleep(0.01)
    return None


def test_spawn_worker_discards_invalidated_and_surfaces_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """确认迟到动作无法驱动新任务，在途槽位和错误能正确传递."""
    monkeypatch.setattr(policy_module, "_worker_main", _fake_worker)
    saved: list[tuple[int, bool]] = []
    worker = InferenceWorker(
        PolicyConfig(device="cpu", startup_timeout_s=20),
        save_result=lambda action, discarded: saved.append((action.request_id, discarded)),
    )
    worker.start()
    try:
        worker.submit(_window(1))
        worker.invalidate()
        assert worker.busy
        with pytest.raises(RuntimeError, match="槽位"):
            worker.submit(_window(2))
        assert _drain(worker) is None
        assert not worker.busy
        worker.submit(_window(2))
        result = _drain(worker)
        assert result is not None and result.request_id == 2
        worker.submit(_window(3))
        with pytest.raises(RuntimeError, match="mock worker failure"):
            _drain(worker)
        assert saved == [(1, True), (2, False)]
    finally:
        worker.close()


@pytest.mark.dataset
@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("LEROBOT_RM75B_RUN_NATIVE") != "1", reason="需显式启用真实检查点验证")
def test_real_checkpoint_native_parity() -> None:
    """使用本机十个验证窗口、三个 RNG 种子与原生 API 对照."""
    root = Path(__file__).resolve().parents[3]
    cfg = load_config(root / "docs/umi/rm75b_deployment.yaml")
    result = benchmark(cfg, load_assets(cfg), warmup=0, samples=1, verify_native=True)
    assert result["native_parity"]["comparisons"] == 30


@pytest.mark.dataset
@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("LEROBOT_RM75B_RUN_WORKER") != "1", reason="需显式启用真实推理子进程验证")
def test_real_checkpoint_worker_lifecycle() -> None:
    """以真实 CUDA 模型验证预热、作废排空、下一请求和限时退出."""
    root = Path(__file__).resolve().parents[3]
    cfg = load_config(root / "docs/umi/rm75b_deployment.yaml")
    first, second = replay_windows(cfg, 2)
    worker = InferenceWorker(cfg.policy)
    worker.start()
    try:
        # start 返回时子进程已加载 EMA 权重并完成一次完整预热。
        process = worker._process
        assert process is not None and process.is_alive()
        pid = process.pid
        assert pid is not None

        worker.submit(first)
        worker.invalidate()
        assert worker.busy
        assert _drain(worker, cfg.execution.inference_timeout_s) is None
        assert not worker.busy

        worker.submit(second)
        result = _drain(worker, cfg.execution.inference_timeout_s)
        assert result is not None
        assert (result.episode_id, result.request_id) == (second.episode_id, second.request_id)
        assert result.observation_timestamp == second.timestamps[-1]
        assert result.actions.shape == (8, 7)
        assert result.actions.dtype == np.float32
        assert np.isfinite(result.actions).all()
    finally:
        started = time.monotonic()
        worker.close()
        closed_in = time.monotonic() - started
    assert closed_in < 6.0
    assert not worker.busy
    assert all(child.pid != pid for child in active_children())
