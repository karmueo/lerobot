"""原生 Diffusion 动作段推理、进程隔离和本地验证集回放."""

from __future__ import annotations

import logging
import multiprocessing as mp
import resource
import signal
import time
from collections.abc import Callable, Iterator
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from queue import Empty
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import (
    IMAGE_KEY,
    IMAGE_SHAPE,
    REPO_ROOT,
    STATE_KEY,
    DeploymentAssets,
    DeploymentConfig,
    PolicyConfig,
    _read_json,
)
from .geometry import pose_to_transform
from .types import ActionChunk, ObservationWindow

if TYPE_CHECKING:
    from torch import Tensor

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

logger = logging.getLogger(__name__)


class PolicyRunner:
    """独占本地模型与处理器，接收完整三帧窗口并返回八步物理目标."""

    def __init__(self, cfg: PolicyConfig) -> None:
        """严格加载本地 EMA 权重及保存的前后处理器."""
        # 推理依赖只在模型进程和离线测速中加载，控制主进程无需初始化 CUDA。
        import torch

        from lerobot.export.package import load_config, load_processors
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        self._torch = torch
        self._device = torch.device(cfg.device)
        if self._device.type == "cuda" and (
            not torch.cuda.is_available()
            or (self._device.index is not None and self._device.index >= torch.cuda.device_count())
        ):
            raise RuntimeError(f"推理设备不可用: {cfg.device}")
        if not cfg.checkpoint.is_dir():
            raise FileNotFoundError(f"找不到本地检查点: {cfg.checkpoint}")
        model_cfg = load_config(cfg.checkpoint, cfg.device)
        if not isinstance(model_cfg, DiffusionConfig) or (
            model_cfg.n_obs_steps,
            model_cfg.n_action_steps,
            model_cfg.horizon,
            model_cfg.use_relative_pose,
        ) != (3, 8, 16, False):
            raise ValueError("检查点与 RM75B 三帧、八步及绝对回合坐标契约不一致")
        model_cfg.use_amp = False
        model_cfg.dtype = torch.float32
        if cfg.num_inference_steps is not None:
            if type(cfg.num_inference_steps) is not int or not 1 <= cfg.num_inference_steps <= 100:
                raise ValueError("推理步数必须为 1～100 的整数")
            model_cfg.num_inference_steps = cfg.num_inference_steps
        self._policy: DiffusionPolicy = (
            DiffusionPolicy.from_pretrained(
                cfg.checkpoint, config=model_cfg, local_files_only=True, strict=True
            )
            .eval()
            .float()
        )
        self._pre, self._post = load_processors(cfg.checkpoint, self._policy.config, cfg.device)
        self._seed = cfg.seed

    @property
    def num_inference_steps(self) -> int:
        """给出 scheduler 实际使用的去噪步数."""
        return int(self._policy.diffusion.num_inference_steps)

    def _synchronize(self) -> None:
        """CUDA 计时包含真实计算时间."""
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)

    def predict(self, window: ObservationWindow, *, seed: int | None = None) -> ActionChunk:
        """完整处理一次窗口，模型已截取八步动作，部署层只反归一化."""
        from lerobot.utils.random_utils import seeded_context

        started = time.perf_counter()
        images, states = window.images, window.states
        if (
            images.shape != (3, *IMAGE_SHAPE)
            or states.shape != (3, 7)
            or images.dtype != np.float32
            or states.dtype != np.float32
            or not np.isfinite(images).all()
            or not np.isfinite(states).all()
            or np.any((images < 0) | (images > 1))
            or np.any((states[:, -1] < 0) | (states[:, -1] > 1))
        ):
            raise ValueError("模型输入须为三帧有限 float32 RGB [0,1] 和七维状态")
        torch = self._torch
        effective_seed = (self._seed + window.request_id) % 2**32 if seed is None else seed
        with torch.inference_mode(), seeded_context(effective_seed):
            self._policy.reset()
            raw = {
                IMAGE_KEY: torch.from_numpy(images).unsqueeze(0),
                STATE_KEY: torch.from_numpy(states).unsqueeze(0),
            }
            batch = self._pre(raw)
            self._synchronize()
            prepared = time.perf_counter()
            # 原生接口从当前帧开始返回八步，且 DDPM 的各步噪声受同一 RNG 控制。
            predicted = self._policy.predict_action_chunk(batch)
            self._synchronize()
            sampled = time.perf_counter()
            actions = self._post(predicted).detach().to(device="cpu", dtype=torch.float32).numpy()
            self._synchronize()
        finished = time.perf_counter()
        if actions.shape != (1, 8, 7) or not np.isfinite(actions).all():
            raise ValueError(f"原生动作须为有限 [1,8,7] 张量，实际形状 {actions.shape}")
        return ActionChunk(
            episode_id=window.episode_id,
            request_id=window.request_id,
            observation_timestamp=window.timestamps[-1],
            actions=actions[0].copy(),
            timings={
                "preprocess_s": prepared - started,
                "inference_s": sampled - prepared,
                "postprocess_s": finished - sampled,
                "total_s": finished - started,
            },
        )

    def warmup(self) -> None:
        """在进入 READY 前执行一次完整的模型和处理器调用."""
        origin = np.eye(4, dtype=np.float64)
        window = ObservationWindow(
            episode_id=0,
            request_id=0,
            frame_ids=(0, 1, 2),
            timestamps=(0.0, 1 / 30, 2 / 30),
            images=np.zeros((3, *IMAGE_SHAPE), dtype=np.float32),
            states=np.zeros((3, 7), dtype=np.float32),
            anchor=origin,
            capture_tcp=origin,
        )
        self.predict(window)


def _worker_main(cfg: PolicyConfig, incoming: Queue, outgoing: Queue) -> None:
    """Spawn 子进程的模型生命周期，错误显式返回控制进程."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        runner = PolicyRunner(cfg)
        runner.warmup()
        outgoing.put(("ready", None))
        while (window := incoming.get()) is not None:
            outgoing.put(("action", runner.predict(window)))
    except Exception as error:
        outgoing.put(("error", f"{type(error).__name__}: {error}"))


class InferenceWorker:
    """单请求模型子进程，停止或换回合时丢弃在途结果."""

    def __init__(
        self, cfg: PolicyConfig, *, save_result: Callable[[ActionChunk, bool], None] | None = None
    ) -> None:
        """保存配置及主进程使用的可选结果存档回调."""
        self._cfg = cfg
        self._save_result = save_result
        self._process: SpawnProcess | None = None
        self._incoming: Queue | None = None
        self._outgoing: Queue | None = None
        self._pending: tuple[int, int, float] | None = None
        self._discard = False

    @property
    def busy(self) -> bool:
        """在途结果即使已作废，也占用槽位直至被 poll 排空."""
        return self._pending is not None

    def start(self) -> None:
        """启动并等待模型预热，在机器人执行前检查模型进程."""
        if self._process is not None:
            raise RuntimeError("推理子进程已启动")
        context = mp.get_context("spawn")
        self._incoming = context.Queue(maxsize=1)
        self._outgoing = context.Queue(maxsize=1)
        self._process = context.Process(
            target=_worker_main,
            args=(self._cfg, self._incoming, self._outgoing),
            name="rm75b-policy",
            daemon=True,
        )
        try:
            self._process.start()
            deadline = time.monotonic() + self._cfg.startup_timeout_s
            while (remaining := deadline - time.monotonic()) > 0:
                try:
                    kind, payload = self._outgoing.get(timeout=min(remaining, 0.1))
                except Empty:
                    self._check_alive()
                    continue
                if kind == "ready":
                    return
                raise RuntimeError(f"推理子进程启动失败: {payload}")
            raise TimeoutError("推理子进程加载或预热超时")
        except BaseException:
            self.close()
            raise

    def _check_alive(self) -> None:
        """空队列时也检测进程崩溃."""
        process = self._process
        if process is None or not process.is_alive():
            exitcode = None if process is None else process.exitcode
            raise RuntimeError(f"推理子进程未运行，退出码 {exitcode}")

    def submit(self, window: ObservationWindow) -> None:
        """发送一个观测请求，不在有请求占用时隐式覆盖."""
        self._check_alive()
        if self._incoming is None or self.busy:
            raise RuntimeError("推理请求槽位尚未清空")
        self._incoming.put_nowait(window)
        self._pending = (window.episode_id, window.request_id, window.timestamps[-1])
        self._discard = False

    def poll(self) -> ActionChunk | None:
        """读取就绪结果，作废结果仅用于释放槽位."""
        if self._outgoing is None:
            raise RuntimeError("推理子进程尚未启动")
        try:
            kind, payload = self._outgoing.get_nowait()
        except Empty:
            self._check_alive()
            return None
        pending, self._pending = self._pending, None
        discard, self._discard = self._discard, False
        if kind == "error":
            raise RuntimeError(f"推理子进程失败: {payload}")
        if (
            kind != "action"
            or not isinstance(payload, ActionChunk)
            or (payload.episode_id, payload.request_id, payload.observation_timestamp) != pending
        ):
            raise RuntimeError("推理子进程返回的请求标识不匹配")
        # 即使本回合已经停止，也保存已取回且编号匹配的模型输出。
        if self._save_result is not None:
            self._save_result(payload, discard)
        return None if discard else payload

    def invalidate(self) -> None:
        """立刻使当前请求失去执行资格，不阻塞模型计算."""
        self._discard = self.busy

    def close(self) -> None:
        """限时关闭子进程及 IPC，允许在部分启动失败后调用."""
        self.invalidate()
        process, self._process = self._process, None
        try:
            if process is not None:
                if process.pid is not None:
                    if process.is_alive() and not self.busy and self._incoming is not None:
                        self._incoming.put_nowait(None)
                        process.join(timeout=1.0)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1.0)
                process.close()
        finally:
            # 退出期间可丢弃尚未送出的相机窗口，避免等待已终止子进程读取队列。
            for channel in (self._incoming, self._outgoing):
                if channel is not None:
                    channel.cancel_join_thread()
                    channel.close()
            self._incoming = self._outgoing = None
            self._pending = None
            self._discard = False


def replay_windows(cfg: DeploymentConfig, count: int) -> Iterator[ObservationWindow]:
    """使用本地验证划分读取互异的三帧窗口，不触发数据集下载."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_reader import DatasetReader
    from lerobot.utils.import_utils import require_package

    require_package("datasets", extra="dataset")
    require_package("av", extra="dataset")
    root = cfg.policy.dataset_root
    manifest = _read_json(cfg.policy.checkpoint / "split_manifest.json")
    dataset = manifest["dataset"]
    if (
        manifest.get("version") != 1
        or (REPO_ROOT / dataset["root"]).resolve() != root.resolve()
        or not isinstance(dataset.get("repo_id"), str)
    ):
        raise ValueError("验证划分清单与部署数据路径不一致")
    train, evaluation = manifest["train_episodes"], manifest["eval_episodes"]
    if (
        not evaluation
        or any(type(episode) is not int or episode < 0 for episode in [*train, *evaluation])
        or len(evaluation) != len(set(evaluation))
        or set(evaluation) & set(train)
    ):
        raise ValueError("验证划分包含重复、无效或训练回合")
    required = ("info.json", "tasks.parquet", "stats.json")
    if any(not (root / "meta" / name).is_file() for name in required) or not any(
        (root / "meta/episodes").glob("*/*.parquet")
    ):
        raise FileNotFoundError(f"本地验证集元信息不完整: {root}")
    meta = LeRobotDatasetMetadata(dataset["repo_id"], root=root)
    episodes = meta.episodes
    if episodes is None or max(evaluation) >= len(episodes):
        raise ValueError("验证回合不在本地数据中")
    # DatasetReader 直接加载本地文件，缺失时返回失败，不使用公共数据集的下载回退。
    reader = DatasetReader(
        meta=meta,
        root=root,
        episodes=evaluation,
        tolerance_s=1e-3,
        video_backend="pyav",
        delta_timestamps={key: [-2 / 30, -1 / 30, 0.0] for key in (IMAGE_KEY, STATE_KEY)},
        image_transforms=None,
    )
    if not reader.try_load():
        raise FileNotFoundError(f"本地验证集数据或视频不完整: {root}")
    mapping = reader.absolute_to_relative_idx
    if mapping is None:
        raise RuntimeError("本地验证集缺少绝对帧索引")
    episode_rows = [(episode, episodes[episode]) for episode in sorted(evaluation)]
    ranges = [
        (episode, int(row["dataset_from_index"]) + 2, int(row["dataset_to_index"]) - 8)
        for episode, row in episode_rows
    ]
    # 第一轮取各回合中点，后续轮次二分尚未采样的间隔。
    windows: list[tuple[int, int]] = []
    while ranges and len(windows) < count:
        next_ranges: list[tuple[int, int, int]] = []
        for episode, low, high in ranges:
            if low > high:
                continue
            frame = (low + high) // 2
            windows.append((episode, frame))
            next_ranges.extend(((episode, low, frame - 1), (episode, frame + 1, high)))
            if len(windows) == count:
                break
        ranges = next_ranges
    if len(windows) != count:
        raise ValueError(f"验证集仅提供 {len(windows)} 个完整窗口，要求 {count} 个")
    for request_id, (episode, frame) in enumerate(windows, start=1):
        batch = reader.get_item(mapping[frame])
        if int(batch["episode_index"]) != episode or int(batch["index"]) != frame:
            raise ValueError("验证窗口与绝对帧索引不一致")
        for key in (IMAGE_KEY, STATE_KEY):
            padded = batch.get(f"{key}_is_pad")
            if padded is not None and padded.any():
                raise ValueError("验证窗口包含补帧")
        images = batch[IMAGE_KEY].numpy().astype(np.float32, copy=False)
        states = batch[STATE_KEY].numpy().astype(np.float32, copy=False)
        timestamp = float(batch["timestamp"])
        yield ObservationWindow(
            episode_id=episode,
            request_id=request_id,
            frame_ids=(frame - 2, frame - 1, frame),
            timestamps=(timestamp - 2 / 30, timestamp - 1 / 30, timestamp),
            images=images,
            states=states,
            anchor=np.eye(4, dtype=np.float64),
            capture_tcp=pose_to_transform(states[-1, :6]),
        )


def _native_parity(
    runner: PolicyRunner, cfg: PolicyConfig, windows: list[ObservationWindow]
) -> dict[str, Any]:
    """以独立加载的处理器和原生 API 对照部署调用，复位完整 RNG."""
    import torch

    from lerobot.export.package import load_processors
    from lerobot.utils.random_utils import seeded_context

    native = runner._policy
    pre, post = load_processors(cfg.checkpoint, native.config, cfg.device)
    maximum_error = 0.0
    seeds = (0, 1, 2)
    logger.info("开始原生一致性验证，共 %d 个窗口、每窗口 %d 个种子", len(windows), len(seeds))
    for index, window in enumerate(windows, start=1):
        started = time.perf_counter()
        for seed in seeds:
            actual = runner.predict(window, seed=seed).actions
            with torch.inference_mode(), seeded_context(seed):
                native.reset()
                native_input: dict[str, Tensor] = {
                    IMAGE_KEY: torch.tensor(window.images[None], dtype=torch.float32),
                    STATE_KEY: torch.tensor(window.states[None], dtype=torch.float32),
                }
                expected = post(native.predict_action_chunk(pre(native_input))).cpu().numpy()[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        logger.info(
            "一致性验证 %d/%d，回合 %d、帧 %d，通过，用时 %.3f 秒",
            index,
            len(windows),
            window.episode_id,
            window.frame_ids[-1],
            time.perf_counter() - started,
        )
    return {
        "windows": len(windows),
        "seeds": list(seeds),
        "comparisons": len(windows) * len(seeds),
        "rtol": 1e-5,
        "atol": 1e-6,
        "maximum_absolute_error": maximum_error,
    }


def benchmark(
    cfg: DeploymentConfig,
    assets: DeploymentAssets,
    *,
    warmup: int = 2,
    samples: int = 20,
    verify_native: bool = False,
) -> dict[str, Any]:
    """测量真实验证窗口的读取和推理时延及 CPU/CUDA 内存."""
    import torch

    if type(warmup) is not int or type(samples) is not int or warmup < 0 or samples < 1:
        raise ValueError("预热次数须为非负整数，测速次数须为正整数")
    logger.info("加载本地模型并准备 %d 次预热和 %d 次测速，设备 %s", warmup, samples, cfg.policy.device)
    runner = PolicyRunner(cfg.policy)
    count = max(warmup + samples, 10 if verify_native else 0)
    stream = replay_windows(cfg, count)
    records: list[dict[str, float]] = []
    parity_windows: list[ObservationWindow] = []
    device = torch.device(cfg.policy.device)
    for index in range(count):
        if index == warmup and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        window = next(stream)
        read_s = time.perf_counter() - started
        if index < 10 and verify_native:
            parity_windows.append(window)
        if index < warmup + samples:
            chunk = runner.predict(window)
            if index >= warmup:
                records.append({"replay_read_s": read_s, **chunk.timings})
            phase, completed, total = (
                ("预热", index + 1, warmup) if index < warmup else ("测速", index + 1 - warmup, samples)
            )
            logger.info(
                "%s %d/%d，回合 %d、帧 %d，窗口读取 %.3f 秒，完整推理 %.3f 秒",
                phase,
                completed,
                total,
                window.episode_id,
                window.frame_ids[-1],
                read_s,
                chunk.timings["total_s"],
            )
    timing = {
        name: {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
        for name in records[0]
        for values in [[sample[name] for sample in records]]
    }
    memory = {"host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}
    if device.type == "cuda":
        memory.update(
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    result: dict[str, Any] = {
        "device": cfg.policy.device,
        "scheduler": "DDPM",
        "num_inference_steps": runner.num_inference_steps,
        "dtype": "float32",
        "warmup": warmup,
        "samples": samples,
        "timing_s": timing,
        "memory_bytes": memory,
        "asset_hashes": assets.hashes,
    }
    if verify_native:
        result["native_parity"] = _native_parity(runner, cfg.policy, parity_windows)
    return result
