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
"""真实观测上的确定性采样比对和完整动作块性能测量."""

import gc
import importlib.metadata
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.export_diffusion import make_export_modules
from lerobot.processor.relative_pose_processor import axis_angle_to_matrix

from .backends import OnnxSession, TensorRTSession, TorchSession
from .package import file_hash, load_manifest, load_policy, load_processors, read_json, write_json
from .runtime import ExportedDiffusion, encoder_inputs, synchronize

DEFAULT_CHECKPOINT = Path("outputs/train/dp_umi_v4/checkpoints/012000/pretrained_model")
DEFAULT_EXPORT = Path("outputs/export/dp_umi_v4/012000")


@dataclass
class ValidationConfig:
    """默认验收当前 UMI 检查点；显式参数也支持独立重新验证."""

    export_dir: Path = DEFAULT_EXPORT
    policy_path: Path = DEFAULT_CHECKPOINT
    dataset_root: Path | None = None
    device: str = "cuda:0"
    backends: list[str] = field(default_factory=lambda: ["onnx", "tensorrt_fp32", "tensorrt_fp16"])
    benchmark: bool = True
    episodes: int = 4
    windows_per_episode: int = 8
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    warmup: int = 10
    iterations: int = 100
    # 未知数据集必须显式提供单位；不能从维度猜测物理含义。
    position_unit: str | None = None
    rotation_unit: str | None = None
    gripper_representation: str | None = None
    position_tolerance_mm: float = 1.0
    rotation_tolerance_deg: float = 0.5
    gripper_tolerance: float = 0.01


def error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    """网络及归一化动作的数值误差，不把余弦相似度当作通过标准."""
    if (
        reference.shape != actual.shape
        or reference.dtype != actual.dtype
        or not torch.isfinite(actual).all()
        or not torch.isfinite(reference).all()
    ):
        raise ValueError("Output shape mismatch or non-finite values")
    delta = (actual.detach().double().cpu() - reference.detach().double().cpu()).abs()
    return {"max_abs": delta.max().item(), "rmse": delta.square().mean().sqrt().item()}


def action_errors(reference: torch.Tensor, actual: torch.Tensor, pose_starts: list[int]) -> dict:
    """物理误差：位置欧氏距离、SO(3) 测地角与位姿组外的夹爪差."""
    error_metrics(reference, actual)
    a, b = reference.double().cpu(), actual.double().cpu()
    position, rotation = [], []
    pose_dims = set()
    for start in pose_starts:
        position.extend(
            ((a[..., start : start + 3] - b[..., start : start + 3]).norm(dim=-1) * 1000).flatten().tolist()
        )
        ra = axis_angle_to_matrix(a[..., start + 3 : start + 6])
        rb = axis_angle_to_matrix(b[..., start + 3 : start + 6])
        relative = ra.transpose(-1, -2) @ rb
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
        rotation.extend(torch.rad2deg(torch.acos(cosine)).flatten().tolist())
        pose_dims.update(range(start, start + 6))
    gripper_dims = [i for i in range(a.shape[-1]) if i not in pose_dims]
    if len(gripper_dims) != len(pose_starts):
        raise ValueError("Physical validation requires one gripper dimension per 6D pose group")
    return {
        "position_mm": position,
        "rotation_deg": rotation,
        "gripper": (a[..., gripper_dims] - b[..., gripper_dims]).abs().flatten().tolist(),
    }


def summarize(values: list[float]) -> dict:
    """汇总完整分布，max 用于硬门槛，避免均值掩盖少量失真动作."""
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": max(values)}


def load_windows(cfg: ValidationConfig, policy_cfg):
    """从训练数据中确定性选择有效窗口，另外保留每个 episode 起点."""
    training = (
        read_json(cfg.policy_path / "train_config.json")
        if (cfg.policy_path / "train_config.json").exists()
        else {}
    )
    dataset_cfg = training.get("dataset", {})
    root_value = cfg.dataset_root or dataset_cfg.get("root")
    if not root_value or not Path(root_value).is_dir():
        raise FileNotFoundError("A local dataset is required; provide --dataset_root")
    root = Path(root_value)
    metadata = read_json(root / "meta" / "fastumi.json") if (root / "meta" / "fastumi.json").exists() else {}
    units = {
        name: getattr(cfg, name) or metadata.get(name)
        for name in ("position_unit", "rotation_unit", "gripper_representation")
    }
    if units != {"position_unit": "m", "rotation_unit": "rad", "gripper_representation": "normalized_0_1"}:
        raise ValueError("Physical acceptance requires explicit m / rad / normalized_0_1 units")
    repo_id = dataset_cfg.get("repo_id", "local/export-validation")
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    # 按原始特征名读取；非空 rename_map 的数据源通过处理器配置反向映射。
    pre_cfg = read_json(cfg.policy_path / "policy_preprocessor.json")
    rename = next(
        (
            s["config"].get("rename_map", {})
            for s in pre_cfg["steps"]
            if s["registry_name"] == "rename_observations_processor"
        ),
        {},
    )
    inverse = {new: old for old, new in rename.items()}
    delta = resolve_delta_timestamps(policy_cfg, meta, rename_map=rename)
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta, video_backend="pyav")

    def decode(index: int):
        item = dataset[index]
        batch = {}
        for key in policy_cfg.input_features:
            source_key = inverse.get(key, key)
            value = item[source_key]
            if source_key in meta.camera_keys and value.dtype == torch.uint8:
                value = value.float() / 255
            batch[source_key] = value.unsqueeze(0)
        return batch

    windows, boundaries, identities = [], [], []
    for episode in sorted(meta.episodes, key=lambda row: row["episode_index"]):
        first, last = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        low = first + policy_cfg.n_obs_steps - 1
        high = last - policy_cfg.n_action_steps
        if high - low + 1 < cfg.windows_per_episode:
            continue
        indices = np.linspace(low, high, cfg.windows_per_episode, dtype=int).tolist()
        windows.extend(decode(index) for index in indices)
        boundary = decode(first)
        # 起点历史必须重复本 episode 首帧，而不是借用上一 episode 的观测。
        for key, value in boundary.items():
            if value.shape[1] != policy_cfg.n_obs_steps or not all(
                torch.equal(value[:, i], value[:, -1]) for i in range(policy_cfg.n_obs_steps)
            ):
                raise ValueError(f"Episode-start padding mismatch: {episode['episode_index']}, {key}")
        boundaries.append(boundary)
        identities.append({"episode": int(episode["episode_index"]), "indices": indices, "start": first})
        if len(identities) == cfg.episodes:
            break
    if len(identities) != cfg.episodes:
        raise ValueError(f"Not enough episodes for requested validation: {len(identities)}")
    return windows, boundaries, {"root": str(root), "units": units, "windows": identities}


class DeviceMemoryMonitor:
    """采样整卡已用显存，包含 TensorRT 原生分配；记录为采样峰值而非精确峰值."""

    def __init__(self, device: str):
        """初始化后台采样器，不改变模型设备或 CUDA 上下文."""
        self.device = device
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        """开始记录整卡显存用量."""
        if torch.device(self.device).type == "cuda":
            self.sample()
            self.thread = threading.Thread(target=self.poll, daemon=True)
            self.thread.start()
        return self

    def sample(self):
        """CudaMemGetInfo 是整卡统计，包含其他进程，因此报告明确说明口径."""
        free, total = torch.cuda.mem_get_info(self.device)
        self.samples.append((total - free) / 1024**2)

    def poll(self):
        """每 10ms 采样一次，避免仅观察 torch caching allocator."""
        while not self.stop_event.wait(0.01):
            self.sample()

    def __exit__(self, *_):
        """终止采样线程并补记退出前的显存用量."""
        self.stop_event.set()
        if self.thread:
            self.thread.join()
            self.sample()


def benchmark_runner(predict, reset, windows: list, cfg: ValidationConfig) -> dict:
    """排除解码/加载/构建，完整动作块包含处理器、采样和必要搬运."""
    latencies = []
    with DeviceMemoryMonitor(cfg.device) as memory:
        for i in range(cfg.warmup + cfg.iterations):
            reset()
            synchronize(cfg.device)
            started = time.perf_counter()
            predict(windows[i % len(windows)], i % 3)
            synchronize(cfg.device)
            elapsed = (time.perf_counter() - started) * 1000
            if i >= cfg.warmup:
                latencies.append(elapsed)
    stages = []
    # 阶段耗时独立采集，额外同步不会污染上面的完整调用分位数。
    for i in range(3):
        reset()
        synchronize(cfg.device)
        stages.append(predict(windows[i % len(windows)], i, profile=True)[3])
    return {
        "p50_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "iterations": cfg.iterations,
        "warmup": cfg.warmup,
        "device_used_memory_sampled_peak_mib": max(memory.samples) if memory.samples else None,
        "memory_scope": "whole device, sampled every 10ms; includes TensorRT and other processes",
        "component_mean_ms": {key: float(np.mean([s[key] for s in stages])) for key in stages[0]},
    }


@torch.inference_mode()
def validate_export(cfg: ValidationConfig) -> dict:
    """逐后端验证并持久化阶段结果，某后端失败不丢弃其他后端的报告."""
    if min(cfg.episodes, cfg.windows_per_episode, cfg.iterations) < 1 or cfg.warmup < 0 or not cfg.seeds:
        raise ValueError("Validation counts must be positive and seeds must not be empty")
    if min(cfg.position_tolerance_mm, cfg.rotation_tolerance_deg, cfg.gripper_tolerance) <= 0:
        raise ValueError("Action tolerances must be positive")
    if not cfg.backends or set(cfg.backends) - {"onnx", "tensorrt_fp32", "tensorrt_fp16"}:
        raise ValueError("Unsupported or empty validation backend list")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = load_manifest(cfg.export_dir)
    if (
        file_hash(cfg.policy_path / "model.safetensors") != manifest["checkpoint_sha256"]
        or file_hash(cfg.policy_path / "config.json") != manifest["config_sha256"]
    ):
        raise ValueError("Reference checkpoint does not match exported package")
    policy = load_policy(cfg.policy_path, cfg.device)
    pre, post = load_processors(cfg.policy_path, policy.config, cfg.device)
    windows, boundaries, dataset_info = load_windows(cfg, policy.config)
    pose_starts = policy.config.relative_pose_action_starts or policy.config.relative_pose_state_starts
    report = {
        "status": "running",
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "software": {
            name: importlib.metadata.version(name) for name in ("torch", "onnx", "onnxruntime", "diffusers")
        },
        "package_manifest_sha256": file_hash(cfg.export_dir / "manifest.json"),
        "dataset": dataset_info,
        "seeds": cfg.seeds,
        "device": cfg.device,
        "thresholds": {
            "position_mm": cfg.position_tolerance_mm,
            "rotation_deg": cfg.rotation_tolerance_deg,
            "gripper": cfg.gripper_tolerance,
        },
        "backends": {},
        "benchmark": {},
        "boundary_windows": len(boundaries),
    }
    report_path = cfg.export_dir / "validation_report.json"
    write_json(report_path, report)

    def reset_reference():
        policy.reset()
        pre.reset()
        post.reset()

    def reference(raw, seed, profile=False):
        timings = {}

        def mark(name, started):
            if profile:
                synchronize(cfg.device)
                timings[name] = (time.perf_counter() - started) * 1000
            return time.perf_counter()

        started = time.perf_counter()
        batch = pre(dict(raw))
        inputs = encoder_inputs(batch, policy.config)
        started = mark("preprocess_ms", started)
        cond = policy.diffusion._prepare_global_conditioning(inputs)
        started = mark("encoder_ms", started)
        generator = torch.Generator(device=cfg.device).manual_seed(seed)
        trajectory = policy.diffusion.conditional_sample(1, global_cond=cond, generator=generator)
        start, end = manifest["action_slice"]
        normalized = trajectory[:, start:end]
        started = mark("sampling_ms", started)
        absolute = post(normalized.clone())
        mark("postprocess_ms", started)
        return absolute, normalized, cond, timings

    # 原生参考只计算一次，并保存 CPU 输出；不同后端共享同一份解码观测。
    all_windows = windows + boundaries
    references = []
    for index, raw in enumerate(all_windows):
        records = []
        for seed in cfg.seeds:
            reset_reference()
            result, normalized, cond, _ = reference(raw, seed)
            records.append((result.cpu(), normalized.cpu(), cond.cpu()))
        references.append(records)
        logging.info("Reference window %s/%s", index + 1, len(all_windows))
    if cfg.benchmark:
        report["benchmark"]["pytorch"] = benchmark_runner(reference, reset_reference, windows, cfg)
        write_json(report_path, report)
    modules = make_export_modules(policy)
    for backend in ["torch_wrapper", *cfg.backends]:
        runner = session = None
        try:
            if backend == "torch_wrapper":
                session = TorchSession(modules)
            elif backend == "onnx":
                session = OnnxSession(cfg.export_dir, manifest, cfg.device)
            else:
                session = TensorRTSession(
                    cfg.export_dir, manifest, cfg.device, backend.removeprefix("tensorrt_")
                )
            runner = ExportedDiffusion(cfg.export_dir, session, cfg.device, manifest)
            errors = {"position_mm": [], "rotation_deg": [], "gripper": []}
            components = []
            observed_ms = []
            for index, raw in enumerate(all_windows):
                for seed_index, seed in enumerate(cfg.seeds):
                    runner.reset()
                    synchronize(cfg.device)
                    started = time.perf_counter()
                    actual, normalized, cond, _ = runner.predict(raw, seed)
                    synchronize(cfg.device)
                    observed_ms.append((time.perf_counter() - started) * 1000)
                    ref_action, ref_normalized, ref_cond = references[index][seed_index]
                    metrics = action_errors(ref_action, actual, pose_starts)
                    for key in errors:
                        errors[key].extend(metrics[key])
                    # 固定网络输入直接检查去噪器，避免将采样差异归因于单层网络。
                    generator = torch.Generator(device=cfg.device).manual_seed(seed)
                    sample = torch.randn(
                        (1, policy.config.horizon, policy.config.model_action_dim),
                        generator=generator,
                        device=cfg.device,
                    )
                    # 固定单步探针覆盖采样时间轴起点、中点和终点。
                    probe_timestep = (
                        seed_index * (policy.config.num_train_timesteps - 1) // max(len(cfg.seeds) - 1, 1)
                    )
                    timestep = torch.tensor([probe_timestep], device=cfg.device)
                    denoise_inputs = {
                        "sample": sample,
                        "timestep": timestep,
                        "global_cond": ref_cond.to(cfg.device),
                    }
                    eager = modules["denoiser"][0](**denoise_inputs)
                    components.append(
                        {
                            "window": index,
                            "seed": seed,
                            "encoder": error_metrics(ref_cond, cond),
                            "denoiser": error_metrics(eager, session.run("denoiser", denoise_inputs)),
                            "normalized_actions": error_metrics(ref_normalized, normalized),
                        }
                    )
                logging.info("%s window %s/%s", backend, index + 1, len(all_windows))
            summary = {key: summarize(values) for key, values in errors.items()}
            passed = all(summary[key]["max"] <= limit for key, limit in report["thresholds"].items())
            report["backends"][backend] = {"passed": passed, "actions": summary, "components": components}
            if backend == "onnx":
                # 此项是验证期间的 CPU ORT 观测延迟，不混入 GPU benchmark 加速比。
                report["onnx_cpu_validation_latency"] = {
                    "p50_ms": float(np.median(observed_ms)),
                    "p95_ms": float(np.percentile(observed_ms, 95)),
                    "samples": len(observed_ms),
                    "includes": "CPU networks + scheduler on reference device + transfers",
                }
            if cfg.benchmark and passed and backend.startswith("tensorrt"):
                report["benchmark"][backend] = benchmark_runner(runner.predict, runner.reset, windows, cfg)
                baseline = report["benchmark"]["pytorch"]["p50_ms"]
                report["benchmark"][backend]["p50_speedup"] = (
                    baseline / report["benchmark"][backend]["p50_ms"]
                )
        except Exception as exc:
            logging.exception("Validation failed for %s", backend)
            report["backends"][backend] = {"passed": False, "error": str(exc)}
        finally:
            del runner, session
            gc.collect()
            if torch.device(cfg.device).type == "cuda":
                torch.cuda.empty_cache()
            write_json(report_path, report)
    report["status"] = "passed" if all(r["passed"] for r in report["backends"].values()) else "failed"
    write_json(report_path, report)
    logging.info("Validation %s; report: %s", report["status"], report_path)
    return report
