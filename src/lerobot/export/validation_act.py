# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""在真实观测上比较 ACT 原生推理与已导出的各后端."""

import gc
import importlib.metadata
import logging
import time

import numpy as np
import torch

from lerobot.policies.act.export_act import make_export_modules

from .backends import OnnxSession, TensorRTSession, TorchSession
from .package import file_hash, load_policy, load_processors, processor_assets, write_json
from .runtime import ExportedACT, synchronize
from .validation import action_errors, benchmark_runner, error_metrics, load_windows, summarize


@torch.inference_mode()
def validate_act_export(cfg, manifest):
    """验证完整 50 步等动作块的物理误差，并保留各后端独立结果."""
    for name in processor_assets(cfg.policy_path):
        if file_hash(cfg.policy_path / name) != manifest["files"].get(f"assets/{name}"):
            raise ValueError(f"Reference processor asset differs from export: {name}")
    policy = load_policy(cfg.policy_path, cfg.device)
    pre, post = load_processors(cfg.policy_path, policy.config, cfg.device)
    windows, boundaries, dataset_info = load_windows(cfg, policy.config)
    if policy.config.action_feature.shape[0] != 7 or dataset_info["action_layout"] != "xyz_rotvec_gripper":
        raise ValueError(
            "ACT physical validation requires metadata action_layout=xyz_rotvec_gripper and 7D actions"
        )
    report = {
        "status": "running",
        "policy_type": "act",
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "software": {name: importlib.metadata.version(name) for name in ("torch", "onnx", "onnxruntime")},
        "package_manifest_sha256": file_hash(cfg.export_dir / "manifest.json"),
        "dataset": dataset_info,
        "device": cfg.device,
        "thresholds": {
            "position_mm": cfg.position_tolerance_mm,
            "rotation_deg": cfg.rotation_tolerance_deg,
            "gripper": cfg.gripper_tolerance,
        },
        "backends": {},
        "benchmark": {},
        "boundary_windows": len(boundaries),
        "action_steps_compared": policy.config.chunk_size,
    }
    report_path = cfg.export_dir / "validation_report.json"
    write_json(report_path, report)

    def reset_reference():
        policy.reset()
        pre.reset()
        post.reset()

    def reference(raw, seed=0, profile=False):
        timings = {}

        def mark(name, start):
            if profile:
                synchronize(cfg.device)
                timings[name] = (time.perf_counter() - start) * 1000
            return time.perf_counter()

        started = time.perf_counter()
        batch = pre(dict(raw))
        started = mark("preprocess_ms", started)
        normalized = policy.predict_action_chunk(batch)
        started = mark("policy_ms", started)
        action = post(normalized.clone())
        mark("postprocess_ms", started)
        return action, normalized, None, timings

    all_windows = windows + boundaries
    references = []
    for raw in all_windows:
        reset_reference()
        action, normalized, _, _ = reference(raw)
        references.append((action.cpu(), normalized.cpu()))
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
            runner = ExportedACT(cfg.export_dir, session, cfg.device, manifest)
            errors = {"position_mm": [], "rotation_deg": [], "gripper": []}
            components, observed_ms = [], []
            for index, raw in enumerate(all_windows):
                runner.reset()
                synchronize(cfg.device)
                started = time.perf_counter()
                actual, normalized, _, _ = runner.predict_chunk(raw)
                synchronize(cfg.device)
                observed_ms.append((time.perf_counter() - started) * 1000)
                expected, ref_normalized = references[index]
                for key, values in action_errors(expected, actual, [0]).items():
                    errors[key].extend(values)
                components.append(
                    {"window": index, "normalized_actions": error_metrics(ref_normalized, normalized)}
                )
            summary = {key: summarize(values) for key, values in errors.items()}
            passed = all(summary[key]["max"] <= limit for key, limit in report["thresholds"].items())
            report["backends"][backend] = {"passed": passed, "actions": summary, "components": components}
            if backend == "onnx":
                report["onnx_cpu_validation_latency"] = {
                    "p50_ms": float(np.median(observed_ms)),
                    "p95_ms": float(np.percentile(observed_ms, 95)),
                    "samples": len(observed_ms),
                    "includes": "CPU network + preprocessing/postprocessing and transfers",
                }
            if cfg.benchmark and passed and backend.startswith("tensorrt"):
                result = benchmark_runner(runner.predict_chunk, runner.reset, windows, cfg)
                result["p50_speedup"] = report["benchmark"]["pytorch"]["p50_ms"] / result["p50_ms"]
                report["benchmark"][backend] = result
        except Exception as exc:
            logging.exception("ACT validation failed for %s", backend)
            report["backends"][backend] = {"passed": False, "error": str(exc)}
        finally:
            del runner, session
            gc.collect()
            if torch.device(cfg.device).type == "cuda":
                torch.cuda.empty_cache()
            write_json(report_path, report)
    report["status"] = (
        "passed" if all(result["passed"] for result in report["backends"].values()) else "failed"
    )
    write_json(report_path, report)
    return report
