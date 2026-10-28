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
"""离线导出编排：先在临时目录完成全部资产，再发布完整模型包."""

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lerobot.policies.diffusion.export_diffusion import make_export_modules
from lerobot.utils.import_utils import _onnx_available, require_package

from .backends import build_engine
from .package import file_hash, load_policy, load_processors, processor_assets, read_json, write_json
from .runtime import encoder_inputs

if TYPE_CHECKING or _onnx_available:
    import onnx


@dataclass
class ExportConfig:
    """导出 CLI；输入输出路径显式提供，避免误用其他训练检查点."""

    policy_path: Path
    output_dir: Path
    format: str = "both"
    device: str = "cuda:0"
    trt_precision: str = "both"
    overwrite: bool = False


def tensor_spec(tensor: torch.Tensor) -> dict:
    """序列化固定输入接口，而不是假定所有策略使用同一维度."""
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def export_policy(cfg: ExportConfig) -> Path:
    """导出配置驱动的 Diffusion 包，错误时保留之前的有效产物."""
    if cfg.format not in {"onnx", "tensorrt", "both"}:
        raise ValueError("format must be onnx, tensorrt or both")
    if cfg.trt_precision not in {"fp32", "fp16", "both"}:
        raise ValueError("trt_precision must be fp32, fp16 or both")
    source, output = cfg.policy_path.resolve(), cfg.output_dir.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Export directory must not overlap the checkpoint directory")
    if output.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite=true explicitly")
        # 即使显式覆盖也只替换本工具创建的目录，不删除任意用户目录。
        if not (output / "manifest.json").is_file():
            raise ValueError("Refusing to overwrite a directory without an export manifest")
    if cfg.format != "onnx" and torch.device(cfg.device).type != "cuda":
        raise ValueError("TensorRT export requires a CUDA device")
    require_package("onnx", extra="export")
    assets = processor_assets(source)
    policy = load_policy(source, cfg.device)
    pre, post = load_processors(source, policy.config, cfg.device)
    modules = make_export_modules(policy)
    # 在写任何产物前确认保存的处理器与模型内部维度兼容。
    raw = {
        name: torch.zeros(1, policy.config.n_obs_steps, *feature.shape)
        for name, feature in policy.config.input_features.items()
    }
    rename = next(
        (
            s["config"].get("rename_map", {})
            for s in read_json(source / "policy_preprocessor.json")["steps"]
            if s["registry_name"] == "rename_observations_processor"
        ),
        {},
    )
    inverse = {new: old for old, new in rename.items()}
    raw = {inverse.get(name, name): value for name, value in raw.items()}
    prepared = encoder_inputs(pre(raw), policy.config)
    for name, sample in zip(modules["encoder"][2], modules["encoder"][1], strict=True):
        if prepared[name].shape != sample.shape or prepared[name].dtype != sample.dtype:
            raise ValueError(f"Checkpoint processor/model interface mismatch: {name}")
    restored = post(
        torch.zeros(1, policy.config.n_action_steps, policy.config.model_action_dim, device=cfg.device)
    )
    if list(restored.shape) != [1, policy.config.n_action_steps, policy.config.action_feature.shape[0]]:
        raise ValueError("Checkpoint action postprocessor shape mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        root = Path(tmp)
        package_assets = root / "assets"
        package_assets.mkdir()
        for name in assets:
            shutil.copy2(source / name, package_assets / name)
        write_json(package_assets / "scheduler.json", dict(policy.diffusion.noise_scheduler.config))
        manifest = {
            "schema_version": 1,
            "status": "building",
            "policy_type": "diffusion",
            "batch_size": 1,
            "checkpoint": str(source),
            "checkpoint_sha256": file_hash(source / "model.safetensors"),
            "config_sha256": file_hash(source / "config.json"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "onnx": onnx.__version__,
            "image_features": list(policy.config.image_features),
            "raw_state_dim": policy.config.robot_state_feature.shape[0],
            "raw_action_dim": policy.config.action_feature.shape[0],
            "model_state_dim": policy.config.model_state_dim,
            "model_action_dim": policy.config.model_action_dim,
            "action_slice": [
                policy.config.n_obs_steps - 1,
                policy.config.n_obs_steps - 1 + policy.config.n_action_steps,
            ],
            "num_inference_steps": policy.diffusion.num_inference_steps,
            "stages": {},
        }
        for stage, (module, inputs, names, output_name) in modules.items():
            logging.info("Exporting %s to ONNX", stage)
            path = root / f"{stage}.onnx"
            with torch.inference_mode():
                torch.onnx.export(
                    module,
                    inputs,
                    str(path),
                    opset_version=18,
                    dynamo=False,
                    input_names=names,
                    output_names=[output_name],
                )
                result = module(*inputs)
            onnx.checker.check_model(str(path))
            spec = {
                "onnx": path.name,
                "inputs": {name: tensor_spec(t) for name, t in zip(names, inputs, strict=True)},
                "output": {"name": output_name, **tensor_spec(result)},
                "engines": {},
            }
            manifest["stages"][stage] = spec
            if cfg.format != "onnx":
                precisions = ["fp32", "fp16"] if cfg.trt_precision == "both" else [cfg.trt_precision]
                for precision in precisions:
                    logging.info("Building %s TensorRT %s", stage, precision)
                    engine_path = root / f"{stage}_{precision}.engine"
                    info = build_engine(path, engine_path, precision, cfg.device)
                    spec["engines"][precision] = {"path": engine_path.name, **info}
        manifest["files"] = {
            p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()
        }
        manifest["status"] = "complete"
        write_json(root / "manifest.json", manifest)
        if output.exists():
            # 先完成新包，再替换旧包；构建失败不会破坏原有成果。
            if read_json(output / "manifest.json").get("policy_type") != "diffusion":
                raise ValueError("Existing directory is not a Diffusion export package")
            with tempfile.TemporaryDirectory(prefix=f".{output.name}-backup-", dir=output.parent) as backup:
                previous = Path(backup) / "previous"
                output.replace(previous)
                try:
                    root.replace(output)
                except OSError:
                    previous.replace(output)
                    raise
        else:
            root.replace(output)
    logging.info("Export package written to %s", output)
    return output
