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
"""模型包的完整性、配置和处理器资产管理."""

import hashlib
import json
from pathlib import Path

import torch

from lerobot.configs.types import NormalizationMode
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def read_json(path: Path):
    """读取 UTF-8 配置，保留 JSON 中声明的特征顺序."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    """原子更新报告，失败时不会留下半份 JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    """流式计算权重摘要，避免将 GB 级文件再次载入内存."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(root: Path, device: str) -> DiffusionConfig:
    """只接受本地 Diffusion 配置，不通过 Hub 自动补齐缺失文件."""
    if not root.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {root}")
    if read_json(root / "config.json").get("type") != "diffusion":
        raise NotImplementedError("Export currently supports Diffusion policies only")
    cfg = DiffusionConfig.from_pretrained(root, local_files_only=True)
    cfg.device = device
    cfg.compile_model = False
    # 权重来自检查点，禁止初始化时联网下载 torchvision 权重。
    cfg.pretrained_backbone_weights = None
    cfg.validate_features()
    return cfg


def load_policy(root: Path, device: str) -> DiffusionPolicy:
    """严格加载全部网络参数，缺失或多余权重都必须报错."""
    return (
        DiffusionPolicy.from_pretrained(
            root, config=load_config(root, device), local_files_only=True, strict=True
        )
        .eval()
        .float()
    )


def processor_assets(root: Path) -> list[str]:
    """检查处理步骤与统计量；不允许未实现的处理器被静默忽略."""
    allowed = {
        "rename_observations_processor",
        "to_batch_processor",
        "device_processor",
        "relative_pose_processor",
        "absolute_pose_processor",
        "normalizer_processor",
        "unnormalizer_processor",
    }
    files = ["config.json"]
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        files.append(name)
        for step in read_json(root / name)["steps"]:
            if step.get("registry_name") not in allowed:
                raise NotImplementedError(f"Unsupported export processor: {step.get('registry_name')}")
            state = step.get("state_file")
            if state:
                # 包内资产必须为同目录文件，禁止源配置引用包外路径。
                if Path(state).name != state or not (root / state).is_file():
                    raise FileNotFoundError(f"Missing or invalid processor state: {state}")
                files.append(state)
            elif step["registry_name"] in ("normalizer_processor", "unnormalizer_processor"):
                modes = step.get("config", {}).get("norm_map", {}).values()
                if any(mode != "IDENTITY" for mode in modes):
                    raise ValueError(f"Missing normalization statistics in {name}")
    if (root / "train_config.json").is_file():
        files.append("train_config.json")
    return sorted(set(files))


def load_processors(root: Path, cfg: DiffusionConfig, device: str):
    """通过现有工厂恢复共享 anchor；统计量始终来自检查点而非数据集."""
    processor_assets(root)
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(root),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    required_stats = {
        NormalizationMode.MEAN_STD: ("mean", "std"),
        NormalizationMode.MIN_MAX: ("min", "max"),
        NormalizationMode.QUANTILES: ("q01", "q99"),
        NormalizationMode.QUANTILE10: ("q10", "q90"),
    }
    # 原始 normalizer 在整项统计缺失时会原样返回；导出必须拒绝这种静默退化。
    for pipeline, features in ((pre, cfg.input_features), (post, cfg.output_features)):
        normalizers = [step for step in pipeline.steps if hasattr(step, "_tensor_stats")]
        for name, feature in features.items():
            mode = cfg.normalization_mapping.get(feature.type, NormalizationMode.IDENTITY)
            if mode == NormalizationMode.IDENTITY:
                continue
            matching = [step for step in normalizers if name in step.features]
            if len(matching) != 1 or matching[0].norm_map.get(feature.type) != mode:
                raise ValueError(f"Processor normalization does not match policy config: {name}")
            values = matching[0]._tensor_stats.get(name, {})
            for stat in required_stats[mode]:
                if stat not in values or not torch.isfinite(values[stat]).all():
                    raise ValueError(f"Missing or invalid normalization statistic: {name}.{stat}")
    return pre, post


def load_manifest(root: Path) -> dict:
    """验证已完成模型包及资产摘要，拒绝残留、损坏或包外文件."""
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Export package is incomplete or has an unsupported schema")
    for name, digest in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Export artifact checksum mismatch: {name}")
    return manifest
