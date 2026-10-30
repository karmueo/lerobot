# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""验证 ACT 导出包与原生推理的一致性及接口边界。"""

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.export.backends import OnnxSession, TorchSession
from lerobot.export.exporter import ExportConfig, export_policy
from lerobot.export.package import load_manifest, load_policy, load_processors
from lerobot.export.runtime import ExportedACT
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.export_act import make_export_modules
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def test_act_export_roundtrip(tmp_path):
    """比较原生推理、包装及 ONNX 完整动作块，验证包不依赖原检查点。"""
    torch.manual_seed(42)
    source, output = tmp_path / "checkpoint", tmp_path / "export"
    image_key = "observation.images.front"
    cfg = ACTConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        pretrained_backbone_weights=None,
        normalization_mapping={
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.MEAN_STD,
        },
    )
    policy = ACTPolicy(cfg).eval()
    policy.save_pretrained(source)
    stats = {
        OBS_STATE: {"mean": torch.full((7,), 0.2), "std": torch.full((7,), 0.5)},
        ACTION: {"mean": torch.full((7,), 0.3), "std": torch.full((7,), 0.4)},
        image_key: {"mean": torch.full((3, 1, 1), 0.4), "std": torch.full((3, 1, 1), 0.2)},
    }
    pre, post = make_act_pre_post_processors(cfg, stats)
    pre.save_pretrained(source)
    post.save_pretrained(source)
    export_policy(ExportConfig(source, output, format="onnx", device="cpu"))
    manifest = load_manifest(output)
    assert manifest["policy_type"] == "act" and manifest["action_slice"] == [0, 2]
    native = load_policy(source, "cpu")
    pre, post = load_processors(source, native.config, "cpu")
    raw = {OBS_STATE: torch.randn(1, 7), image_key: torch.rand(1, 3, 64, 64)}
    with torch.inference_mode():
        expected = post(native.predict_action_chunk(pre(dict(raw))).clone())
        modules = make_export_modules(native)
        source.rename(tmp_path / "unavailable")
        for session in (TorchSession(modules), OnnxSession(output, manifest, "cpu")):
            runner = ExportedACT(output, session, "cpu", manifest)
            full = runner.predict_chunk(raw)[0]
            torch.testing.assert_close(full, expected, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(runner.predict(raw)[0], full[:, :2], atol=0, rtol=0)
            if isinstance(session, OnnxSession):
                with pytest.raises(ValueError, match="shape/dtype"):
                    runner.predict({OBS_STATE: torch.zeros(2, 7), image_key: torch.rand(2, 3, 64, 64)})
    (output / "policy.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(output)


def test_export_rejects_temporal_ensemble():
    """运行器未实现时间集成时拒绝导出，避免静默改变策略行为."""
    cfg = ACTConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=4,
        n_action_steps=1,
        temporal_ensemble_coeff=0.01,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    with pytest.raises(NotImplementedError, match="temporal ensembling"):
        make_export_modules(ACTPolicy(cfg).eval())
