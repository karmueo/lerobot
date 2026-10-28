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
"""验证导出后的完整采样和位姿处理，不依赖真实大模型或 GPU."""

import json

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.export.backends import OnnxSession, TorchSession, check_inputs
from lerobot.export.exporter import ExportConfig, export_policy
from lerobot.export.package import load_manifest, load_policy, load_processors, processor_assets
from lerobot.export.runtime import ExportedDiffusion, encoder_inputs
from lerobot.export.validation import action_errors, error_metrics
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.export_diffusion import FixedRgbEncoder, make_export_modules
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def checkpoint(root, relative=False, scheduler="DDPM", images=False, cameras=1, separate=True):
    """小网络保留真实调度器与处理器，统计量维度覆盖 7→10 的区别."""
    torch.manual_seed(42)
    features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))}
    if images:
        # 保留至少 3×3 backbone 特征，避免 spatial softmax 在单像素上退化成常量。
        features["observation.images.front"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96))
        if cameras == 2:
            features["observation.images.wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96))
    else:
        features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(2,))
    cfg = DiffusionConfig(
        device="cpu",
        input_features=features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        use_relative_pose=relative,
        noise_scheduler_type=scheduler,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        num_train_timesteps=4,
        down_dims=(16, 32),
        n_groups=4,
        diffusion_step_embed_dim=16,
        pretrained_backbone_weights=None,
        crop_shape=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=4,
        use_separate_rgb_encoder_per_camera=separate,
        normalization_mapping={
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
            "VISUAL": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
        },
    )
    policy = DiffusionPolicy(cfg).eval()
    policy.save_pretrained(root)
    stats = {
        key: {"min": torch.full((dim,), -1.0), "max": torch.ones(dim)}
        for key, dim in [(OBS_STATE, cfg.model_state_dim), (ACTION, cfg.model_action_dim)]
    }
    pre, post = make_diffusion_pre_post_processors(cfg, stats)
    pre.save_pretrained(root)
    post.save_pretrained(root)
    return cfg


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("scheduler", ["DDPM", "DDIM"])
def test_roundtrip_full_sampler(tmp_path, relative, scheduler):
    """对齐原生采样、包装采样与重新加载的 ORT 包，并验证 seed/anchor/reset."""
    source, output = tmp_path / "checkpoint", tmp_path / "export"
    cfg = checkpoint(source, relative, scheduler)
    export_policy(ExportConfig(source, output, format="onnx", device="cpu"))
    manifest = load_manifest(output)
    assert manifest["model_action_dim"] == (10 if relative else 7)
    assert manifest["action_slice"] == [1, 3]
    native = load_policy(source, "cpu")
    pre, post = load_processors(source, cfg, "cpu")
    raw = {OBS_STATE: torch.randn(1, 2, 7) * 0.1, OBS_ENV_STATE: torch.randn(1, 2, 2)}
    inputs = encoder_inputs(pre(dict(raw)), cfg)
    with torch.inference_mode():
        cond = native.diffusion._prepare_global_conditioning(inputs)
        trajectory = native.diffusion.conditional_sample(1, cond, generator=torch.Generator().manual_seed(3))
        expected = post(trajectory[:, 1:3].clone())
        # 运行模型包不应偷偷回读原检查点的权重、配置或统计量。
        source.rename(tmp_path / "original_checkpoint_unavailable")
        for session in [TorchSession(make_export_modules(native)), OnnxSession(output, manifest, "cpu")]:
            runtime = ExportedDiffusion(output, session, "cpu", manifest)
            actual = runtime.predict(raw, seed=3)[0]
            torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-4)
            # 干扰历史后 reset，必须恢复同一窗口的相同动作。
            runtime.predict({k: v + 0.1 for k, v in raw.items()}, seed=4)
            runtime.reset()
            torch.testing.assert_close(actual, runtime.predict(raw, seed=3)[0], atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        export_policy(ExportConfig(source, output, format="onnx", device="cpu"))
    (output / "encoder.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(output)


@pytest.mark.parametrize("cameras,separate", [(1, True), (2, True), (2, False)])
def test_image_encoder_export(tmp_path, cameras, separate):
    """实际导出视觉网络，确保相机与时间维顺序没有因打平改变."""
    source, output = tmp_path / "checkpoint", tmp_path / "export"
    checkpoint(source, relative=True, images=True, cameras=cameras, separate=separate)
    export_policy(ExportConfig(source, output, format="onnx", device="cpu"))
    policy = load_policy(source, "cpu")
    modules = make_export_modules(policy)
    module, sample_inputs, names, _ = modules["encoder"]
    inputs = {name: torch.randn_like(tensor) for name, tensor in zip(names, sample_inputs, strict=True)}
    with torch.inference_mode():
        expected = module(*inputs.values())
        actual = OnnxSession(output, load_manifest(output), "cpu").run("encoder", inputs)
    torch.testing.assert_close(expected, actual, atol=1e-4, rtol=1e-3)
    # 与原生编码器也比较，避免包装和 ONNX 共享同一个错误而测试仍通过。
    with torch.inference_mode():
        native = policy.diffusion._prepare_global_conditioning(inputs)
    torch.testing.assert_close(expected, native, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("resize", [(64, 72), (120, 130), (96, 96)])
def test_fixed_resize_crop_matches_native(tmp_path, resize):
    """抗锯齿矩阵与原生下采样、上采样及奇数中心裁剪保持相同结果."""
    cfg = checkpoint(tmp_path, images=True)
    cfg.resize_shape = resize
    cfg.crop_shape = (65, 67)
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder

    native = DiffusionRgbEncoder(cfg).eval()
    wrapped = FixedRgbEncoder(native, cfg).eval()
    images = torch.rand(2, 3, 96, 96)
    with torch.inference_mode():
        torch.testing.assert_close(native(images), wrapped(images), atol=1e-5, rtol=1e-4)


def test_invalid_assets_and_policy(tmp_path):
    """统计量缺失或类型不符时提前失败，不生成误导性的有效包."""
    checkpoint(tmp_path)
    config = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    filename = next(s["state_file"] for s in config["steps"] if "state_file" in s)
    (tmp_path / filename).unlink()
    with pytest.raises(FileNotFoundError):
        processor_assets(tmp_path)
    (tmp_path / "config.json").write_text('{"type":"act"}')
    with pytest.raises(NotImplementedError):
        load_policy(tmp_path, "cpu")


def test_fixed_shapes_and_physical_metrics():
    """拒绝错误 batch；使用测地角避免等价轴角的符号跳变产生假阳性."""
    with pytest.raises(ValueError, match="shape/dtype"):
        check_inputs({"x": torch.ones(2, 7)}, {"x": {"shape": [1, 7], "dtype": "torch.float32"}})
    a = torch.zeros(1, 2, 7)
    b = a.clone()
    b[..., 0] = 0.001
    b[..., 5] = torch.pi / 180
    b[..., 6] = 0.01
    errors = action_errors(a, b, [0])
    assert errors["position_mm"] == pytest.approx([1, 1])
    assert errors["rotation_deg"] == pytest.approx([1, 1])
    assert errors["gripper"] == pytest.approx([0.01, 0.01])
    a[..., 5], b[..., 5] = torch.pi, -torch.pi
    assert max(action_errors(a, b, [0])["rotation_deg"]) < 1e-4
    with pytest.raises(ValueError, match="non-finite"):
        error_metrics(a, torch.full_like(a, torch.nan))


def test_build_failure_preserves_previous_package(tmp_path, monkeypatch):
    """构建失败发生在临时目录，显式覆盖也不能破坏之前的有效导出."""
    from lerobot.export import exporter

    source, output = tmp_path / "checkpoint", tmp_path / "export"
    checkpoint(source)
    export_policy(ExportConfig(source, output, format="onnx", device="cpu"))
    original = (output / "manifest.json").read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(exporter.torch.onnx, "export", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        export_policy(ExportConfig(source, output, format="onnx", device="cpu", overwrite=True))
    assert (output / "manifest.json").read_bytes() == original


def test_missing_optional_dependency(monkeypatch, tmp_path):
    """未安装运行时依赖时报告安装提示，不被后续文件或 SDK 错误掩盖."""
    from lerobot.export import backends

    def missing(*args, **kwargs):
        raise ImportError("Install lerobot[export]")

    monkeypatch.setattr(backends, "require_package", missing)
    with pytest.raises(ImportError, match="Install"):
        backends.OnnxSession(tmp_path, {}, "cpu")


def test_missing_stat_feature_rejected(tmp_path):
    """统计文件存在但缺少整个状态特征时，不能沿用 normalizer 的静默直通行为."""
    from safetensors.torch import load_file, save_file

    cfg = checkpoint(tmp_path)
    config = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    filename = next(s["state_file"] for s in config["steps"] if "state_file" in s)
    tensors = load_file(tmp_path / filename)
    save_file({key: value for key, value in tensors.items() if OBS_STATE not in key}, tmp_path / filename)
    with pytest.raises(ValueError, match="statistic"):
        load_processors(tmp_path, cfg, "cpu")
