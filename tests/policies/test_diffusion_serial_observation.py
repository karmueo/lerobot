"""验证原图缩放、双帧静止观测和相对位姿在训练及部署中的一致性。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("datasets")

from lerobot.configs.default import DatasetConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.compute_stats import compute_relative_pose_stats
from lerobot.datasets.factory import make_dataset_image_transforms, resolve_delta_timestamps
from lerobot.export.package import load_processors, processor_assets
from lerobot.export.runtime import encoder_inputs
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.export_diffusion import make_export_modules
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor import DiffusionObservationProcessorStep
from lerobot.processor.diffusion_observation_processor import repeat_latest_observation
from lerobot.transforms.image_resize import LetterboxImageTransform, letterbox_image
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

IMAGE = "observation.images.front"
pytestmark = pytest.mark.dataset


def serial_config(relative: bool = False) -> DiffusionConfig:
    """使用真实视觉编码器与小 U-Net，保留完整窗口和调度器行为。"""
    return DiffusionConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (7,)),
            IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 48, 80)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        serial_observation=True,
        use_relative_pose=relative,
        resize_shape=(64, 64),
        resize_mode="letterbox",
        horizon=8,
        n_action_steps=2,
        down_dims=(16, 32),
        n_groups=4,
        diffusion_step_embed_dim=16,
        num_train_timesteps=4,
        noise_scheduler_type="DDIM",
        pretrained_backbone_weights=None,
        spatial_softmax_num_keypoints=4,
    )


def trajectory() -> torch.Tensor:
    """非零位置、非单位旋转与连续夹爪，可揭示错误的相对锚点。"""
    start = torch.tensor([0.4, -0.2, 0.3, 0.4, -0.3, 1.2, 0.9])
    direction = torch.tensor([0.01, 0.02, -0.005, 0.001, 0.002, 0.003, -0.01])
    return start + torch.arange(16)[:, None] * direction


def statistics(cfg: DiffusionConfig) -> dict:
    """当前帧统计复用生产函数，其他模式给定非退化归一化范围。"""
    if cfg.use_relative_pose:
        states = trajectory().numpy()
        stats = compute_relative_pose_stats(
            {OBS_STATE: states, ACTION: states, "episode_index": np.repeat([0, 1], 8)},
            cfg.observation_delta_indices,
            cfg.action_delta_indices,
            [0],
        )
    else:
        stats = {key: {"min": -torch.ones(7), "max": torch.ones(7)} for key in (OBS_STATE, ACTION)}
    stats[IMAGE] = {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)}
    return stats


@pytest.mark.parametrize("size", [(0, 224), (224, -1), (224,), (2.5, 224), (True, 224)])
def test_invalid_resize_shape(size: tuple) -> None:
    """禁止非正尺寸和静默截断的浮点尺寸。"""
    with pytest.raises(ValueError, match="positive integers"):
        DiffusionConfig(resize_shape=size)


def test_resize_defaults_and_validation() -> None:
    assert DiffusionConfig().resize_shape is None
    assert DiffusionConfig().resize_mode == "stretch"
    assert not DiffusionConfig().serial_observation
    with pytest.raises(ValueError, match="resize_mode"):
        DiffusionConfig(resize_mode="unknown")
    with pytest.raises(ValueError, match="n_obs_steps"):
        DiffusionConfig(n_obs_steps=0)


def test_letterbox_pixels_odd_padding_and_identity() -> None:
    """3×5 缩放为 5×8，额外黑边像素应在下侧；原尺寸保持同一张量。"""
    image = torch.full((2, 3, 3, 5), 255, dtype=torch.uint8)
    result = letterbox_image(image, (8, 8))
    assert result.shape == (2, 3, 8, 8)
    torch.testing.assert_close(result[..., 1:6, :], torch.ones(2, 3, 5, 8))
    assert not result[..., :1, :].any() and not result[..., 6:, :].any()
    assert letterbox_image(result, (8, 8)) is result
    # 转置场景同时验证右侧多出的一个像素。
    torch.testing.assert_close(letterbox_image(image.transpose(-1, -2), (8, 8)), result.transpose(-1, -2))
    with pytest.raises(ValueError, match="image"):
        letterbox_image(torch.zeros(2), (8, 8))


def test_loader_and_inference_geometry_agree() -> None:
    """加载器的 uint8 原图与部署 float 图像使用完全一致的缩放计算。"""
    cfg = serial_config()
    dataset = DatasetConfig(repo_id="handumi/test")
    dataset.image_transforms.enable = False
    pipeline = SimpleNamespace(dataset=dataset, trainable_config=cfg)
    transform = make_dataset_image_transforms(pipeline)
    evaluation_transform = make_dataset_image_transforms(pipeline, training=False)
    assert isinstance(transform, LetterboxImageTransform)
    image = torch.randint(256, (2, 3, 48, 80), dtype=torch.uint8)
    expected = transform(image)
    torch.testing.assert_close(expected, evaluation_transform(image), atol=0, rtol=0)
    step = DiffusionObservationProcessorStep(image_keys=[IMAGE], resize_shape=(64, 64))
    actual = step.observation({IMAGE: image.float() / 255})[IMAGE]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # 禁用 letterbox 后不改变旧的 DataLoader 行为。
    cfg.resize_mode = "stretch"
    assert make_dataset_image_transforms(pipeline) is None


def test_serial_offsets_keep_original_action_alignment() -> None:
    cfg = serial_config()
    meta = SimpleNamespace(features={OBS_STATE: {}, IMAGE: {}, ACTION: {}}, fps=30)
    offsets = resolve_delta_timestamps(cfg, meta)
    assert offsets[OBS_STATE] == offsets[IMAGE] == [0, 0]
    assert offsets[ACTION] == [i / 30 for i in range(-1, 7)]
    cfg.serial_observation = False
    assert cfg.observation_delta_indices == [-1, 0]
    assert cfg.action_delta_indices == list(range(-1, 7))


@pytest.mark.parametrize("relative", [False, True])
def test_preprocessor_roundtrip_serialization_and_black_padding(tmp_path: Path, relative: bool) -> None:
    """历史差异（包括增强差异）被当前帧覆盖，动作与归一化后的黑边可往返恢复。"""
    cfg = serial_config(relative)
    pre, post = make_diffusion_pre_post_processors(cfg, statistics(cfg))
    state = trajectory()[:2].unsqueeze(0)
    image = torch.stack([torch.zeros(3, 48, 80), torch.ones(3, 48, 80)]).unsqueeze(0)
    action = trajectory()[:8].unsqueeze(0)
    raw = {OBS_STATE: state, IMAGE: image, ACTION: action}
    processed = pre(raw)
    torch.testing.assert_close(processed[OBS_STATE][:, 0], processed[OBS_STATE][:, 1], atol=0, rtol=0)
    torch.testing.assert_close(processed[IMAGE][:, 0], processed[IMAGE][:, 1], atol=0, rtol=0)
    assert processed[OBS_STATE].shape == (1, 2, 10 if relative else 7)
    # mean=0.5、std=0.25，归一化前的黑边 0 应成为 -2。
    torch.testing.assert_close(processed[IMAGE][..., 0, :], torch.full((1, 2, 3, 64), -2.0))
    torch.testing.assert_close(post(processed[ACTION]), action, atol=1e-6, rtol=1e-6)
    cfg.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    assert "policy_preprocessor.json" in processor_assets(tmp_path)
    loaded_cfg = DiffusionConfig.from_pretrained(tmp_path, local_files_only=True)
    assert loaded_cfg.serial_observation and loaded_cfg.resize_mode == "letterbox"
    loaded_pre, loaded_post = load_processors(tmp_path, loaded_cfg, "cpu")
    restored = loaded_pre(raw)
    for key in (OBS_STATE, IMAGE, ACTION):
        torch.testing.assert_close(restored[key], processed[key], atol=0, rtol=0)
    torch.testing.assert_close(loaded_post(restored[ACTION]), action, atol=1e-6, rtol=1e-6)


def test_serial_relative_statistics_are_exact_identity() -> None:
    cfg = serial_config(True)
    stats = statistics(cfg)
    identity = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32)
    torch.testing.assert_close(
        torch.as_tensor(stats[OBS_STATE]["mean"][:9]).float(), identity, atol=0, rtol=0
    )
    # 常量位姿的归一化范围仍有下限，避免除零和微小误差放大。
    assert np.all(stats[OBS_STATE]["max"] - stats[OBS_STATE]["min"] > 0)
    pre, _ = make_diffusion_pre_post_processors(cfg, stats)
    relative_step = next(step for step in pre.steps if hasattr(step, "anchor"))
    transition = {"observation": {OBS_STATE: trajectory()[1:2].unsqueeze(0).expand(-1, 2, -1)}}
    # 直接检查归一化前的精确单位位姿，而非被统计范围抵消后的值。
    output = relative_step(transition)["observation"][OBS_STATE]
    torch.testing.assert_close(output[0, :, :9], identity.expand(2, -1), atol=0, rtol=0)


@pytest.mark.parametrize("relative", [False, True])
def test_serial_online_queue_and_latched_anchor(monkeypatch: pytest.MonkeyPatch, relative: bool) -> None:
    """第二次消费旧动作时保持首轮锚点，第三次重新推理才切换到本轮锚点。"""
    cfg = serial_config(relative)
    cfg.normalization_mapping = dict.fromkeys(("STATE", "ACTION", "VISUAL"), NormalizationMode.IDENTITY)
    pre, post = make_diffusion_pre_post_processors(cfg)
    policy = DiffusionPolicy(cfg).eval()
    calls: list[dict] = []

    def generate(batch: dict, noise: torch.Tensor | None = None) -> torch.Tensor:
        calls.append(batch)
        torch.testing.assert_close(batch[OBS_STATE][:, 0], batch[OBS_STATE][:, 1], atol=0, rtol=0)
        torch.testing.assert_close(batch[OBS_IMAGES][:, 0], batch[OBS_IMAGES][:, 1], atol=0, rtol=0)
        result = torch.zeros(1, cfg.n_action_steps, cfg.model_action_dim)
        if relative:
            result[..., 3] = result[..., 7] = 1
        result[..., -1] = 0.65
        return result

    monkeypatch.setattr(policy.diffusion, "generate_actions", generate)
    states = trajectory()
    for index in range(3):
        raw = {OBS_STATE: states[index], IMAGE: torch.full((3, 48, 80), float(index))}
        output = post(policy.select_action(pre(raw)))
        assert len(calls) == (1 if index < 2 else 2)
        if relative:
            torch.testing.assert_close(output[0, :6], states[0 if index < 2 else 2, :6], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(calls[-1][OBS_IMAGES][0, 0, 0, :, 20:40, :], torch.full((3, 20, 64), 2.0))
    # 完整动作块接口不能因为 select_action 留有队列而忽略本轮新输入。
    policy.predict_action_chunk(pre({OBS_STATE: states[4], IMAGE: torch.ones(3, 48, 80)}))
    assert len(calls) == 3
    torch.testing.assert_close(calls[-1][OBS_IMAGES][0, 0, 0, :, 20:40, :], torch.ones(3, 20, 64))
    policy.reset()
    pre.reset()
    policy.predict_action_chunk(pre({OBS_STATE: states[4], IMAGE: torch.ones(3, 48, 80)}))
    assert len(calls) == 4


@pytest.mark.parametrize("relative", [False, True])
def test_training_backward_and_export_conditioning(relative: bool) -> None:
    """真实训练梯度及固定导出编码器都应使用补边后的相同双帧窗口。"""
    cfg = serial_config(relative)
    pre, post = make_diffusion_pre_post_processors(cfg, statistics(cfg))
    policy = DiffusionPolicy(cfg)
    raw = {
        OBS_STATE: trajectory()[:2].unsqueeze(0),
        IMAGE: torch.rand(1, 2, 3, 48, 80),
        ACTION: trajectory()[:8].unsqueeze(0),
        "action_is_pad": torch.zeros(1, 8, dtype=torch.bool),
    }
    batch = pre(raw)
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in policy.parameters())
    policy.eval()
    with torch.inference_mode():
        native = policy.predict_action_chunk(batch)
        assert post(native).shape == (1, 2, 7)
        inputs = encoder_inputs(batch, cfg)
        module, samples, names, _ = make_export_modules(policy)["encoder"]
        assert samples[1].shape[-2:] == (64, 64)
        expected = policy.diffusion._prepare_global_conditioning(inputs)
        actual = module(*(inputs[key] for key in names))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_repeat_rejects_invalid_or_empty_history() -> None:
    with pytest.raises(ValueError, match="history"):
        repeat_latest_observation(torch.empty(1, 0, 7), 2, 2)
    with pytest.raises(ValueError, match="history"):
        repeat_latest_observation(torch.zeros(7), 2, 2)
