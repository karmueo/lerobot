"""Diffusion 的确定性图像处理与静止观测重复，可随策略检查点保存。"""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.transforms.image_resize import letterbox_image
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


def repeat_latest_observation(value: Tensor, n_obs_steps: int, single_ndim: int) -> Tensor:
    """接受带批次的单帧或历史窗口，只重复最后一帧，不改变批次维。"""
    if value.ndim == single_ndim:
        value = value.unsqueeze(1)
    elif value.ndim != single_ndim + 1:
        raise ValueError(f"Expected single frame or observation history, got {tuple(value.shape)}")
    if value.shape[1] == 0:
        raise ValueError("Observation history must not be empty")
    return value[:, -1:].expand(-1, n_obs_steps, *value.shape[2:]).clone()


@ProcessorStepRegistry.register("diffusion_observation_processor")
@dataclass
class DiffusionObservationProcessorStep(ObservationProcessorStep):
    """归一化前补黑边，并在相对位姿转换前将本轮当前观测重复到完整窗口。"""

    image_keys: list[str] = field(default_factory=list)
    resize_shape: tuple[int, int] | None = None
    serial_observation: bool = False
    n_obs_steps: int = 2

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = dict(observation)
        for key in self.image_keys:
            value = result.get(key)
            if not isinstance(value, Tensor):
                continue
            if self.serial_observation:
                value = repeat_latest_observation(value, self.n_obs_steps, 4)
            if self.resize_shape is not None:
                value = letterbox_image(value, self.resize_shape)
            result[key] = value
        if self.serial_observation:
            for key in (OBS_STATE, OBS_ENV_STATE):
                value = result.get(key)
                if isinstance(value, Tensor):
                    result[key] = repeat_latest_observation(value, self.n_obs_steps, 2)
        return result

    def get_config(self) -> dict[str, Any]:
        """保存重复语义与几何参数，推理恢复时不依赖训练数据集。"""
        return {
            "image_keys": self.image_keys,
            "resize_shape": self.resize_shape,
            "serial_observation": self.serial_observation,
            "n_obs_steps": self.n_obs_steps,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if self.resize_shape is not None:
            observations = features.get(PipelineFeatureType.OBSERVATION, {})
            for key in self.image_keys:
                if key in observations:
                    feature = observations[key]
                    observations[key] = PolicyFeature(feature.type, (feature.shape[0], *self.resize_shape))
        return features
