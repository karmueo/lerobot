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
"""Diffusion 固定形状导出适配：保持模型内部的相对位姿维度与相机顺序."""

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .modeling_diffusion import DiffusionModel, DiffusionPolicy


class FixedRgbEncoder(nn.Module):
    """用固定形状张量运算复现 torchvision 的抗锯齿 resize 与中心裁剪."""

    def __init__(self, rgb: nn.Module, cfg):
        super().__init__()
        self.backbone, self.pool, self.out, self.relu = rgb.backbone, rgb.pool, rgb.out, rgb.relu
        height, width = next(iter(cfg.image_features.values())).shape[-2:]
        resized = cfg.resize_shape or (height, width)
        self.resize_needed = tuple(resized) != (height, width)
        if self.resize_needed:
            # 对基向量应用原生插值，得到同一线性算子的两组权重；避免 ONNX 不支持 AA 算子。
            vertical = torch.nn.functional.interpolate(
                torch.eye(height)[None, None],
                size=(resized[0], height),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0, 0]
            horizontal = torch.nn.functional.interpolate(
                torch.eye(width)[None, None],
                size=(width, resized[1]),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0, 0]
            device = next(rgb.parameters()).device
            self.register_buffer("vertical", vertical.to(device))
            self.register_buffer("horizontal", horizontal.to(device))
        crop = cfg.crop_shape or resized
        # torchvision 的 center_crop 对更大输出先补零，再按 Python round 取中心。
        pad_h, pad_w = max(crop[0] - resized[0], 0), max(crop[1] - resized[1], 0)
        self.padding = (pad_w // 2, (pad_w + 1) // 2, pad_h // 2, (pad_h + 1) // 2)
        self.top = round((max(resized[0], crop[0]) - crop[0]) / 2)
        self.left = round((max(resized[1], crop[1]) - crop[1]) / 2)
        self.height, self.width = crop

    def forward(self, images: Tensor) -> Tensor:
        """固定裁剪坐标来自配置，避免追踪时对 Tensor 执行 Python round."""
        if self.resize_needed:
            images = self.vertical @ images @ self.horizontal
        if any(self.padding):
            images = torch.nn.functional.pad(images, self.padding)
        images = images[..., self.top : self.top + self.height, self.left : self.left + self.width]
        return self.relu(self.out(self.pool(self.backbone(images)).flatten(start_dim=1)))


class DiffusionEncoder(nn.Module):
    """将预处理后的历史观测编码为 conditioning，不包含归一化和位姿变换."""

    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.n_obs_steps = model.config.n_obs_steps
        self.keys = [OBS_STATE]
        if model.config.image_features:
            self.keys.append(OBS_IMAGES)
            self.n_cameras = len(model.config.image_features)
            encoders = (
                list(model.rgb_encoder)
                if model.config.use_separate_rgb_encoder_per_camera
                else [model.rgb_encoder] * self.n_cameras
            )
            self.rgb_encoders = nn.ModuleList([FixedRgbEncoder(rgb, model.config) for rgb in encoders])
        if model.config.env_state_feature:
            self.keys.append(OBS_ENV_STATE)

    def forward(self, *inputs: Tensor) -> Tensor:
        """输入顺序由 keys 确定，导出后写入 manifest."""
        batch = dict(zip(self.keys, inputs, strict=True))
        features = [batch[OBS_STATE]]
        if OBS_IMAGES in batch:
            images = batch[OBS_IMAGES]
            features.append(
                torch.cat(
                    [
                        encoder(images[:, :, i].flatten(0, 1)).reshape(1, self.n_obs_steps, -1)
                        for i, encoder in enumerate(self.rgb_encoders)
                    ],
                    dim=-1,
                )
            )
        if OBS_ENV_STATE in batch:
            features.append(batch[OBS_ENV_STATE])
        return torch.cat(features, dim=-1).flatten(start_dim=1)


class DiffusionDenoiser(nn.Module):
    """只导出单次网络预测，DDPM/DDIM 更新与随机数保留在外部."""

    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.unet = getattr(model.unet, "_orig_mod", model.unet)

    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        """返回 scheduler 所需预测值，而不是完成去噪后的动作."""
        return self.unet(sample, timestep, global_cond=global_cond)


def make_export_modules(policy: DiffusionPolicy):
    """构造网络和 FP32 样例；模型维度与原始状态维度可能不同."""
    cfg = policy.config
    device = next(policy.parameters()).device
    encoder = DiffusionEncoder(policy.diffusion).eval()
    inputs = [torch.zeros(1, cfg.n_obs_steps, cfg.model_state_dim, device=device)]
    if cfg.image_features:
        shape = next(iter(cfg.image_features.values())).shape
        inputs.append(torch.zeros(1, cfg.n_obs_steps, len(cfg.image_features), *shape, device=device))
    if cfg.env_state_feature:
        inputs.append(torch.zeros(1, cfg.n_obs_steps, *cfg.env_state_feature.shape, device=device))
    with torch.no_grad():
        cond = encoder(*inputs)
    denoiser_inputs = (
        torch.zeros(1, cfg.horizon, cfg.model_action_dim, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
        cond,
    )
    return {
        "encoder": (encoder, tuple(inputs), encoder.keys, "global_cond"),
        "denoiser": (
            DiffusionDenoiser(policy.diffusion).eval(),
            denoiser_inputs,
            ["sample", "timestep", "global_cond"],
            "prediction",
        ),
    }
