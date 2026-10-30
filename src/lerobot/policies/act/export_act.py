# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""ACT 固定 batch 推理图；归一化与动作反归一化由模型包运行器处理。"""

import copy

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class FixedPositionEmbedding(nn.Module):
    """固定图像大小下缓存原生二维位置编码，避免依赖 TensorRT 的 CumSum 实现。"""

    def __init__(self, position: Tensor):
        super().__init__()
        self.register_buffer("position", position)

    def forward(self, features: Tensor) -> Tensor:
        """位置编码仅依赖特征图尺寸，与图像内容无关。"""
        return self.position


class ACTExportModule(nn.Module):
    """复用原生推理网络，按配置顺序接收已归一化的状态与相机图像。"""

    def __init__(self, policy):
        super().__init__()
        # 浅复制模块容器以替换导出位置编码，不修改原始参考策略。
        self.model = copy.copy(policy.model)
        self.model._modules = policy.model._modules.copy()
        if policy.config.image_features:
            device = next(policy.parameters()).device
            shape = next(iter(policy.config.image_features.values())).shape
            with torch.inference_mode():
                feature_map = policy.model.backbone(torch.zeros(1, *shape, device=device))["feature_map"]
                position = policy.model.encoder_cam_feat_pos_embed(feature_map).detach().clone()
            self.model.encoder_cam_feat_pos_embed = FixedPositionEmbedding(position)
        self.image_keys = list(policy.config.image_features)
        self.has_env = bool(policy.config.env_state_feature)

    def forward(self, state: Tensor, *inputs: Tensor) -> Tensor:
        """输出完整动作块；eval 模式下 VAE 隐变量与原策略一致，固定为零。"""
        batch = {OBS_STATE: state}
        if self.image_keys:
            batch[OBS_IMAGES] = list(inputs[: len(self.image_keys)])
        if self.has_env:
            batch[OBS_ENV_STATE] = inputs[len(self.image_keys)]
        return self.model(batch)[0]


def act_inputs(batch: dict, config) -> dict[str, Tensor]:
    """将处理器输出映射到稳定的 ONNX 输入名，避免相机顺序发生变化。"""
    inputs = {"state": batch[OBS_STATE]}
    for index, key in enumerate(config.image_features):
        inputs[f"image_{index}"] = batch[key]
    if config.env_state_feature:
        inputs["environment_state"] = batch[OBS_ENV_STATE]
    return inputs


def make_export_modules(policy):
    """根据检查点特征构造固定 batch=1、FP32 的样例输入。"""
    config = policy.config
    if config.temporal_ensemble_coeff is not None:
        raise NotImplementedError("ACT export runtime does not implement temporal ensembling")
    if not config.robot_state_feature:
        raise ValueError("ACT export requires an observation.state feature")
    # 位置编码为固定图像大小预计算，多相机必须使用相同分辨率。
    if len({tuple(feature.shape) for feature in config.image_features.values()}) > 1:
        raise ValueError("ACT export requires identical image feature shapes")
    device = next(policy.parameters()).device
    inputs = {"state": torch.zeros(1, *config.robot_state_feature.shape, device=device)}
    for index, feature in enumerate(config.image_features.values()):
        inputs[f"image_{index}"] = torch.zeros(1, *feature.shape, device=device)
    if config.env_state_feature:
        inputs["environment_state"] = torch.zeros(1, *config.env_state_feature.shape, device=device)
    return {"policy": (ACTExportModule(policy).eval(), tuple(inputs.values()), list(inputs), "actions")}
