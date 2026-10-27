# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""UMI / FastUMI 风格的 SE(3) 相对位姿表示。

数据集中的末端位姿以 ``[x, y, z, rx, ry, rz]``（位置 + 轴角）存储。本模块把观测历史和动作块
都表达为相对"当前帧"位姿的齐次变换 ``inv(T_cur) @ T``，再编码为 ``[x, y, z] + rot6d``
（旋转矩阵前两行，与 UMI 的 ``mat_to_pose10d`` 一致）。位姿组以外的维度（如夹爪）保持绝对值。

每个位姿组在原始布局中占 6 维，在模型布局中占 9 维，其余维度按原顺序保留。
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry

# 原始布局中一个位姿组的宽度（位置 3 + 轴角 3）
RAW_POSE_DIM = 6
# 模型布局中一个位姿组的宽度（位置 3 + rot6d 6）
MODEL_POSE_DIM = 9
# 推理时预处理器写入观测字典的锚点句柄键名，策略生成新动作块时通过它锁定参考位姿
RELATIVE_POSE_ANCHOR_KEY = "observation.relative_pose_anchor"


# ---------------------------------------------------------------------------
# 旋转工具
# ---------------------------------------------------------------------------


def _skew(vector: Tensor) -> Tensor:
    """向量 (..., 3) 的反对称矩阵 (..., 3, 3)。"""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """轴角 (..., 3) 转旋转矩阵 (..., 3, 3)，采用 Rodrigues 公式。

    直接使用未归一化的反对称矩阵 ``K = [r]x``：``R = I + A K + B K^2``，其中
    ``A = sin(θ)/θ``、``B = (1 - cos θ)/θ²``。θ 接近 0 时用泰勒展开，避免除零。
    """
    angle_sq = (axis_angle * axis_angle).sum(-1, keepdim=True)
    angle = angle_sq.sqrt()
    small = angle < 1e-4
    safe_angle = torch.where(small, torch.ones_like(angle), angle)
    coef_a = torch.where(small, 1.0 - angle_sq / 6.0, torch.sin(safe_angle) / safe_angle)
    coef_b = torch.where(small, 0.5 - angle_sq / 24.0, (1.0 - torch.cos(safe_angle)) / safe_angle**2)
    skew = _skew(axis_angle)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(skew)
    return eye + coef_a[..., None] * skew + coef_b[..., None] * (skew @ skew)


def _sqrt_positive_part(value: Tensor) -> Tensor:
    """sqrt(max(0, x))，且在 0 处梯度为 0。"""
    result = torch.zeros_like(value)
    positive = value > 0
    result[positive] = torch.sqrt(value[positive])
    return result


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """旋转矩阵 (..., 3, 3) 转四元数 (..., 4)，顺序为 wxyz，且保证 w >= 0。

    从四个候选解中选取分母最大的一个，在旋转角接近 π 时依旧数值稳定。
    """
    batch_shape = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.reshape(*batch_shape, 9).unbind(-1)
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    best = F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5
    quaternion = candidates[best].reshape(*batch_shape, 4)
    # 四元数 q 与 -q 表示同一旋转，统一取 w >= 0，使转出的轴角满足 ‖r‖ <= π
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def quaternion_to_axis_angle(quaternion: Tensor) -> Tensor:
    """四元数 (..., 4，wxyz) 转轴角 (..., 3)。"""
    vector = quaternion[..., 1:]
    sin_half = vector.norm(dim=-1, keepdim=True)
    half_angle = torch.atan2(sin_half, quaternion[..., :1])
    angle = 2.0 * half_angle
    small = angle.abs() < 1e-6
    safe_angle = torch.where(small, torch.ones_like(angle), angle)
    # rotvec = vector * angle / sin(angle / 2)；小角度下 sin(θ/2)/θ ≈ 1/2 - θ²/48
    sin_half_over_angle = torch.where(small, 0.5 - angle**2 / 48.0, torch.sin(half_angle) / safe_angle)
    return vector / sin_half_over_angle


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """旋转矩阵 (..., 3, 3) 转规范轴角 (..., 3)，‖r‖ <= π。"""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """旋转矩阵 (..., 3, 3) 转 rot6d (..., 6)：取前两行，与 UMI 的 ``mat_to_rot6d`` 一致。"""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """rot6d (..., 6) 转旋转矩阵 (..., 3, 3)，用 Gram-Schmidt 正交化，结果按行堆叠。

    模型输出的 6 维不一定正交，正交化保证解码结果是合法旋转。
    """
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# ---------------------------------------------------------------------------
# 位姿布局与相对/绝对转换
# ---------------------------------------------------------------------------


def validate_pose_starts(starts: Sequence[int], raw_dim: int, name: str = "pose_starts") -> None:
    """检查位姿组起始下标：升序、不重叠，且每组 6 维都落在 ``raw_dim`` 之内。"""
    previous_end = 0
    for start in starts:
        if start < previous_end:
            raise ValueError(
                f"`{name}` must be ascending and non-overlapping 6-wide groups, got {list(starts)}"
            )
        if start + RAW_POSE_DIM > raw_dim:
            raise ValueError(
                f"`{name}` group starting at {start} needs {RAW_POSE_DIM} dims but the feature has {raw_dim}"
            )
        previous_end = start + RAW_POSE_DIM


def relative_pose_model_dim(raw_dim: int, num_groups: int) -> int:
    """原始维度对应的模型维度：每个位姿组从 6 维扩展为 9 维。"""
    return raw_dim + (MODEL_POSE_DIM - RAW_POSE_DIM) * num_groups


def _pose_from_raw(raw: Tensor, start: int) -> tuple[Tensor, Tensor]:
    """从原始布局取出某个位姿组，返回 (位置 (..., 3), 旋转矩阵 (..., 3, 3))。"""
    position = raw[..., start : start + 3]
    rotation = axis_angle_to_matrix(raw[..., start + 3 : start + RAW_POSE_DIM])
    return position, rotation


def _broadcast_reference(reference: Tensor, values: Tensor) -> Tensor:
    """参考位姿为 (B, D)，待转换值可能带时间维 (B, T, D)，此时在时间维上广播。"""
    if values.ndim == reference.ndim + 1:
        return reference.unsqueeze(-2)
    return reference


def to_relative_pose(
    values: Tensor,
    reference: Tensor,
    value_starts: Sequence[int],
    reference_starts: Sequence[int],
) -> Tensor:
    """把原始布局的位姿转换为相对参考位姿的模型布局。

    Args:
        values: (B, D_raw) 或 (B, T, D_raw)，原始布局（位置 + 轴角）。
        reference: (B, D_ref)，参考（当前帧）状态。
        value_starts: ``values`` 中各位姿组的起始下标。
        reference_starts: ``reference`` 中与之一一对应的位姿组起始下标。

    Returns:
        (B, D_model) 或 (B, T, D_model)。位姿组变为 ``[p_rel, rot6d(R_rel)]``，其中
        ``R_rel = R_ref^T R``、``p_rel = R_ref^T (p - p_ref)``，即 ``inv(T_ref) @ T``；其余维度原样保留。
    """
    if len(value_starts) != len(reference_starts):
        raise ValueError("value_starts and reference_starts must contain the same number of pose groups")
    reference = _broadcast_reference(reference.to(device=values.device, dtype=values.dtype), values)
    segments: list[Tensor] = []
    cursor = 0
    for value_start, reference_start in zip(value_starts, reference_starts, strict=True):
        segments.append(values[..., cursor:value_start])
        position, rotation = _pose_from_raw(values, value_start)
        ref_position, ref_rotation = _pose_from_raw(reference, reference_start)
        ref_rotation_t = ref_rotation.transpose(-1, -2)
        rel_rotation = ref_rotation_t @ rotation
        rel_position = (ref_rotation_t @ (position - ref_position).unsqueeze(-1)).squeeze(-1)
        segments.append(rel_position)
        segments.append(matrix_to_rotation_6d(rel_rotation))
        cursor = value_start + RAW_POSE_DIM
    segments.append(values[..., cursor:])
    return torch.cat(segments, dim=-1)


def to_absolute_pose(
    values: Tensor,
    reference: Tensor,
    value_starts: Sequence[int],
    reference_starts: Sequence[int],
) -> Tensor:
    """``to_relative_pose`` 的逆变换：把模型布局的相对位姿还原成原始布局的绝对位姿。

    Args:
        values: (B, D_model) 或 (B, T, D_model)。
        reference: (B, D_ref)，生成这些相对量时所用的参考状态（原始布局）。
        value_starts: 位姿组在 **原始** 布局中的起始下标。
        reference_starts: ``reference`` 中对应的位姿组起始下标。

    Returns:
        (B, D_raw) 或 (B, T, D_raw)，位姿组为 ``T_ref @ T_rel`` 的位置 + 规范轴角。
    """
    if len(value_starts) != len(reference_starts):
        raise ValueError("value_starts and reference_starts must contain the same number of pose groups")
    reference = _broadcast_reference(reference.to(device=values.device, dtype=values.dtype), values)
    segments: list[Tensor] = []
    model_cursor = 0
    for group_index, (value_start, reference_start) in enumerate(
        zip(value_starts, reference_starts, strict=True)
    ):
        # 原始布局中前面每个位姿组比模型布局少 3 维，据此换算模型布局中的起始位置
        model_start = value_start + (MODEL_POSE_DIM - RAW_POSE_DIM) * group_index
        segments.append(values[..., model_cursor:model_start])
        rel_position = values[..., model_start : model_start + 3]
        rel_rotation = rotation_6d_to_matrix(values[..., model_start + 3 : model_start + MODEL_POSE_DIM])
        ref_position, ref_rotation = _pose_from_raw(reference, reference_start)
        rotation = ref_rotation @ rel_rotation
        position = (ref_rotation @ rel_position.unsqueeze(-1)).squeeze(-1) + ref_position
        segments.append(position)
        segments.append(matrix_to_axis_angle(rotation))
        model_cursor = model_start + MODEL_POSE_DIM
    segments.append(values[..., model_cursor:])
    return torch.cat(segments, dim=-1)


def set_frame_to_identity(relative: Tensor, frame_index: int, value_starts: Sequence[int]) -> Tensor:
    """把模型布局历史 (..., T, D_model) 中第 ``frame_index`` 帧的位姿组写成精确的单位位姿。

    当前帧相对自身理论上恒为单位变换，但 ``R^T R`` 的浮点误差会留下 1e-7（float32）量级的
    抖动。若统计量中该维度近乎常量（如 ``n_obs_steps=1``），归一化会把这点误差放大到 1e8 以上。
    统一写成精确值后，训练、推理与统计量在这些维度上完全一致。
    """
    out = relative.clone()
    frame = out[..., frame_index, :]
    for group_index, value_start in enumerate(value_starts):
        model_start = value_start + (MODEL_POSE_DIM - RAW_POSE_DIM) * group_index
        frame[..., model_start : model_start + 3] = 0.0
        frame[..., model_start + 3 : model_start + MODEL_POSE_DIM] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=out.dtype, device=out.device
        )
    return out


# ---------------------------------------------------------------------------
# 处理步骤
# ---------------------------------------------------------------------------


class RelativePoseAnchor:
    """预处理器与后处理器之间共享的参考位姿句柄。

    - ``current``：最近一次预处理时的当前帧原始状态 (B, D)。
    - ``chunk``：策略最近一次生成动作块时锁定的参考状态。动作块会被逐步执行多个时间步，
      队列中后续动作必须用生成时的参考位姿还原，而不是用最新观测。
    """

    def __init__(self) -> None:
        self.current: Tensor | None = None
        self.chunk: Tensor | None = None

    def latch(self) -> None:
        """策略生成新动作块时调用：把当前参考位姿锁定为该动作块的参考。"""
        self.chunk = self.current

    def reset(self) -> None:
        self.current = None
        self.chunk = None


@ProcessorStepRegistry.register("relative_pose_processor")
@dataclass
class RelativePoseProcessorStep(ProcessorStep):
    """把 ``observation.state`` 历史和 ``action`` 块转换为相对当前帧的 SE(3) 位姿（rot6d 表示）。

    必须放在归一化之前，使归一化统计量描述的是相对空间中的分布。

    - 训练/离线：state 为 (B, T_obs, D)，当前帧取最后一帧 ``state[:, -1]``
      （diffusion 的 ``observation_delta_indices`` 为 ``[1 - n_obs_steps, ..., 0]``）。
    - 在线推理：state 为 (B, D)。本步骤自行缓存最近 ``n_obs_steps`` 帧原始状态（首帧复制填满，
      与策略内部队列的行为一致），输出 (B, n_obs_steps, D_model) 的完整相对历史。
    - 当前帧的位姿组被写成精确单位位姿，避免浮点误差在近常量维度上被归一化放大。
    - 观测中总是写入 ``RELATIVE_POSE_ANCHOR_KEY`` 句柄，供策略在生成新动作块时锁定参考位姿。

    Attributes:
        enabled: 是否启用转换。
        state_pose_starts: ``observation.state`` 中各位姿组（6 维）的起始下标。
        action_pose_starts: ``action`` 中各位姿组的起始下标，``None`` 表示与 state 相同；
            第 i 个动作位姿组相对于第 i 个状态位姿组计算。
        n_obs_steps: 推理时缓存的观测帧数，应与策略的 ``n_obs_steps`` 一致。
    """

    enabled: bool = False
    state_pose_starts: list[int] = field(default_factory=lambda: [0])
    action_pose_starts: list[int] | None = None
    n_obs_steps: int = 1
    _history: deque = field(default_factory=deque, init=False, repr=False)
    _anchor: RelativePoseAnchor = field(default_factory=RelativePoseAnchor, init=False, repr=False)

    @property
    def resolved_action_pose_starts(self) -> list[int]:
        return list(self.state_pose_starts if self.action_pose_starts is None else self.action_pose_starts)

    @property
    def anchor(self) -> RelativePoseAnchor:
        return self._anchor

    def _stack_history(self, state: Tensor) -> Tensor:
        """推理时维护原始状态历史，返回 (B, n_obs_steps, D)。批大小变化时重新填充。"""
        if self._history.maxlen != self.n_obs_steps:
            self._history = deque(maxlen=self.n_obs_steps)
        if len(self._history) > 0 and self._history[-1].shape != state.shape:
            self._history.clear()
        if len(self._history) == 0:
            self._history.extend([state] * self.n_obs_steps)
        else:
            self._history.append(state)
        return torch.stack(list(self._history), dim=1)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None:
            return transition

        new_transition = transition.copy()
        new_observation = dict(observation)
        if state.ndim == 3:
            # 训练批，或推理时调用方直接给出完整历史：最后一帧为当前帧
            current = state[:, -1]
            history = state
        elif state.ndim == 2:
            # 在线推理单帧输入：由本步骤拼出历史
            current = state
            history = self._stack_history(state)
        else:
            raise ValueError(f"Expected observation.state with 2 or 3 dims, got shape {tuple(state.shape)}")

        self._anchor.current = current
        # 无论输入形状如何都把锚点句柄交给策略：只要经过 select_action 消费动作队列，
        # 就必须在生成新动作块时锁定参考位姿。训练时策略的 forward 会忽略它。
        new_observation[RELATIVE_POSE_ANCHOR_KEY] = self._anchor
        relative_history = to_relative_pose(history, current, self.state_pose_starts, self.state_pose_starts)
        new_observation[OBS_STATE] = set_frame_to_identity(relative_history, -1, self.state_pose_starts)
        new_transition[TransitionKey.OBSERVATION] = new_observation

        action = new_transition.get(TransitionKey.ACTION)
        if isinstance(action, Tensor):
            new_transition[TransitionKey.ACTION] = to_relative_pose(
                action, current, self.resolved_action_pose_starts, self.state_pose_starts
            )
        return new_transition

    def reset(self) -> None:
        self._history.clear()
        self._anchor.reset()

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state_pose_starts": list(self.state_pose_starts),
            "action_pose_starts": None if self.action_pose_starts is None else list(self.action_pose_starts),
            "n_obs_steps": self.n_obs_steps,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if not self.enabled:
            return features
        _expand_feature(features, PipelineFeatureType.OBSERVATION, OBS_STATE, len(self.state_pose_starts))
        _expand_feature(features, PipelineFeatureType.ACTION, ACTION, len(self.resolved_action_pose_starts))
        return features


@ProcessorStepRegistry.register("absolute_pose_processor")
@dataclass
class AbsolutePoseProcessorStep(ProcessorStep):
    """把模型输出的相对位姿动作（rot6d）还原为原始布局的绝对位姿（位置 + 轴角）。

    放在反归一化之后。参考位姿的选择：
    - 单步动作 (B, D)：来自策略动作队列，用该动作块生成时锁定的 ``anchor.chunk``；
      若策略未锁定（例如未经过 ``select_action``），退回 ``anchor.current``。
    - 动作块 (B, T, D)：刚由当前观测生成，用 ``anchor.current``。

    Attributes:
        enabled: 是否启用转换。
        state_pose_starts / action_pose_starts: 与配对的 ``RelativePoseProcessorStep`` 相同。
        relative_step: 配对的预处理步骤（不序列化，加载后由工厂重新连接）。
    """

    enabled: bool = False
    state_pose_starts: list[int] = field(default_factory=lambda: [0])
    action_pose_starts: list[int] | None = None
    relative_step: RelativePoseProcessorStep | None = field(default=None, repr=False)

    @property
    def resolved_action_pose_starts(self) -> list[int]:
        return list(self.state_pose_starts if self.action_pose_starts is None else self.action_pose_starts)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, Tensor):
            return transition
        if self.relative_step is None:
            raise RuntimeError(
                "AbsolutePoseProcessorStep requires a paired RelativePoseProcessorStep, but relative_step is None."
            )

        anchor = self.relative_step.anchor
        reference = anchor.chunk if action.ndim == 2 and anchor.chunk is not None else anchor.current
        if reference is None:
            raise RuntimeError(
                "AbsolutePoseProcessorStep has no reference pose. Run the preprocessor before the postprocessor."
            )

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = to_absolute_pose(
            action, reference, self.resolved_action_pose_starts, self.state_pose_starts
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state_pose_starts": list(self.state_pose_starts),
            "action_pose_starts": None if self.action_pose_starts is None else list(self.action_pose_starts),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if not self.enabled:
            return features
        _expand_feature(features, PipelineFeatureType.ACTION, ACTION, -len(self.resolved_action_pose_starts))
        return features


def _expand_feature(
    features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    feature_type: PipelineFeatureType,
    key: str,
    num_groups: int,
) -> None:
    """按位姿组数调整特征宽度；``num_groups`` 为负表示从模型布局收缩回原始布局。"""
    feature = features.get(feature_type, {}).get(key)
    if feature is None:
        return
    new_dim = relative_pose_model_dim(feature.shape[-1], num_groups)
    features[feature_type][key] = PolicyFeature(type=feature.type, shape=(*feature.shape[:-1], new_dim))


def reconnect_relative_pose_steps(
    preprocessor_steps: Sequence[Any], postprocessor_steps: Sequence[Any]
) -> None:
    """从磁盘加载处理器后，把 ``AbsolutePoseProcessorStep`` 重新连接到配对的预处理步骤。"""
    relative_step = next((s for s in preprocessor_steps if isinstance(s, RelativePoseProcessorStep)), None)
    if relative_step is None:
        return
    for step in postprocessor_steps:
        if isinstance(step, AbsolutePoseProcessorStep) and step.relative_step is None:
            step.relative_step = relative_step
