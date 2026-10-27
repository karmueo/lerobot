#!/usr/bin/env python

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
"""SE(3) 相对位姿（UMI / FastUMI 风格）处理步骤及其在 diffusion 策略中的接入测试。"""

import math

import numpy as np
import pytest
import torch

from lerobot.configs import PipelineFeatureType
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.compute_stats import compute_relative_pose_stats
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor import (
    RELATIVE_POSE_ANCHOR_KEY,
    AbsolutePoseProcessorStep,
    RelativePoseProcessorStep,
    to_absolute_pose,
    to_relative_pose,
)
from lerobot.processor.converters import create_transition
from lerobot.processor.relative_pose_processor import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot.utils.rotation import Rotation

IDENTITY_ROT6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)


def _random_poses(generator: torch.Generator, *shape: int, extra: int = 1) -> torch.Tensor:
    """随机生成 ``[x, y, z, rx, ry, rz, extra...]`` 的原始布局位姿（轴角模长 < π）。"""
    position = torch.randn(*shape, 3, generator=generator, dtype=torch.float64)
    axis = torch.nn.functional.normalize(
        torch.randn(*shape, 3, generator=generator, dtype=torch.float64), dim=-1
    )
    angle = torch.rand(*shape, 1, generator=generator, dtype=torch.float64) * (math.pi - 1e-3)
    others = torch.rand(*shape, extra, generator=generator, dtype=torch.float64)
    return torch.cat([position, axis * angle, others], dim=-1)


def _homogeneous(pose: np.ndarray) -> np.ndarray:
    """原始布局单个位姿 -> 4x4 齐次矩阵（用仓库自带的 numpy Rotation 作为独立参考实现）。"""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    out[:3, 3] = pose[:3]
    return out


# ---------------------------------------------------------------------------
# 旋转工具
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.0, 1e-7, 1e-3, 0.5, 2.0, math.pi - 1e-4, math.pi])
def test_axis_angle_matrix_round_trip(angle):
    axis = torch.nn.functional.normalize(torch.tensor([[0.3, -0.5, 0.8]], dtype=torch.float64), dim=-1)
    rotvec = axis * angle
    matrix = axis_angle_to_matrix(rotvec)
    torch.testing.assert_close(matrix @ matrix.transpose(-1, -2), torch.eye(3, dtype=torch.float64)[None])
    reference = Rotation.from_rotvec(rotvec[0].numpy()).as_matrix()
    np.testing.assert_allclose(matrix[0].numpy(), reference, atol=1e-9)
    torch.testing.assert_close(axis_angle_to_matrix(matrix_to_axis_angle(matrix)), matrix, atol=1e-6, rtol=0)


def test_rotation_6d_uses_umi_first_two_rows_and_round_trips():
    generator = torch.Generator().manual_seed(0)
    matrix = axis_angle_to_matrix(_random_poses(generator, 16)[..., 3:6])
    rotation_6d = matrix_to_rotation_6d(matrix)
    torch.testing.assert_close(rotation_6d[..., :3], matrix[..., 0, :])
    torch.testing.assert_close(rotation_6d[..., 3:], matrix[..., 1, :])
    torch.testing.assert_close(rotation_6d_to_matrix(rotation_6d), matrix)
    # 非正交的 6D 输入也应解码为合法旋转
    noisy = rotation_6d_to_matrix(rotation_6d + 0.1)
    torch.testing.assert_close(
        noisy @ noisy.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand_as(noisy)
    )


# ---------------------------------------------------------------------------
# 相对 / 绝对位姿转换
# ---------------------------------------------------------------------------


def test_relative_pose_matches_umi_homogeneous_formula():
    generator = torch.Generator().manual_seed(1)
    reference = _random_poses(generator, 4)
    values = _random_poses(generator, 4, 5)
    relative = to_relative_pose(values, reference, [0], [0])
    assert relative.shape == (4, 5, 10)

    for b in range(4):
        base = _homogeneous(reference[b].numpy())
        for t in range(5):
            expected = np.linalg.inv(base) @ _homogeneous(values[b, t].numpy())
            np.testing.assert_allclose(relative[b, t, :3].numpy(), expected[:3, 3], atol=1e-9)
            np.testing.assert_allclose(relative[b, t, 3:9].numpy(), expected[:2, :3].reshape(6), atol=1e-9)
    # 位姿组以外的维度（夹爪）保持绝对值
    torch.testing.assert_close(relative[..., 9], values[..., 6])


def test_relative_pose_of_current_frame_is_identity():
    generator = torch.Generator().manual_seed(2)
    current = _random_poses(generator, 3)
    relative = to_relative_pose(current, current, [0], [0])
    torch.testing.assert_close(relative[:, :3], torch.zeros(3, 3, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(relative[:, 3:9], IDENTITY_ROT6D.expand(3, 6), atol=1e-12, rtol=0)


def test_bimanual_pose_groups_round_trip():
    generator = torch.Generator().manual_seed(3)
    left, right = _random_poses(generator, 2, 8), _random_poses(generator, 2, 8)
    values = torch.cat([left, right], dim=-1)  # [pose6, gripper, pose6, gripper]
    reference = values[:, -1]
    relative = to_relative_pose(values, reference, [0, 7], [0, 7])
    assert relative.shape == (2, 8, 20)
    restored = to_absolute_pose(relative, reference, [0, 7], [0, 7])
    # 轴角可能被规范化到 ‖r‖ <= π，比较旋转矩阵和其余维度
    for start in (0, 7):
        torch.testing.assert_close(restored[..., start : start + 3], values[..., start : start + 3])
        torch.testing.assert_close(
            axis_angle_to_matrix(restored[..., start + 3 : start + 6]),
            axis_angle_to_matrix(values[..., start + 3 : start + 6]),
            atol=1e-6,
            rtol=0,
        )
    torch.testing.assert_close(restored[..., [6, 13]], values[..., [6, 13]])


# ---------------------------------------------------------------------------
# 处理步骤
# ---------------------------------------------------------------------------


def test_step_training_batch_uses_last_observation_frame_as_reference():
    generator = torch.Generator().manual_seed(4)
    state = _random_poses(generator, 2, 2).float()
    action = _random_poses(generator, 2, 8).float()
    step = RelativePoseProcessorStep(enabled=True, state_pose_starts=[0], n_obs_steps=2)
    out = step(create_transition(observation={OBS_STATE: state}, action=action))

    rel_state = out[TransitionKey.OBSERVATION][OBS_STATE]
    rel_action = out[TransitionKey.ACTION]
    assert rel_state.shape == (2, 2, 10) and rel_action.shape == (2, 8, 10)
    # 当前帧（最后一帧）为精确单位位姿（不留浮点误差）
    assert torch.equal(rel_state[:, -1, :3], torch.zeros(2, 3))
    assert torch.equal(rel_state[:, -1, 3:9], IDENTITY_ROT6D.float().expand(2, 6))
    # 任何输入形状都会把锚点句柄交给策略（训练时由策略忽略）
    assert out[TransitionKey.OBSERVATION][RELATIVE_POSE_ANCHOR_KEY] is step.anchor
    torch.testing.assert_close(rel_action, to_relative_pose(action, state[:, -1], [0], [0]))


def test_step_inference_builds_history_and_absolute_uses_latched_anchor():
    generator = torch.Generator().manual_seed(5)
    frames = _random_poses(generator, 3, 1).float()  # 3 个时间步，B=1
    relative_step = RelativePoseProcessorStep(enabled=True, n_obs_steps=2)
    absolute_step = AbsolutePoseProcessorStep(enabled=True, relative_step=relative_step)

    out0 = relative_step(create_transition(observation={OBS_STATE: frames[0]}))
    history0 = out0[TransitionKey.OBSERVATION][OBS_STATE]
    assert history0.shape == (1, 2, 10)
    # 首帧复制填满历史，两帧都为单位位姿
    torch.testing.assert_close(history0[:, 0], history0[:, 1])
    anchor = out0[TransitionKey.OBSERVATION][RELATIVE_POSE_ANCHOR_KEY]
    anchor.latch()  # 模拟策略在第 0 步生成动作块

    out1 = relative_step(create_transition(observation={OBS_STATE: frames[1]}))
    history1 = out1[TransitionKey.OBSERVATION][OBS_STATE]
    torch.testing.assert_close(history1, to_relative_pose(frames[:2].transpose(0, 1), frames[1], [0], [0]))

    # 单步动作来自第 0 步生成的动作块，必须用第 0 步的参考位姿还原
    relative_action = to_relative_pose(frames[2], frames[0], [0], [0])
    restored = absolute_step(create_transition(action=relative_action))[TransitionKey.ACTION]
    torch.testing.assert_close(restored, frames[2], atol=1e-5, rtol=0)

    # 动作块 (B, T, D) 由本次观测生成，使用最新参考位姿
    chunk = to_relative_pose(frames[2][:, None], frames[1], [0], [0])
    restored_chunk = absolute_step(create_transition(action=chunk))[TransitionKey.ACTION]
    torch.testing.assert_close(restored_chunk[:, 0], frames[2], atol=1e-5, rtol=0)

    relative_step.reset()
    assert relative_step.anchor.current is None and relative_step.anchor.chunk is None


def test_step_transform_features_expands_pose_groups():
    step = RelativePoseProcessorStep(enabled=True, state_pose_starts=[0, 7])
    features = {
        PipelineFeatureType.OBSERVATION: {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        PipelineFeatureType.ACTION: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    }
    out = step.transform_features(features)
    assert out[PipelineFeatureType.OBSERVATION][OBS_STATE].shape == (20,)
    assert out[PipelineFeatureType.ACTION][ACTION].shape == (20,)
    back = AbsolutePoseProcessorStep(enabled=True, state_pose_starts=[0, 7]).transform_features(out)
    assert back[PipelineFeatureType.ACTION][ACTION].shape == (14,)


# ---------------------------------------------------------------------------
# 相对空间统计量
# ---------------------------------------------------------------------------


def test_compute_relative_pose_stats_matches_training_samples():
    datasets = pytest.importorskip("datasets")
    generator = torch.Generator().manual_seed(6)
    episode_lengths = [5, 4]
    states = _random_poses(generator, sum(episode_lengths)).float()
    actions = _random_poses(generator, sum(episode_lengths)).float()
    episode_index = np.repeat(np.arange(len(episode_lengths)), episode_lengths)
    hf_dataset = datasets.Dataset.from_dict(
        {OBS_STATE: states.numpy(), ACTION: actions.numpy(), "episode_index": episode_index}
    )
    obs_deltas, act_deltas = [-1, 0], [-1, 0, 1, 2]
    stats = compute_relative_pose_stats(hf_dataset, obs_deltas, act_deltas, state_pose_starts=[0])

    # 逐样本按 LeRobotDataset 的边界填充规则手动复现
    rel_states, rel_actions = [], []
    starts = np.concatenate([[0], np.cumsum(episode_lengths)])
    for lo, hi in zip(starts[:-1], starts[1:], strict=True):
        for row in range(lo, hi):
            obs_rows = [min(max(row + d, lo), hi - 1) for d in obs_deltas]
            act_rows = [min(max(row + d, lo), hi - 1) for d in act_deltas]
            current = states[row : row + 1].double()
            rel_states.append(to_relative_pose(states[obs_rows].double()[None], current, [0], [0])[0])
            rel_actions.append(to_relative_pose(actions[act_rows].double()[None], current, [0], [0])[0])
    rel_states = torch.cat(rel_states).numpy()
    rel_actions = torch.cat(rel_actions).numpy()

    assert stats[OBS_STATE]["mean"].shape == (10,) and stats[ACTION]["mean"].shape == (10,)
    np.testing.assert_allclose(stats[OBS_STATE]["min"], rel_states.min(0), atol=1e-6)
    np.testing.assert_allclose(stats[OBS_STATE]["max"], rel_states.max(0), atol=1e-6)
    np.testing.assert_allclose(stats[ACTION]["mean"], rel_actions.mean(0), atol=1e-6)
    np.testing.assert_allclose(stats[ACTION]["max"], rel_actions.max(0), atol=1e-6)


def test_single_observation_frame_stats_do_not_amplify_float_error():
    """n_obs_steps=1 时 state 位姿维度恒为单位位姿，归一化后不能被浮点误差放大。"""
    datasets = pytest.importorskip("datasets")
    generator = torch.Generator().manual_seed(8)
    states = _random_poses(generator, 200).float()
    hf_dataset = datasets.Dataset.from_dict(
        {OBS_STATE: states.numpy(), ACTION: states.numpy(), "episode_index": np.zeros(200, dtype=np.int64)}
    )
    stats = compute_relative_pose_stats(hf_dataset, [0], list(range(16)), state_pose_starts=[0])[OBS_STATE]
    assert np.all(stats["max"] - stats["min"] >= 1e-4 - 1e-12)
    assert np.all(stats["std"] >= 1e-4)

    step = RelativePoseProcessorStep(enabled=True, n_obs_steps=1)
    rel = step(create_transition(observation={OBS_STATE: states[:, None]}))[TransitionKey.OBSERVATION][
        OBS_STATE
    ]
    low = torch.from_numpy(stats["min"]).float()
    high = torch.from_numpy(stats["max"]).float()
    normalized = 2 * (rel - low) / (high - low) - 1
    assert normalized[..., :9].abs().max() < 1e-3


def test_floor_stat_ranges_only_widens_narrow_dims():
    from lerobot.datasets.compute_stats import _floor_stat_ranges

    stats = {
        "min": np.array([0.0, 1.0, -2.0]),
        "max": np.array([0.0, 1.0 + 1e-12, 3.0]),
        "std": np.array([0.0, 1e-13, 0.5]),
        "q01": np.array([0.0, 1.0, -1.0]),
        "q99": np.array([0.0, 1.0, 2.0]),
    }
    out = _floor_stat_ranges(stats, 1e-4)
    np.testing.assert_allclose(out["max"] - out["min"], [1e-4, 1e-4, 5.0])
    np.testing.assert_allclose((out["max"] + out["min"]) / 2, [0.0, 1.0, 0.5], atol=1e-11)
    np.testing.assert_allclose(out["std"], [1e-4, 1e-4, 0.5])
    np.testing.assert_allclose(out["q99"] - out["q01"], [1e-4, 1e-4, 3.0])


# ---------------------------------------------------------------------------
# diffusion 端到端
# ---------------------------------------------------------------------------


def _relative_pose_diffusion_config(**overrides) -> DiffusionConfig:
    kwargs = {
        "use_relative_pose": True,
        "relative_pose_state_starts": [0],
        "n_obs_steps": 2,
        "horizon": 8,
        "n_action_steps": 3,
        "down_dims": (16, 32),
        "diffusion_step_embed_dim": 16,
        "num_train_timesteps": 4,
        "num_inference_steps": 2,
        "device": "cpu",
    }
    config = DiffusionConfig(**{**kwargs, **overrides})
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    return config


def _identity_stats(state_dim: int, action_dim: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {"min": -torch.ones(state_dim), "max": torch.ones(state_dim)},
        OBS_ENV_STATE: {"min": -torch.ones(2), "max": torch.ones(2)},
        ACTION: {"min": -torch.ones(action_dim), "max": torch.ones(action_dim)},
    }


def test_diffusion_config_model_dims_and_validation():
    config = _relative_pose_diffusion_config()
    config.validate_features()
    assert config.model_state_dim == 10 and config.model_action_dim == 10

    bad = _relative_pose_diffusion_config(relative_pose_state_starts=[3])
    with pytest.raises(ValueError, match="needs 6 dims"):
        bad.validate_features()
    with pytest.raises(ValueError, match="as many pose groups"):
        _relative_pose_diffusion_config(relative_pose_action_starts=[0, 7])


def test_diffusion_relative_pose_train_and_rollout(tmp_path):
    pytest.importorskip("diffusers")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(0)
    config = _relative_pose_diffusion_config()
    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, _identity_stats(10, 10))
    policy = DiffusionPolicy(config)

    # 训练：数据集给出 (B, n_obs_steps, 7) 的 state 与 (B, horizon, 7) 的 action
    generator = torch.Generator().manual_seed(7)
    batch = {
        OBS_STATE: _random_poses(generator, 2, 2).float(),
        OBS_ENV_STATE: torch.randn(2, 2, 2),
        ACTION: _random_poses(generator, 2, 8).float(),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }
    processed = preprocessor(batch)
    assert processed[OBS_STATE].shape == (2, 2, 10) and processed[ACTION].shape == (2, 8, 10)
    loss, _ = policy.forward(processed)
    loss.backward()

    # 推理：逐步输入单帧观测，记录每步的参考位姿与模型输出（相对量）
    policy.eval()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    relative_step = next(s for s in preprocessor.steps if isinstance(s, RelativePoseProcessorStep))
    frames = _random_poses(generator, 5, 1).float()
    for t in range(5):
        observation = {OBS_STATE: frames[t], OBS_ENV_STATE: torch.randn(1, 2)}
        processed_obs = preprocessor(observation)
        assert processed_obs[OBS_STATE].shape == (1, 2, 10)
        model_action = policy.select_action(processed_obs)
        assert model_action.shape == (1, 10)
        chunk_start = (t // config.n_action_steps) * config.n_action_steps
        # 队列中的动作必须以生成该动作块时的观测帧为参考还原
        expected = to_absolute_pose(model_action.clone(), frames[chunk_start], [0], [0])
        action = postprocessor(model_action)
        assert action.shape == (1, 7)
        torch.testing.assert_close(action, expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(relative_step.anchor.chunk, frames[chunk_start])

    # 保存后重新加载：后处理器的绝对位姿步骤应重新连接到预处理器
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    loaded_pre, loaded_post = make_pre_post_processors(config, pretrained_path=str(tmp_path))
    loaded_relative = next(s for s in loaded_pre.steps if isinstance(s, RelativePoseProcessorStep))
    loaded_absolute = next(s for s in loaded_post.steps if isinstance(s, AbsolutePoseProcessorStep))
    assert loaded_absolute.relative_step is loaded_relative
    assert loaded_relative.n_obs_steps == 2 and loaded_relative.state_pose_starts == [0]


def test_diffusion_relative_pose_rollout_with_stacked_history_latches_anchor():
    """推理时调用方直接传入 (B, T, D) 完整历史，队列中的动作仍需用生成时的参考位姿还原。"""
    pytest.importorskip("diffusers")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    torch.manual_seed(0)
    config = _relative_pose_diffusion_config()
    preprocessor, postprocessor = make_diffusion_pre_post_processors(config, _identity_stats(10, 10))
    policy = DiffusionPolicy(config).eval()
    relative_step = next(s for s in preprocessor.steps if isinstance(s, RelativePoseProcessorStep))

    generator = torch.Generator().manual_seed(9)
    frames = _random_poses(generator, 6, 1).float()
    for t in range(1, 6):
        observation = {
            OBS_STATE: torch.stack([frames[t - 1], frames[t]], dim=1),
            OBS_ENV_STATE: torch.randn(1, 2),
        }
        model_action = policy.select_action(preprocessor(observation))
        chunk_start = 1 + ((t - 1) // config.n_action_steps) * config.n_action_steps
        expected = to_absolute_pose(model_action.clone(), frames[chunk_start], [0], [0])
        torch.testing.assert_close(postprocessor(model_action), expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(relative_step.anchor.chunk, frames[chunk_start])
