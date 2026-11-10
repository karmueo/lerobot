"""验证旋转向量数值边界和回合起点坐标方向."""

import math

import numpy as np
import pytest

from lerobot.deployment.rm75b.geometry import (
    invert,
    pose_to_transform,
    rotation_distance,
    transform_to_pose,
    validate_transform,
)


@pytest.mark.parametrize("angle", [0.0, 1e-10, 0.25, math.pi - 1e-8, math.pi])
def test_rotvec_round_trip(angle: float) -> None:
    """接近零和 π 时通过矩阵比对确认旋转保持不变."""
    axis = np.array([-2.0, 3.0, 5.0])
    pose = np.r_[0.1, -0.2, 0.3, axis * angle / np.linalg.norm(axis)]
    transform = pose_to_transform(pose)
    validate_transform(transform)
    np.testing.assert_allclose(pose_to_transform(transform_to_pose(transform)), transform, atol=1e-8)
    assert rotation_distance(np.eye(3), transform[:3, :3]) == pytest.approx(angle, abs=1e-8)


def test_episode_anchor_and_tool_offset_have_distinct_order() -> None:
    """将绕 z 轴旋转 90° 的起点相对 x 位移转换为基座 y 位移."""
    start = pose_to_transform([0.3, 0.2, 0.4, 0, 0, math.pi / 2])
    target = pose_to_transform([0.05, 0, 0, 0, 0, 0])
    tool = pose_to_transform([0.10, 0, 0, 0, math.pi / 4, 0])
    base_tcp = start @ target
    np.testing.assert_allclose(base_tcp[:3, 3], [0.3, 0.25, 0.4], atol=1e-12)
    flange = base_tcp @ invert(tool)
    np.testing.assert_allclose(invert(start) @ flange @ tool, target, atol=1e-12)


def test_invalid_transform_reflection_is_rejected() -> None:
    """正交但行列式为负的反射不是机械臂旋转."""
    reflection = np.eye(4)
    reflection[2, 2] = -1
    with pytest.raises(ValueError, match="无效的旋转"):
        validate_transform(reflection)
