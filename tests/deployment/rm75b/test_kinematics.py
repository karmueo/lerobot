"""校验 URDF 父子链、FK 轴与阻尼最小二乘 IK。"""

from pathlib import Path

import numpy as np
import pytest

from lerobot.deployment.rm75b.geometry import rotation_distance
from lerobot.deployment.rm75b.kinematics import ArmKinematics


def write_urdf(path: Path) -> Path:
    """构造已知几何解的七轴平面链，用于独立核对 FK 和限制。"""
    joints = []
    for index in range(1, 8):
        parent = "base_link" if index == 1 else f"Link{index - 1}"
        joints.append(
            f'<joint name="joint{index}" type="revolute">'
            f'<parent link="{parent}"/><child link="Link{index}"/>'
            '<origin xyz="0.1 0 0"/><axis xyz="0 0 2"/>'
            f'<limit lower="-3" upper="3" velocity="{index}"/></joint>'
        )
    path.write_text('<robot name="test">' + "".join(joints) + "</robot>")
    return path


def test_fk_reads_urdf_origin_axis_and_velocity(tmp_path):
    chain = ArmKinematics.from_urdf(write_urdf(tmp_path / "arm.urdf"))
    angles = np.zeros(7)
    angles[0] = np.pi / 2
    pose = chain.forward(angles)
    np.testing.assert_allclose(pose[:3, 3], [0.1, 0.6, 0.0], atol=1e-12)
    np.testing.assert_allclose(pose[:3, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_array_equal(chain.velocities, np.arange(1, 8))
    with pytest.raises(ValueError, match="长度 7"):
        chain.forward(np.zeros(6))


def test_ik_reaches_pose_and_rejects_unreachable_target(tmp_path):
    chain = ArmKinematics.from_urdf(write_urdf(tmp_path / "arm.urdf"))
    joints = np.array([0.2, -0.3, 0.1, 0.0, 0.0, 0.0, 0.0])
    target_joints = joints.copy()
    target_joints[2] += 0.05
    goal = chain.forward(target_joints)
    solution = chain.solve_ik(goal, joints)
    assert solution is not None and chain.within_limits(solution)
    reached = chain.forward(solution)
    assert np.linalg.norm(reached[:3, 3] - goal[:3, 3]) < 5e-4
    assert rotation_distance(reached[:3, :3], goal[:3, :3]) < np.deg2rad(0.1)
    goal[2, 3] = 2
    assert chain.solve_ik(goal, joints) is None


@pytest.mark.parametrize(
    ("before", "after", "error"),
    [
        ('<parent link="Link3"/>', '<parent link="Link2"/>', "父子链"),
        ('<axis xyz="0 0 2"/>', '<axis xyz="0 0 0"/>', "关节 joint1 的轴"),
        ('velocity="1"', 'velocity="0"', "速度上限"),
    ],
)
def test_invalid_urdf_is_rejected(tmp_path, before, after, error):
    path = write_urdf(tmp_path / "arm.urdf")
    path.write_text(path.read_text().replace(before, after, 1))
    with pytest.raises(ValueError, match=error):
        ArmKinematics.from_urdf(path)
