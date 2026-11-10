"""覆盖整段拒绝、base 系 TCP 几何和保留全部路点的统一限速。"""

from dataclasses import replace

import numpy as np
import pytest

from lerobot.deployment.rm75b.config import ExecutionConfig, SafetyConfig
from lerobot.deployment.rm75b.geometry import pose_to_transform, transform_to_pose
from lerobot.deployment.rm75b.trajectory import TrajectoryExecutor


class LinearArm:
    """可控的可逆关节模型，用于将轨迹保护与 IK 收敛误差分开测试。"""

    velocities = np.full(7, 0.2)

    def __init__(self, *, fail_at=None, flip=False, bulge=False):
        self.fail_at = fail_at
        self.flip = flip
        self.bulge = bulge
        self.seeds = []

    def forward(self, joints):
        transform = pose_to_transform(joints[:6])
        if self.bulge and 0.204 < joints[0] < 0.206:
            transform[0, 3] = 0.01
        return transform

    def within_limits(self, joints):
        return joints.shape == (7,) and bool((np.abs(joints) < 3).all())

    def solve_ik(self, target, seed):
        self.seeds.append(seed.copy())
        if len(self.seeds) == self.fail_at:
            return None
        return np.append(transform_to_pose(target), 1.0 if self.flip else seed[-1])


def setup_trajectory(**kwargs):
    """让回合轴相对于基座旋转，且 Link7 到 TCP 带平移。"""
    kin = LinearArm(**kwargs)
    initial = np.array([0.2, 0.0, 0.2, 0, 0, np.pi / 2, 0])
    tcp = pose_to_transform([0, 0, 0.1, 0, 0, 0])
    anchor = kin.forward(initial) @ tcp
    actions = np.zeros((8, 7))
    actions[:, 0] = np.arange(1, 9) * 0.012
    actions[:, 6] = np.linspace(0.9, 0.2, 8)
    executor = TrajectoryExecutor(kin, tcp, ExecutionConfig(), SafetyConfig())
    return executor, actions, anchor, initial


def test_timing_and_absolute_tcp_targets_preserve_all_waypoints():
    executor, actions, anchor, initial = setup_trajectory()
    plan = executor.plan(actions, anchor, initial, 1.0)
    assert plan.joints.shape == (9, 7)
    assert plan.tcp_targets.shape == (9, 4, 4)
    np.testing.assert_allclose(plan.joints[:, 0], 0.2)
    np.testing.assert_allclose(plan.joints[1:, 1], actions[:, 0], atol=1e-10)
    np.testing.assert_allclose(plan.tcp_targets[:, 2, 3], 0.3)
    np.testing.assert_allclose(plan.joints[:, 2], 0.2)
    np.testing.assert_allclose(executor.kinematics.seeds, plan.joints[:-1])
    assert plan.times[1] >= 0.012 / (0.2 * 0.25)
    np.testing.assert_allclose(np.diff(plan.times), plan.times[1])
    np.testing.assert_allclose(plan.times * 50, np.round(plan.times * 50))
    for index, t in enumerate(plan.times):
        joints, grip = executor.sample(plan, float(t))
        np.testing.assert_allclose(joints, plan.joints[index])
        assert grip == pytest.approx(plan.grippers[index])
    halfway, gripper = executor.sample(plan, float(plan.times[1] / 2))
    np.testing.assert_allclose(halfway, (initial + plan.joints[1]) / 2)
    assert gripper == pytest.approx(0.95)


def test_trajectory_rejects_late_ik_failure_and_redundant_joint_flip():
    executor, actions, anchor, initial = setup_trajectory(fail_at=8)
    with pytest.raises(ValueError, match="动作点 8 IK"):
        executor.plan(actions, anchor, initial, 1.0)
    executor, actions, anchor, initial = setup_trajectory(flip=True)
    with pytest.raises(ValueError, match="关节跳变"):
        executor.plan(actions, anchor, initial, 1.0)


def test_cartesian_bounds_and_first_waypoint_step():
    executor, actions, anchor, initial = setup_trajectory()
    actions[0, 0] = 0.025
    with pytest.raises(ValueError, match="动作点 1 超出 TCP 步长"):
        executor.plan(actions, anchor, initial, 1.0)
    actions[0, 0] = 1
    with pytest.raises(ValueError, match="超出基座系 TCP 工作空间"):
        executor.plan(actions, anchor, initial, 1.0)


def test_interpolated_fk_outside_workspace_is_rejected():
    executor, actions, anchor, initial = setup_trajectory(bulge=True)
    # 回合 y 轴对应基座 -x，令关节插值从 x=.20 穿过 x=.205。
    actions[:, :6] = 0
    actions[:, 1] = -0.01
    with pytest.raises(ValueError, match="插值段 1"):
        executor.plan(actions, anchor, initial, 1.0)


def test_nonfinite_interpolated_fk_rejects_entire_trajectory(monkeypatch):
    executor, actions, anchor, initial = setup_trajectory()
    forward = executor.kinematics.forward

    def invalid_forward(joints):
        pose = forward(joints)
        if np.any(joints != initial):
            pose[0, 3] = np.nan
        return pose

    monkeypatch.setattr(executor.kinematics, "forward", invalid_forward)
    with pytest.raises(ValueError, match="插值段 1"):
        executor.plan(actions, anchor, initial, 1.0)


@pytest.mark.parametrize(
    ("velocity_scale", "message"),
    [(1e-12, "预检上限"), (1e-320, "时长不是有限值"), (np.nextafter(0.0, 1.0), "速度必须为正且有限")],
)
def test_unsafe_retiming_rejected_before_interpolated_fk(velocity_scale, message, monkeypatch):
    executor, actions, anchor, initial = setup_trajectory()
    executor.execution = replace(executor.execution, velocity_scale=velocity_scale)
    forward = executor.kinematics.forward
    calls = 0

    def count_forward(joints):
        nonlocal calls
        calls += 1
        assert calls == 1, "无效时长不得进入插值 FK 循环"
        return forward(joints)

    monkeypatch.setattr(executor.kinematics, "forward", count_forward)
    with pytest.raises(ValueError, match=message):
        executor.plan(actions, anchor, initial, 1.0)
