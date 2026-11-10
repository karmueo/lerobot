"""基础验收的 mock 设备状态、运动门槛及停止路径。"""

import os
import signal
import sys
import time
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lerobot.deployment.rm75b import acceptance, cli, planning_worker
from lerobot.deployment.rm75b.acceptance import BasicAcceptance, acceptance_planner, axis_plan, reverse_plan
from lerobot.deployment.rm75b.config import DeploymentAssets, DeploymentConfig
from lerobot.deployment.rm75b.geometry import pose_to_transform, transform_to_pose
from lerobot.deployment.rm75b.kinematics import ArmKinematics
from lerobot.deployment.rm75b.planning_worker import PlanningWorker, precheck_motion
from lerobot.deployment.rm75b.trajectory import ActionWorkspaceError, TrajectoryExecutor
from lerobot.deployment.rm75b.types import ArmState, FloatArray, GripperState, JointTrajectory


class Arm:
    """同周期接受关节目标，按需保留静止反馈以模拟跟踪失效。"""

    def __init__(self, joints: FloatArray) -> None:
        self.joints = joints.copy()
        self.now = 0.0
        self.stale = False
        self.track = True
        self.calls = []

    def read_state(self) -> ArmState:
        return ArmState(self.now - int(self.stale), self.joints.copy(), np.zeros(7))

    def home(self, joints: FloatArray, speed: int) -> None:
        self.calls.append(("home", speed))
        self.joints = joints.copy()

    def send_joint_target(self, joints: FloatArray) -> None:
        self.calls.append(("joint", joints.copy()))
        if self.track:
            self.joints = joints.copy()

    def stop(self) -> None:
        self.calls.append(("stop",))


class Grip:
    """记录目标和 50 Hz 更新，仅更新后才模拟得到新开度。"""

    def __init__(self, arm: Arm) -> None:
        self.arm = arm
        self.goal = self.open = 1.0
        self.calls = []

    def read_state(self) -> GripperState:
        return GripperState(self.arm.now, self.open, self.open)

    def set_target(self, openness: float) -> None:
        self.calls.append(("target", openness))
        self.goal = openness

    def tick(self, now: float) -> None:
        self.calls.append(("tick", now))
        self.open = self.goal


class LinearKin:
    """独立可逆的七轴几何模型，方便区分预检与实机跟踪。"""

    velocities = np.full(7, 3.0)

    def forward(self, joints: FloatArray) -> FloatArray:
        return pose_to_transform(joints[:6])

    def solve_ik(self, target: FloatArray, seed: FloatArray) -> FloatArray:
        return np.append(transform_to_pose(target), seed[-1])

    def within_limits(self, joints: FloatArray) -> bool:
        return joints.shape == (7,) and bool(np.all(np.abs(joints) < 2))


class Events:
    """保留相同事件结构，不写磁盘。"""

    def __init__(self) -> None:
        self.events = []

    def write(self, event: str, **values: Any) -> None:
        self.events.append((event, values))


class InlinePlanner:
    """测试内同步计算纯轨迹，保留生产环境的非阻塞接口。"""

    def __init__(self, planner: TrajectoryExecutor, initial_joints: FloatArray) -> None:
        self.planner = planner
        self.initial_joints = initial_joints
        self.pid = 1

    def start(self) -> None:
        pass

    def submit(
        self, step: str, joints: FloatArray, openness: float, distance_mm: int
    ) -> Future[JointTrajectory | None]:
        future: Future[JointTrajectory | None] = Future()
        try:
            future.set_result(
                precheck_motion(self.planner, self.initial_joints, step, joints, openness, distance_mm)
            )
        except Exception as error:
            future.set_exception(error)
        return future

    def check_alive(self) -> None:
        pass

    def close(self) -> None:
        pass


type MockHardware = tuple[BasicAcceptance, Arm, Grip, Events]


@pytest.fixture
def hardware() -> MockHardware:
    cfg = DeploymentConfig()
    joints = np.array([0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    assets = DeploymentAssets(np.eye(4), joints, {}, {}, {})
    arm, events = Arm(joints), Events()
    grip = Grip(arm)
    planner = acceptance_planner(cfg, assets, LinearKin())
    session = BasicAcceptance(
        cfg,
        assets,
        planner,
        arm,
        grip,
        events,
        planning_worker=InlinePlanner(planner, joints),
        clock=lambda: arm.now,
    )
    return session, arm, grip, events


def advance(session: BasicAcceptance, arm: Arm, seconds: float) -> None:
    """用单调时钟推进50Hz循环。"""
    for _ in range(round(seconds * 50)):
        arm.now += 0.02
        session.tick(arm.now)


def test_manual_sequence_uses_one_baseline_and_delayed_return(hardware: MockHardware) -> None:
    session, arm, grip, events = hardware
    assert session.command("x", arm.now) == "请先完成回位"
    assert not arm.calls and not grip.calls
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    assert arm.calls == [("home", 5)]
    assert not grip.calls
    baseline = session.baseline.copy()
    session.command("close", arm.now)
    advance(session, arm, 0.5)
    assert grip.open == 0.0 and session.phase == "ready"
    assert "张开" in session.command("x", arm.now)
    session.command("open", arm.now)
    advance(session, arm, 0.5)
    assert grip.open == 1.0
    for index, axis in enumerate("xyz"):
        session.command(axis, arm.now)
        advance(session, arm, 2.8)
        assert session.phase == "offset"
        delta = np.zeros(7)
        delta[index] = 0.005
        np.testing.assert_allclose(arm.joints - baseline, delta, atol=1e-12)
        assert "停留" in session.command("back", arm.now)
        advance(session, arm, 2.0)
        session.command("back", arm.now)
        advance(session, arm, 2.8)
        assert session.phase == "ready"
        np.testing.assert_allclose(arm.joints, baseline, atol=1e-12)
    assert len([e for e, _ in events.events if e == "baseline"]) == 1


def test_path_timing_pose_and_reverse(hardware: MockHardware) -> None:
    session, arm, _, _ = hardware
    plan = axis_plan(session.planner, arm.joints, 1.0, "x")
    back = reverse_plan(plan)
    assert plan.duration >= 2.0 and back.duration == plan.duration
    np.testing.assert_allclose(plan.tcp_targets[-1, :3, 3] - plan.tcp_targets[0, :3, 3], [0.005, 0, 0])
    np.testing.assert_array_equal(plan.joints[0], back.joints[-1])
    np.testing.assert_allclose(plan.tcp_targets[:, :3, :3], np.tile(np.eye(3), (9, 1, 1)), atol=1e-12)


def test_50mm_plan_preserves_all_segments_and_frozen_axis(hardware: MockHardware) -> None:
    session, arm, _, _ = hardware
    plan = axis_plan(session.planner, arm.joints, 1.0, "y", 50)
    back = reverse_plan(plan)
    assert len(plan.times) == 81 and plan.duration >= 20 and back.duration == plan.duration
    assert np.all(np.diff(plan.times) > 0)
    np.testing.assert_allclose(plan.tcp_targets[:, 1, 3], np.linspace(0.0, 0.05, 81), atol=1e-12)
    np.testing.assert_allclose(plan.tcp_targets[:, 0, 3], 0.3, atol=1e-12)
    np.testing.assert_allclose(plan.tcp_targets[:, 2, 3], 0.3, atol=1e-12)
    np.testing.assert_array_equal(back.joints[-1], arm.joints)
    with pytest.raises(ValueError, match="只允许"):
        axis_plan(session.planner, arm.joints, 1.0, "y", 100)


def test_reverse_preserves_nonuniform_waypoint_intervals(hardware: MockHardware) -> None:
    session, arm, _, _ = hardware
    original = axis_plan(session.planner, arm.joints, 1.0, "x")
    times = np.array([0, 0.2, 0.5, 1, 1.3, 1.7, 2, 2.3, 3.0], dtype=np.float64)
    forward = replace(original, times=times)
    backward = reverse_plan(forward)
    np.testing.assert_allclose(np.diff(backward.times), np.diff(times)[::-1], atol=1e-12)
    np.testing.assert_array_equal(backward.joints, forward.joints[::-1])


def test_50mm_late_workspace_failure_sends_no_partial_trajectory(hardware: MockHardware) -> None:
    session, arm, _, _ = hardware
    session.distance_mm = 50
    session.planner._workspace_max[0] = 0.349
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    session.command("x", arm.now)
    with pytest.raises(ValueError, match=r"第 10 个 5 mm 段.*工作空间"):
        advance(session, arm, 0.02)
    assert arm.calls == [("home", 5)]


def test_speed_20_scales_time_and_home_without_dropping_waypoints(hardware: MockHardware) -> None:
    session, arm, grip, events = hardware
    fast = acceptance_planner(session.cfg, session.assets, LinearKin(), speed_percent=20)
    normal_plan = axis_plan(session.planner, arm.joints, 1.0, "y", 50)
    fast_plan = axis_plan(fast, arm.joints, 1.0, "y", 50)
    assert session.planner.execution.velocity_scale == 0.05
    assert fast.execution.velocity_scale == 0.20
    assert fast.execution.fps == 16 and fast.execution.control_hz == 50
    assert normal_plan.duration == pytest.approx(20.8) and fast_plan.duration == pytest.approx(6.4)
    np.testing.assert_array_equal(fast_plan.joints, normal_plan.joints)
    fast_session = BasicAcceptance(
        session.cfg,
        session.assets,
        fast,
        arm,
        grip,
        events,
        speed_percent=20,
        planning_worker=InlinePlanner(fast, session.assets.initial_joints),
        clock=lambda: arm.now,
    )
    fast_session.command("home", arm.now)
    advance(fast_session, arm, 0.7)
    assert arm.calls == [("home", 20)]


def test_deployment_speed_cap_still_determines_slowest_time(hardware: MockHardware) -> None:
    session, arm, _, _ = hardware
    slow_cap = 0.0005
    config = replace(session.cfg, execution=replace(session.cfg.execution, velocity_scale=slow_cap))
    normal = acceptance_planner(config, session.assets, LinearKin(), speed_percent=5)
    fast = acceptance_planner(config, session.assets, LinearKin(), speed_percent=20)
    assert fast.execution.velocity_scale == normal.execution.velocity_scale == slow_cap
    normal_plan = axis_plan(normal, arm.joints, 1.0, "x", 50)
    fast_plan = axis_plan(fast, arm.joints, 1.0, "x", 50)
    assert fast_plan.duration == normal_plan.duration


def test_speed_percent_rejects_arbitrary_values(hardware: MockHardware) -> None:
    session, _, _, _ = hardware
    with pytest.raises(ValueError, match="只允许 5% 或 20%"):
        acceptance_planner(session.cfg, session.assets, LinearKin(), speed_percent=12)


def test_stale_feedback_and_stop_prevent_further_commands(hardware: MockHardware) -> None:
    session, arm, grip, _ = hardware
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    session.command("x", arm.now)
    advance(session, arm, 0.08)
    arm.stale = True
    count = len(arm.calls)
    with pytest.raises(RuntimeError, match="反馈超时"):
        advance(session, arm, 0.02)
    assert len(arm.calls) == count
    session.halt()
    advance(session, arm, 0.08)
    assert arm.calls[-1] == ("stop",)
    assert len(arm.calls) == count + 1
    assert not grip.calls


def test_stationary_arm_that_does_not_reach_tcp_target_fails(hardware: MockHardware) -> None:
    session, arm, _, _ = hardware
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    arm.track = False
    session.command("x", arm.now)
    with pytest.raises(RuntimeError, match="末点稳定超时"):
        advance(session, arm, 8.0)


@pytest.mark.parametrize("block_event", ["feedback", "joint_command_prepared"])
def test_in_tick_blocking_never_dispatches_overdue_target(
    hardware: MockHardware, monkeypatch: pytest.MonkeyPatch, block_event: str
) -> None:
    session, arm, grip, events = hardware
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    session.command("x", arm.now)
    advance(session, arm, 0.02)
    assert session.phase == "outbound"
    original = events.write

    def blocking_write(event: str, **values: Any) -> None:
        if event == block_event:
            arm.now += 0.11
        original(event, **values)

    monkeypatch.setattr(events, "write", blocking_write)
    with pytest.raises(RuntimeError, match="发送截止时刻"):
        advance(session, arm, 0.02)
    assert arm.calls == [("home", 5)] and not grip.calls
    assert not any(event == "joint_command" for event, _ in events.events)


def test_acceptance_rejects_unsafe_sample(hardware: MockHardware, monkeypatch: pytest.MonkeyPatch) -> None:
    """即使轴向计划通过预检，也要校验实际采样的下发点."""
    session, arm, _, _ = hardware
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    session.command("x", arm.now)
    advance(session, arm, 0.02)
    assert session.phase == "outbound"
    outside = arm.joints.copy()
    outside[0] = 0.01
    monkeypatch.setattr(session.planner, "sample", lambda *_: (outside, 1.0))
    with pytest.raises(RuntimeError, match="TCP 超出工作空间"):
        advance(session, arm, 0.02)
    assert arm.calls == [("home", 5)]


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_signal_stops_acceptance_before_closing_devices(
    hardware: MockHardware, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, signum: signal.Signals
) -> None:
    """验收在下发非阻塞回位后收到退出信号，先停臂再释放资源."""
    from lerobot.deployment.rm75b import arm as arm_module, gripper as gripper_module

    session, arm, grip, _ = hardware
    worker = InlinePlanner(session.planner, session.assets.initial_joints)
    previous = signal.getsignal(signum)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "conflicting_processes", lambda: [])
    monkeypatch.setattr(acceptance, "build_manifest", lambda *_: {"mode": "basic-acceptance"})
    monkeypatch.setattr(acceptance, "PlanningWorker", lambda *_: worker)
    monkeypatch.setattr(arm_module, "ArmDriver", lambda *args, **kwargs: arm)
    monkeypatch.setattr(gripper_module, "GripperDriver", lambda *args, **kwargs: grip)
    monkeypatch.setattr(arm, "connect", lambda: None, raising=False)
    monkeypatch.setattr(grip, "connect", lambda: None, raising=False)
    monkeypatch.setattr(arm, "close", lambda: arm.calls.append(("arm_close",)), raising=False)
    monkeypatch.setattr(grip, "close", lambda: arm.calls.append(("gripper_close",)), raising=False)
    monkeypatch.setattr(worker, "close", lambda: arm.calls.append(("worker_close",)))

    def interrupt(running: BasicAcceptance) -> None:
        running._clock = lambda: arm.now
        running.command("home", arm.now)
        advance(running, arm, 0.02)
        assert running.phase == "homing"
        assert signal.getsignal(signum) != previous
        signal.raise_signal(signum)

    monkeypatch.setattr(acceptance, "_interactive", interrupt)
    cfg = replace(session.cfg, logging=replace(session.cfg.logging, lock_path=tmp_path / "acceptance.lock"))
    assert acceptance._execute(cfg, session.assets, session.planner, tmp_path, 5, 5) == 128 + signum
    assert arm.calls == [("home", 5), ("stop",), ("gripper_close",), ("arm_close",), ("worker_close",)]
    assert signal.getsignal(signum) == previous


def test_in_tick_blocking_prevents_gripper_dispatch(
    hardware: MockHardware, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, arm, grip, events = hardware
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    session.command("close", arm.now)
    original = events.write

    def blocking_write(event: str, **values: Any) -> None:
        if event == "feedback":
            arm.now += 0.11
        original(event, **values)

    monkeypatch.setattr(events, "write", blocking_write)
    with pytest.raises(RuntimeError, match="发送截止时刻"):
        advance(session, arm, 0.02)
    assert grip.calls == [("target", 0.0)]
    assert not any(event == "gripper_tick" for event, _ in events.events)


def test_stop_discards_in_flight_plan(hardware: MockHardware, monkeypatch: pytest.MonkeyPatch) -> None:
    session, arm, _, _ = hardware
    future = Future()
    assert future.set_running_or_notify_cancel()
    monkeypatch.setattr(session._planning_worker, "submit", lambda *_: future)
    session.command("home", arm.now)
    assert session.phase == "planning" and not arm.calls
    session.command("stop", arm.now)
    future.set_result(None)
    advance(session, arm, 0.1)
    assert not arm.calls


def test_offset_drift_after_tick_feedback_does_not_send_correction(
    hardware: MockHardware, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, arm, _, events = hardware
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    session.command("x", arm.now)
    advance(session, arm, 2.8)
    assert session.phase == "offset"
    original = events.write

    def drift_before_dispatch(event: str, **values: Any) -> None:
        if event == "joint_command_prepared":
            arm.joints[0] += 0.001
        original(event, **values)

    monkeypatch.setattr(events, "write", drift_before_dispatch)
    count = len(arm.calls)
    with pytest.raises(RuntimeError, match="偏离末点"):
        advance(session, arm, 0.02)
    assert len(arm.calls) == count


def test_back_requires_stationary_target_within_tolerance(hardware: MockHardware) -> None:
    session, arm, _, _ = hardware
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    session.command("z", arm.now)
    advance(session, arm, 4.8)
    assert session.phase == "offset"
    arm.joints[2] -= 0.001
    count = len(arm.calls)
    with pytest.raises(RuntimeError, match="偏离末点"):
        session.command("back", arm.now)
    assert session.phase == "offset" and len(arm.calls) == count


def test_plan_timeout_keeps_home_unapplied(hardware: MockHardware, monkeypatch: pytest.MonkeyPatch) -> None:
    session, arm, _, _ = hardware
    future: Future[JointTrajectory | None] = Future()
    monkeypatch.setattr(session._planning_worker, "submit", lambda *_: future)
    session.command("home", arm.now)
    with pytest.raises(RuntimeError, match="预检超时"):
        advance(session, arm, 5.1)
    session.halt()
    assert future.cancelled() and not arm.calls


def test_worker_death_after_plan_stops_new_arm_targets(
    hardware: MockHardware, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, arm, _, _ = hardware
    session.command("home", arm.now)
    advance(session, arm, 0.7)
    session.command("x", arm.now)
    advance(session, arm, 0.1)
    calls = len(arm.calls)

    def dead() -> None:
        raise RuntimeError("规划子进程已退出")

    monkeypatch.setattr(session._planning_worker, "check_alive", dead)
    with pytest.raises(RuntimeError, match="子进程已退出"):
        advance(session, arm, 0.02)
    assert len(arm.calls) == calls
    session.halt()
    assert arm.calls[-1] == ("stop",)


@pytest.fixture
def spawn_params(tmp_path: Path) -> tuple[Path, FloatArray, DeploymentConfig]:
    """构造只用于纯计算的可达平面关节链，不依赖本机设备资产。"""
    joints = []
    for index in range(1, 8):
        parent = "base_link" if index == 1 else f"Link{index - 1}"
        origin = "0.1 0 0.1" if index == 1 else "0.1 0 0"
        joints.append(
            f'<joint name="joint{index}" type="revolute">'
            f'<parent link="{parent}"/><child link="Link{index}"/>'
            f'<origin xyz="{origin}"/><axis xyz="0 0 1"/>'
            '<limit lower="-3" upper="3" velocity="3"/></joint>'
        )
    urdf = tmp_path / "arm.urdf"
    urdf.write_text('<robot name="test">' + "".join(joints) + "</robot>")
    return urdf, np.array([0.5, -1, 0.5, 0.0, 0.0, 0.0, 0.0]), DeploymentConfig()


def test_spawn_worker_warmup_repeated_result_and_env_restoration(
    spawn_params: tuple[Path, FloatArray, DeploymentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    urdf, joints, cfg = spawn_params
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    worker = PlanningWorker(urdf, np.eye(4), joints, cfg.execution, cfg.safety)
    try:
        worker.start()
        assert worker.pid is not None and worker.pid != os.getpid()
        assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
        assert worker.submit("home", joints, 1.0, 5).result(timeout=5) is None
        start = joints.copy()
        start[0] -= 0.05
        home = worker.submit_home(start).result(timeout=5)
        assert home is not None
        np.testing.assert_array_equal(home.joints, [start, joints])
        assert home.duration >= 0.05 / (3.0 * 0.10)
        for _ in range(2):
            result = worker.submit("y", joints, 1.0, 50).result(timeout=5)
            assert result is not None and len(result.times) == 81
        kin = ArmKinematics.from_urdf(urdf)
        anchor = kin.forward(joints)
        actions = np.zeros((cfg.execution.n_action_steps, 7), dtype=np.float64)
        actions[:, 6] = 1.0
        # 第八点越界须跨进程保留错误类别，后续请求仍能正常规划。
        actions[-1, 0] = 2.0
        with pytest.raises(ActionWorkspaceError, match="动作点 8 超出基座系 TCP 工作空间"):
            worker.submit_policy(actions, anchor, joints, 1.0).result(timeout=5)
        actions[-1, 0] = 0.0
        result = worker.submit_policy(actions, anchor, joints, 1.0).result(timeout=5)
        assert result is not None and len(result.times) == 9
        np.testing.assert_allclose(result.joints, np.tile(joints, (9, 1)))
        with pytest.raises(ValueError, match="完整的有限"):
            worker.submit_policy(actions[:-1], anchor, joints, 1.0)
    finally:
        worker.close()
    assert worker.pid is None


def test_spawn_startup_timeout_releases_child(
    spawn_params: tuple[Path, FloatArray, DeploymentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    urdf, joints, cfg = spawn_params
    monkeypatch.setattr(planning_worker, "_START_TIMEOUT_S", 0)
    worker = PlanningWorker(urdf, np.eye(4), joints, cfg.execution, cfg.safety)
    with pytest.raises(RuntimeError, match="就绪超时"):
        worker.start()
    assert worker.pid is None


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="需要 POSIX 子进程暂停信号")
def test_spawn_worker_busy_crash_and_close(spawn_params: tuple[Path, FloatArray, DeploymentConfig]) -> None:
    urdf, joints, cfg = spawn_params
    worker = PlanningWorker(urdf, np.eye(4), joints, cfg.execution, cfg.safety)
    try:
        worker.start()
        pid = worker.pid
        assert pid is not None
        os.kill(pid, signal.SIGSTOP)
        future = worker.submit("home", joints, 1.0, 5)
        with pytest.raises(RuntimeError, match="未完成请求"):
            worker.submit("home", joints, 1.0, 5)
        os.kill(pid, signal.SIGKILL)
        with pytest.raises(RuntimeError, match="响应中断"):
            future.result(timeout=5)
        with pytest.raises(RuntimeError, match="响应中断|已退出"):
            worker.check_alive()
    finally:
        worker.close()
    assert worker.pid is None


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="需要 POSIX 子进程暂停信号")
def test_spawn_worker_bounded_cleanup_with_stalled_child(
    spawn_params: tuple[Path, FloatArray, DeploymentConfig],
) -> None:
    urdf, joints, cfg = spawn_params
    worker = PlanningWorker(urdf, np.eye(4), joints, cfg.execution, cfg.safety)
    worker.start()
    pid = worker.pid
    assert pid is not None
    try:
        os.kill(pid, signal.SIGSTOP)
        future = worker.submit("home", joints, 1.0, 5)
        started = time.monotonic()
    finally:
        worker.close()
    assert time.monotonic() - started < 3
    assert worker.pid is None
    with pytest.raises(RuntimeError, match="会话已关闭"):
        future.result(timeout=0)
