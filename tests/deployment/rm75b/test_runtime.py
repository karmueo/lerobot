"""以虚拟时钟测试动作授权、执行时序和故障停止."""

import json
from concurrent.futures import Executor, Future
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lerobot.deployment.rm75b.config import DeploymentAssets, DeploymentConfig
from lerobot.deployment.rm75b.observations import ObservationAssembler
from lerobot.deployment.rm75b.runtime import SerialDeployment, State
from lerobot.deployment.rm75b.telemetry import RunLog
from lerobot.deployment.rm75b.trajectory import ActionWorkspaceError, TrajectoryExecutor
from lerobot.deployment.rm75b.types import (
    ActionChunk,
    ArmState,
    CameraFrame,
    FloatArray,
    GripperState,
    JointTrajectory,
    ObservationWindow,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0


class _Arm:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.joints = np.zeros(7)
        self.commands: list[FloatArray] = []
        self.homes = self.stops = self.closes = 0
        self.stamp: float | None = None
        self.errors: tuple[int, ...] = ()
        self.drop = False
        self.send_failure = False
        self.stop_failure = False

    def connect(self) -> None:
        pass

    def read_state(self) -> ArmState | None:
        if self.drop:
            return None
        return ArmState(
            self.clock.now if self.stamp is None else self.stamp,
            self.joints.copy(),
            np.zeros(7),
            self.errors,
        )

    def send_joint_target(self, joints: FloatArray) -> None:
        if self.send_failure:
            raise RuntimeError("测试 SDK 发送失败")
        self.commands.append(joints.copy())
        self.joints = joints.copy()

    def home(self, joints: FloatArray, speed: int = 10) -> None:
        self.homes += 1
        self.joints = joints.copy()

    def stop(self) -> None:
        self.stops += 1
        if self.stop_failure:
            raise RuntimeError("测试停臂失败")

    def close(self) -> None:
        self.closes += 1


class _Gripper:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.openness = 1.0
        self.track = True
        self.targets: list[float] = []
        self.ticks = self.holds = self.closes = 0

    def connect(self) -> None:
        pass

    def read_state(self) -> GripperState:
        return GripperState(self.clock.now, self.openness, self.openness)

    def set_target(self, openness: float) -> None:
        self.targets.append(openness)
        if self.track:
            self.openness = openness

    def at_target(self, feedback: GripperState, openness: float) -> bool:
        return abs(feedback.openness - openness) <= 0.05

    def tick(self, now: float) -> None:
        self.ticks += 1

    def hold(self) -> None:
        self.holds += 1

    def close(self) -> None:
        self.closes += 1


class _Camera:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.frame_id = 0
        self.frame = np.zeros((36, 64, 3), dtype=np.uint8)
        self.fail = False
        self.closes = 0
        self.jpeg: bytes | None = None

    def connect(self) -> None:
        if self.fail:
            raise RuntimeError("测试相机无法连接")

    def read_frames(self) -> list[CameraFrame]:
        frames = []
        while self.frame_id / 30.0 <= self.clock.now:
            frames.append(CameraFrame(self.frame_id, self.frame_id / 30.0, self.frame, self.jpeg))
            self.frame_id += 1
        return frames

    def close(self) -> None:
        self.closes += 1


class _Worker:
    def __init__(self, clock: _Clock, delay: float, archive: RunLog | None) -> None:
        self.clock, self.delay = clock, delay
        self.archive = archive
        self.due = 0.0
        self.pending: ObservationWindow | None = None
        self.submitted: list[ObservationWindow] = []
        self.openness = 1.0
        self.closes = 0

    @property
    def busy(self) -> bool:
        return self.pending is not None

    def start(self) -> None:
        pass

    def submit(self, window: ObservationWindow) -> None:
        assert not self.busy
        assert window.archive is None
        self.pending, self.due = window, self.clock.now + self.delay
        self.submitted.append(window)

    def poll(self) -> ActionChunk | None:
        if self.pending is None or self.clock.now < self.due:
            return None
        window, self.pending = self.pending, None
        actions = np.zeros((8, 7), dtype=np.float32)
        actions[:, 0] = np.arange(1, 9) * 0.001
        actions[:, 6] = self.openness
        result = ActionChunk(window.episode_id, window.request_id, window.timestamps[-1], actions)
        if self.archive is not None:
            self.archive.write_inference_output(result, False)
        return result

    def invalidate(self) -> None:
        # 模拟仍会传回作废请求的后端，核对运行层自身的授权判断。
        pass

    def close(self) -> None:
        self.closes += 1


class _Kinematics:
    velocities = np.full(7, 2.0)

    def forward(self, joints: FloatArray) -> FloatArray:
        tcp = np.eye(4)
        tcp[:3, 3] = (0.3 + joints[0], 0.0, 0.2)
        return tcp

    def within_limits(self, joints: FloatArray) -> bool:
        return joints.shape == (7,) and bool(np.all(np.abs(joints) <= 2.0))


class _Planner:
    def __init__(self, cfg: DeploymentConfig, kin: _Kinematics) -> None:
        self.calls = 0
        self.error: ValueError | None = None
        self.home = TrajectoryExecutor(kin, np.eye(4), cfg.execution, cfg.safety)

    def plan_home(self, start: FloatArray, end: FloatArray) -> JointTrajectory:
        return self.home.plan_home(start, end)

    def plan(
        self, actions: np.ndarray, anchor: FloatArray, current_joints: FloatArray, current_gripper: float
    ) -> JointTrajectory:
        self.calls += 1
        if self.error is not None:
            raise self.error
        q = np.tile(current_joints, (9, 1))
        q[1:, 0] += actions[:, 0]
        times = np.arange(9) * 0.04
        grippers = np.r_[current_gripper, actions[:, 6]]
        return JointTrajectory(times, q, grippers, np.stack([_Kinematics().forward(joints) for joints in q]))

    def sample(self, trajectory: JointTrajectory, elapsed: float) -> tuple[FloatArray, float]:
        joints = np.asarray([np.interp(elapsed, trajectory.times, q) for q in trajectory.joints.T])
        return joints, float(np.interp(elapsed, trajectory.times, trajectory.grippers))


class _Executor(Executor):
    def __init__(self, *, manual: bool = False) -> None:
        self.manual = manual
        self.finish: Any = None

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future:
        future: Future[JointTrajectory] = Future()
        assert future.set_running_or_notify_cancel()

        def finish() -> None:
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:
                future.set_exception(exc)

        if self.manual:
            self.finish = finish
        else:
            finish()
        return future


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    def write(self, event: str, **values: Any) -> None:
        self.items.append((event, values))

    def get(self, event: str) -> list[dict[str, Any]]:
        return [values for name, values in self.items if name == event]


class _Service:
    def __init__(self) -> None:
        self.started = False
        self.dead = False

    def start(self) -> None:
        self.started = True

    def check_alive(self) -> None:
        assert self.started
        if self.dead:
            raise RuntimeError("测试厂商服务已退出")


class _Harness:
    def __init__(
        self,
        *,
        delay: float = 0.1,
        motion: bool = True,
        max_chunks: int | None = None,
        manual_plan: bool = False,
        service: _Service | None = None,
        archive: RunLog | None = None,
    ) -> None:
        cfg = DeploymentConfig()
        cfg = replace(cfg, execution=replace(cfg.execution, start_settle_s=0.04, settle_s=0.06))
        self.clock = _Clock()
        self.arm = _Arm(self.clock)
        self.gripper = _Gripper(self.clock)
        self.camera = _Camera(self.clock)
        self.camera.jpeg = b"original" if archive is not None else None
        self.worker = _Worker(self.clock, delay, archive)
        kin = _Kinematics()
        self.planner = _Planner(cfg, kin)
        self.executor = _Executor(manual=manual_plan)
        self.events = _Events()
        assets = DeploymentAssets(np.eye(4), np.zeros(7), {}, {}, {})
        self.runtime = SerialDeployment(
            cfg,
            assets,
            self.arm,
            self.gripper,
            self.camera,
            self.worker,
            ObservationAssembler(kin, assets.tcp_offset, cfg.execution, save_inference=archive is not None),
            kin,
            self.planner,
            self.events,
            motion_enabled=motion,
            max_chunks=max_chunks,
            planner_executor=self.executor,
            gripper_server=service,
            save_observation=archive.write_inference_input if archive is not None else None,
        )

    def advance(self, seconds: float = 0.02) -> None:
        self.clock.now = round(self.clock.now + seconds, 9)
        self.runtime.tick(self.clock.now)

    def run_to(self, state: State) -> None:
        for _ in range(1200):
            if self.runtime.state is state:
                return
            if self.runtime.state is State.FAULT:
                pytest.fail(self.runtime.fault_reason)
            self.advance()
        pytest.fail(f"未到达 {state}，当前为 {self.runtime.state}")

    def start(self) -> None:
        self.runtime.connect()
        self.advance()
        assert self.runtime.command("start", self.clock.now) == "等待起始反馈稳定"


@pytest.mark.parametrize("delay", [0.1, 1.0, 5.0])
def test_two_serial_chunks_preserve_anchor_and_waypoints(delay: float) -> None:
    run = _Harness(delay=delay, max_chunks=2)
    run.start()
    run.run_to(State.READY)
    assert len(run.events.get("chunk_completed")) == 2
    assert len(run.worker.submitted) == 2
    first, second = run.worker.submitted
    np.testing.assert_array_equal(first.anchor, second.anchor)
    assert second.request_id > first.request_id
    assert second.timestamps[0] >= run.events.get("chunk_completed")[0]["monotonic"]
    assert first.episode_id == second.episode_id
    for request_id, trajectory in enumerate(run.events.get("trajectory"), start=1):
        points = [
            values["joints"]
            for values in run.events.get("joint_command")
            if values["request_id"] == request_id
        ]
        # 八个完整目标在慢推理后仍全部经过。
        for waypoint in trajectory["joints"][1:]:
            assert any(np.allclose(point, waypoint) for point in points)
    assert run.arm.stops == 1
    assert run.gripper.ticks > 0
    run.runtime.close(run.clock.now)


def test_control_delay_cannot_skip_waypoints() -> None:
    run = _Harness(max_chunks=1)
    run.start()
    run.run_to(State.EXECUTING)
    while run.runtime.state is State.EXECUTING:
        run.advance(0.06)
    assert run.runtime.state is State.SETTLING
    trajectory = run.events.get("trajectory")[0]
    for waypoint in trajectory["joints"][1:]:
        assert any(np.allclose(point, waypoint) for point in run.arm.commands)
    steps = np.diff(np.asarray(run.arm.commands), axis=0)
    assert np.max(steps) <= 0.0005 + 1e-9
    run.runtime.close(run.clock.now)


def test_send_rejects_unsafe_off_grid_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """控制间隔偏离 20 ms 网格时，目标越界须在发送前发现."""
    run = _Harness(max_chunks=1)
    run.start()
    run.run_to(State.EXECUTING)
    forward = run.runtime.kinematics.forward

    def near_boundary(joints: FloatArray) -> FloatArray:
        tcp = forward(joints)
        if 0.00024 < joints[0] < 0.00026:
            tcp[2, 3] = 0.01
        return tcp

    monkeypatch.setattr(run.runtime.kinematics, "forward", near_boundary)
    monkeypatch.setattr(run.runtime, "_dispatch_time", lambda now: now + 0.01)
    run.advance()
    assert len(run.arm.commands) == 1
    run.advance()
    assert run.runtime.state is State.FAULT
    assert "TCP 超出工作空间" in run.runtime.fault_reason
    assert len(run.arm.commands) == run.arm.stops == 1
    run.runtime.close(run.clock.now)


def test_home_rejects_path_with_safe_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """预检中间点越界时不移动机械臂或夹爪."""
    run = _Harness()
    run.runtime.connect()
    run.arm.joints[0] = 0.1
    forward = run.runtime.kinematics.forward

    def near_boundary(joints: FloatArray) -> FloatArray:
        tcp = forward(joints)
        if 0.04 < joints[0] < 0.06:
            tcp[2, 3] = 0.01
        return tcp

    monkeypatch.setattr(run.runtime.kinematics, "forward", near_boundary)
    run.advance()
    assert run.runtime.command("home", run.clock.now) == "预检回位路径"
    run.run_to(State.READY)
    assert "回位路径" in run.events.get("home_rejected")[0]["reason"]
    assert not run.arm.commands and not run.arm.homes and not run.gripper.targets
    run.runtime.close(run.clock.now)


def test_stop_discards_pending_home_plan() -> None:
    """停止后返回的回位计划不会驱动任何执行器."""
    run = _Harness(manual_plan=True)
    run.runtime.connect()
    run.advance()
    assert run.runtime.command("home", run.clock.now) == "预检回位路径"
    assert run.runtime.state is State.HOME_PLANNING
    run.runtime.command("stop", run.clock.now)
    run.executor.finish()
    run.advance()
    assert run.runtime.state is State.READY
    assert not run.arm.commands and not run.arm.homes and not run.gripper.targets
    run.runtime.close(run.clock.now)


@pytest.mark.parametrize("reaches_target", [True, False])
def test_chunk_limit_waits_for_gripper_feedback(reaches_target: bool) -> None:
    """夹爪滞后时继续闭合，到位后才完成，持续不到位则超时."""
    run = _Harness(max_chunks=1)
    run.worker.openness = 0.0
    run.gripper.track = False
    run.start()
    run.run_to(State.SETTLING)
    assert run.gripper.targets[-1] == 0.0
    for _ in range(10):
        run.advance()
    assert run.runtime.state is State.SETTLING
    assert not run.events.get("chunk_completed")
    if reaches_target:
        run.gripper.openness = 0.0
        run.run_to(State.READY)
        assert len(run.events.get("chunk_completed")) == 1
        assert run.events.get("stopped")[-1]["reason"] == "chunk_limit"
    else:
        for _ in range(300):
            run.advance()
        assert run.runtime.state is State.FAULT
        assert not run.events.get("chunk_completed")
        assert run.arm.stops == 1
    run.runtime.close(run.clock.now)


def test_stop_during_inference_drops_old_result_and_refreezes_anchor() -> None:
    run = _Harness(delay=6.0)
    run.start()
    run.run_to(State.INFERRING)
    old = run.worker.submitted[-1]
    run.runtime.command("stop", run.clock.now)
    assert run.runtime.state is State.READY
    run.arm.joints[0] = 0.001
    run.advance()
    assert "已作废" in run.runtime.command("start", run.clock.now)
    while run.worker.busy:
        run.advance()
    assert run.runtime.state is State.READY
    run.worker.delay = 0.1
    assert run.runtime.command("start", run.clock.now) == "等待起始反馈稳定"
    run.run_to(State.PLANNING)
    assert not run.arm.commands
    assert run.events.get("discard_inference")[0]["request_id"] == old.request_id
    new = run.worker.submitted[-1]
    assert new.episode_id > old.episode_id
    assert new.request_id > old.request_id
    assert not np.array_equal(new.anchor, old.anchor)
    assert run.planner.calls == 1
    run.runtime.close(run.clock.now)


def test_stop_during_planning_discards_late_ik() -> None:
    run = _Harness(manual_plan=True)
    run.start()
    run.run_to(State.PLANNING)
    run.runtime.command("stop", run.clock.now)
    assert run.runtime.state is State.READY
    run.executor.finish()
    run.advance()
    assert run.events.get("discard_plan")
    assert not run.arm.commands
    assert not run.events.get("trajectory")
    run.runtime.close(run.clock.now)


@pytest.mark.parametrize("condition", ["drift", "feedback_loss", "latched_error", "control_gap"])
def test_faults_block_further_actuators(condition: str) -> None:
    run = _Harness(delay=1.0)
    run.start()
    run.run_to(State.INFERRING)
    if condition == "drift":
        run.arm.joints[0] = 0.003
    elif condition == "feedback_loss":
        run.arm.drop = True
    elif condition == "latched_error":
        run.arm.errors = (42,)
        run.arm.stamp = run.clock.now
    if condition == "control_gap":
        run.advance(0.11)
    else:
        run.run_to(State.FAULT)
    assert run.runtime.state is State.FAULT
    assert not run.arm.commands
    assert not run.events.get("trajectory")
    assert "fault" in run.runtime.command("start", run.clock.now)
    run.runtime.close(run.clock.now)


@pytest.mark.parametrize(
    ("motion", "error"),
    [(True, ValueError("第八点 IK 失败")), (False, ActionWorkspaceError("第八点超出工作空间"))],
)
def test_fatal_trajectory_rejection_sends_no_arm_command(motion: bool, error: ValueError) -> None:
    """其他规划错误和空跑越界继续按故障处理。"""
    run = _Harness(motion=motion)
    run.planner.error = error
    run.start()
    run.run_to(State.FAULT)
    assert "第八点" in run.runtime.fault_reason
    assert not run.arm.commands
    assert not run.gripper.targets
    run.runtime.close(run.clock.now)


def test_action_workspace_rejection_stops_and_waits_for_new_command() -> None:
    """第二段越界时停臂，等待操作员发起新回合。"""
    run = _Harness()
    run.start()
    run.run_to(State.EXECUTING)
    run.run_to(State.OBSERVING)
    reason = "动作点 8 超出基座系 TCP 工作空间"
    run.planner.error = ActionWorkspaceError(reason)
    run.run_to(State.PLANNING)
    sent = len(run.arm.commands)
    gripper_targets = len(run.gripper.targets)

    run.run_to(State.READY)
    assert run.arm.stops == 1
    assert len(run.events.get("trajectory")) == 1
    assert run.events.get("episode_summary")[0]["reason"] == reason
    assert run.events.get("episode_summary")[0]["chunks_completed"] == 1
    assert not run.events.get("fault")
    assert run.arm.closes == run.worker.closes == 0
    for _ in range(10):
        run.advance()
    assert run.runtime.state is State.READY
    assert len(run.worker.submitted) == 2
    assert len(run.arm.commands) == sent
    assert len(run.gripper.targets) == gripper_targets

    run.planner.error = None
    assert run.runtime.command("home", run.clock.now) == "预检回位路径"
    run.run_to(State.READY)
    assert run.events.get("home_joint_command") and run.arm.homes == 0
    np.testing.assert_array_equal(run.arm.joints, run.runtime.assets.initial_joints)
    assert run.runtime.command("start", run.clock.now) == "等待起始反馈稳定"
    run.run_to(State.EXECUTING)
    assert run.runtime.episode_id == 2
    run.runtime.close(run.clock.now)


def test_rejected_action_keeps_inference_input_and_output(tmp_path: Path) -> None:
    """模型输出在越界预检之前落入存档，且不会下发被拒绝的轨迹."""
    with RunLog(tmp_path, {"mode": "run"}, save_inference=True) as archive:
        run = _Harness(archive=archive)
        run.planner.error = ActionWorkspaceError("动作点 8 超出基座系 TCP 工作空间")
        run.start()
        run.run_to(State.READY)
        assert not run.arm.commands
        assert not run.events.get("trajectory")
        run.runtime.close(run.clock.now)
    name = "episode_000001_request_000001"
    assert (archive.directory / "inference_inputs" / name / "00.jpg").read_bytes() == b"original"
    output = json.loads((archive.directory / "inference_outputs" / f"{name}.json").read_text())
    assert len(output["actions"]) == 8
    assert output["discarded"] is False


def test_send_failure_requests_stop_even_when_command_may_not_arrive() -> None:
    run = _Harness()
    run.start()
    run.run_to(State.EXECUTING)
    run.arm.send_failure = True
    run.advance()
    assert run.runtime.state is State.FAULT
    assert run.arm.stops == 1
    assert not run.gripper.targets
    run.runtime.close(run.clock.now)


def test_dry_run_plans_and_plays_clock_without_actuator_calls() -> None:
    run = _Harness(motion=False, max_chunks=1)
    run.worker.openness = 0.0
    run.start()
    run.run_to(State.READY)
    assert run.events.get("trajectory")
    assert "禁止" in run.runtime.command("home", run.clock.now)
    run.runtime.stop(run.clock.now)
    run.runtime.close(run.clock.now)
    assert all(not values["applied"] for values in run.events.get("joint_command"))
    assert not run.arm.commands and run.arm.homes == run.arm.stops == 0
    assert not run.gripper.targets and run.gripper.holds == run.gripper.ticks == 0


def test_partial_connection_closes_each_resource_without_motion() -> None:
    run = _Harness()
    run.camera.fail = True
    with pytest.raises(RuntimeError, match="相机"):
        run.runtime.connect()
    assert run.runtime.state is State.CLOSED
    assert run.arm.closes == run.gripper.closes == run.camera.closes == run.worker.closes == 1
    assert not run.arm.commands and run.arm.homes == run.arm.stops == 0
    assert not run.gripper.targets and run.gripper.holds == run.gripper.ticks == 0


def test_failed_stop_is_reported_after_closing_all_resources() -> None:
    run = _Harness()
    run.start()
    run.run_to(State.EXECUTING)
    run.advance()
    run.arm.stop_failure = True
    with pytest.raises(RuntimeError, match="停臂失败"):
        run.runtime.close(run.clock.now)
    assert run.runtime.state is State.CLOSED
    assert run.arm.closes == run.gripper.closes == run.camera.closes == run.worker.closes == 1
    assert run.events.get("closed")[0]["errors"]


def test_in_tick_block_prevents_first_send(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _Harness()
    run.start()
    run.run_to(State.EXECUTING)
    # 模拟本周期采集/处理已耗时 110 ms，下一条指令必须在 SDK 边界之前取消。
    timestamps = iter([100.0, 100.11])
    monkeypatch.setattr("lerobot.deployment.rm75b.runtime.time.monotonic", lambda: next(timestamps))
    run.advance()
    assert run.runtime.state is State.FAULT
    assert "截止" in run.runtime.fault_reason
    assert not run.arm.commands
    run.runtime.close(run.clock.now)


def test_managed_service_ready_before_dds_and_death_faults_active_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _Service()
    run = _Harness(service=service)

    def connect_gripper() -> None:
        assert service.started

    monkeypatch.setattr(run.gripper, "connect", connect_gripper)
    run.start()
    run.run_to(State.EXECUTING)
    run.advance()
    assert run.arm.commands
    commands = len(run.arm.commands)
    service.dead = True
    run.advance()
    assert run.runtime.state is State.FAULT
    assert "厂商服务已退出" in run.runtime.fault_reason
    assert run.arm.stops == 1
    assert len(run.arm.commands) == commands
    run.runtime.close(run.clock.now)


def test_cli_owned_failed_connect_defers_device_cleanup() -> None:
    run = _Harness(service=_Service())
    run.camera.fail = True
    with pytest.raises(RuntimeError, match="相机"):
        run.runtime.connect(cleanup_on_error=False)
    assert run.arm.closes == run.gripper.closes == run.camera.closes == 0
    run.runtime.close(run.clock.now)
    assert run.arm.closes == run.gripper.closes == run.camera.closes == 1


def test_dry_run_ignores_injected_managed_service() -> None:
    service = _Service()
    run = _Harness(motion=False, service=service)
    run.start()
    assert not service.started
    run.runtime.close(run.clock.now)


def test_ready_gripper_gap_cannot_resume_output() -> None:
    run = _Harness(max_chunks=1)
    run.start()
    run.run_to(State.READY)
    count = run.gripper.ticks
    assert count > 0
    run.advance(0.11)
    assert run.runtime.state is State.FAULT
    assert run.runtime.fault_reason == "夹爪命令间隔超时"
    assert run.gripper.ticks == count
    run.runtime.close(run.clock.now)
