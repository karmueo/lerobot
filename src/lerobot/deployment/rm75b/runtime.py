"""RM75B 串行状态机。控制线程拥有驱动，推理与规划进程只处理数据."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import replace
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .geometry import rotation_distance
from .planning_worker import PlanningWorker
from .trajectory import HOME_VELOCITY_SCALE, ActionWorkspaceError
from .types import (
    ActionChunk,
    ArmInterface,
    ArmState,
    CameraFrame,
    CameraInterface,
    EventLogger,
    FloatArray,
    GripperInterface,
    GripperState,
    InferenceInterface,
    JointTrajectory,
    ObservationWindow,
)

if TYPE_CHECKING:
    from .config import DeploymentAssets, DeploymentConfig
    from .gripper_server import GripperServer

logger = logging.getLogger(__name__)


class State(StrEnum):
    """一个任务只在稳定后进入下一次采集."""

    READY = "ready"
    HOME_PLANNING = "home_planning"
    HOMING = "homing"
    STARTING = "starting"
    OBSERVING = "observing"
    INFERRING = "inferring"
    PLANNING = "planning"
    EXECUTING = "executing"
    SETTLING = "settling"
    FAULT = "fault"
    CLOSED = "closed"


_EPISODE_STATES = frozenset(
    {State.OBSERVING, State.INFERRING, State.PLANNING, State.EXECUTING, State.SETTLING}
)
_ACTIVE_STATES = _EPISODE_STATES | {State.HOME_PLANNING, State.HOMING, State.STARTING}


class Kinematics(Protocol):
    """运行层仅使用只读正运动学."""

    velocities: FloatArray

    def forward(self, joints: FloatArray) -> FloatArray:
        """从七轴关节反馈计算 base→Link7."""
        ...


class WindowAssembler(Protocol):
    """按统一时钟保存近期反馈，并构造三帧窗口."""

    def add_arm(self, state: ArmState) -> None:
        """保存机械臂反馈."""
        ...

    def add_gripper(self, state: GripperState) -> None:
        """保存夹爪反馈."""
        ...

    def add_frame(self, frame: CameraFrame) -> None:
        """保存独立相机帧."""
        ...

    def clear(self) -> None:
        """清除当前回合输入历史."""
        ...

    def build(
        self, anchor: FloatArray, episode_id: int, request_id: int, not_before: float, now: float
    ) -> ObservationWindow | None:
        """构造稳定时刻之后的同步观测."""
        ...


class Planner(Protocol):
    """规划不接触驱动，线程中的迟到结果可被安全丢弃."""

    def plan(
        self,
        actions: np.ndarray,
        anchor: FloatArray,
        current_joints: FloatArray,
        current_gripper: float,
    ) -> JointTrajectory:
        """整段预检并计算关节路点."""
        ...

    def sample(self, trajectory: JointTrajectory, elapsed: float) -> tuple[FloatArray, float]:
        """在共同时间轴上插值机械臂与夹爪."""
        ...

    def plan_home(self, start: FloatArray, end: FloatArray) -> JointTrajectory:
        """预检实际执行的整条回位关节路径."""
        ...


class SerialDeployment:
    """以单线程时钟推进机器人状态，外部循环以 50 Hz 调用 tick."""

    def __init__(
        self,
        cfg: DeploymentConfig,
        assets: DeploymentAssets,
        arm: ArmInterface,
        gripper: GripperInterface,
        camera: CameraInterface,
        inference: InferenceInterface,
        assembler: WindowAssembler,
        kinematics: Kinematics,
        planner: Planner,
        events: EventLogger,
        *,
        motion_enabled: bool = False,
        max_chunks: int | None = None,
        planner_executor: Executor | None = None,
        gripper_server: GripperServer | None = None,
        save_observation: Callable[[ObservationWindow], None] | None = None,
    ) -> None:
        """注入设备与纯规划接口，便于在无硬件条件下独立验证状态机."""
        self.cfg, self.assets = cfg, assets
        self.arm, self.gripper, self.camera = arm, gripper, camera
        self.inference, self.assembler, self.kinematics = inference, assembler, kinematics
        self.planner, self.events = planner, events
        self._save_observation = save_observation
        self.motion_enabled, self.max_chunks = motion_enabled, max_chunks
        self._gripper_server = gripper_server if motion_enabled else None
        self._executor = planner_executor
        self._planning_worker = (
            PlanningWorker(cfg.arm.urdf, assets.tcp_offset, assets.initial_joints, cfg.execution, cfg.safety)
            if planner_executor is None
            else None
        )
        self.period = 1.0 / cfg.execution.control_hz
        self.state = State.READY
        self.fault_reason: str | None = None
        self.episode_id = 0
        self.request_id = 0
        self._phase_started = 0.0
        self._episode_started = 0.0
        self._stable_since: float | None = None
        self._not_before = 0.0
        self._last_tick: float | None = None
        self._tick_wall_started = 0.0
        self._arm_state: ArmState | None = None
        self._gripper_state: GripperState | None = None
        self._camera_stamp: float | None = None
        self._camera_id: int | None = None
        self._anchor: FloatArray | None = None
        self._window: ObservationWindow | None = None
        self._future: Future[JointTrajectory] | Future[JointTrajectory | None] | None = None
        self._plan_key: tuple[int, int] | None = None
        self._trajectory: JointTrajectory | None = None
        self._arm_target: FloatArray | None = None
        self._arm_may_move = False
        self._gripper_enabled = False
        self._stop_gripper = False
        self._last_send: float | None = None
        self._last_gripper_send: float | None = None
        self._sent_this_tick = False
        self._elapsed = 0.0
        self._trajectory_started = False
        self._next_waypoint = 1
        self._chunks_executed = 0
        self._resources: list[
            tuple[
                str, ArmInterface | GripperInterface | CameraInterface | InferenceInterface | PlanningWorker
            ]
        ] = []

    def connect(self, *, cleanup_on_error: bool = True) -> None:
        """先预热再连接；CLI 托管时把失败清理统一留给外层."""
        try:
            if self._planning_worker is not None:
                self._resources.append(("planning", self._planning_worker))
                self._planning_worker.start()
                self.events.write(
                    "planning_ready", backend="spawn", numeric_threads=1, pid=self._planning_worker.pid
                )
            for name, component, opener in (
                ("inference", self.inference, self.inference.start),
                ("arm", self.arm, self.arm.connect),
                ("gripper", self.gripper, self.gripper.connect),
                ("camera", self.camera, self.camera.connect),
            ):
                service = self._gripper_server if name == "gripper" else None
                if service is not None:
                    service.start()
                self._resources.append((name, component))
                try:
                    opener()
                except Exception as exc:
                    if service is not None:
                        try:
                            service.check_alive()
                        except RuntimeError as server_error:
                            raise server_error from exc
                    raise
                if service is not None:
                    service.check_alive()
            self.events.write("connected", motion_enabled=self.motion_enabled)
        except BaseException as exc:
            if cleanup_on_error:
                try:
                    self.close()
                except Exception as cleanup_error:
                    exc.add_note(f"启动失败后的资源清理也失败 {cleanup_error}")
            raise

    def _transition(self, state: State, now: float) -> None:
        """同时记录阶段开始与上一阶段耗时."""
        previous = self.state
        duration = now - self._phase_started
        self.state, self._phase_started = state, now
        self._stable_since = None
        self.events.write(
            "phase",
            previous=previous,
            state=state,
            duration_s=duration if self._last_tick is not None else 0.0,
            episode_id=self.episode_id,
            request_id=self.request_id,
            monotonic=now,
        )

    def _fresh(self, timestamp: float | None, limit: float, now: float) -> bool:
        """设备回调可能晚于 tick 取时几毫秒，拒绝明显不一致的时钟."""
        return timestamp is not None and -self.period <= now - timestamp <= limit

    def _dispatch_time(self, now: float) -> float:
        """检查本次 tick 内发生的阻塞，避免超期后仍发送一次目标."""
        elapsed = time.monotonic() - self._tick_wall_started
        actual = now + elapsed
        limit = self.cfg.safety.max_send_gap_s
        if elapsed > limit or (self._last_send is not None and actual - self._last_send > limit):
            raise RuntimeError("控制周期内发送截止时刻已过")
        return actual

    def _read_feedback(self, now: float) -> None:
        """仅向历史缓存和日志写入新收到的反馈."""
        arm, gripper = self.arm.read_state(), self.gripper.read_state()
        if arm is not None and arm.errors:
            # 驱动故障可能锁存于同一时刻的反馈，不能等待下一帧再发现。
            raise RuntimeError(f"机械臂报告故障 {arm.errors}")
        if arm is not None and (self._arm_state is None or arm.timestamp > self._arm_state.timestamp):
            self._arm_state = arm
            self.assembler.add_arm(arm)
            self.events.write("arm_feedback", feedback=arm, monotonic=now)
        if gripper is not None and (
            self._gripper_state is None or gripper.timestamp > self._gripper_state.timestamp
        ):
            self._gripper_state = gripper
            self.assembler.add_gripper(gripper)
            self.events.write("gripper_feedback", feedback=gripper, monotonic=now)
        for frame in self.camera.read_frames():
            if self._camera_id is None or frame.frame_id > self._camera_id:
                self._camera_id, self._camera_stamp = frame.frame_id, frame.timestamp
                self.assembler.add_frame(frame)
                self.events.write(
                    "camera_frame", frame_id=frame.frame_id, timestamp=frame.timestamp, monotonic=now
                )

    def _feedback(self, now: float) -> tuple[ArmState, GripperState]:
        """返回健康且新鲜的反馈。旧值不能用于继续动作."""
        arm, gripper = self._arm_state, self._gripper_state
        if arm is None or not self._fresh(arm.timestamp, self.cfg.safety.feedback_timeout_s, now):
            raise RuntimeError("机械臂反馈超时")
        if arm.errors:
            raise RuntimeError(f"机械臂报告故障 {arm.errors}")
        if (
            arm.joints.shape != (7,)
            or arm.velocities.shape != (7,)
            or not np.isfinite(arm.joints).all()
            or not np.isfinite(arm.velocities).all()
        ):
            raise RuntimeError("机械臂反馈格式无效")
        if gripper is None or not self._fresh(gripper.timestamp, self.cfg.gripper.feedback_timeout_s, now):
            raise RuntimeError("夹爪反馈超时")
        if not math.isfinite(gripper.openness) or not 0.0 <= gripper.openness <= 1.0:
            raise RuntimeError("夹爪反馈开度无效")
        if not self._fresh(self._camera_stamp, self.cfg.camera.frame_timeout_s, now):
            raise RuntimeError("相机输入超时")
        return arm, gripper

    def _tcp(self, joints: FloatArray) -> FloatArray:
        """由反馈关节角获取 base 系 TCP 并检查当前工作空间."""
        tcp = self.kinematics.forward(joints) @ self.assets.tcp_offset
        xyz = tcp[:3, 3]
        if (
            not np.isfinite(tcp).all()
            or np.any(xyz < self.cfg.safety.workspace_min)
            or np.any(xyz > self.cfg.safety.workspace_max)
        ):
            raise RuntimeError(f"TCP 超出工作空间 {xyz.tolist()}")
        return tcp

    def _at_target(self, arm: ArmState, target: FloatArray) -> bool:
        """末点和回合起点统一要求关节跟踪误差与速度达标."""
        return bool(
            np.max(np.abs(arm.joints - target)) <= self.cfg.execution.start_joint_tolerance_rad
            and np.max(np.abs(arm.velocities)) <= self.cfg.execution.stationary_velocity_rad_s
        )

    def _stable(self, qualifying: bool, now: float, duration: float) -> bool:
        """中途离开容差时重新开始计时."""
        if not qualifying:
            self._stable_since = None
            return False
        if self._stable_since is None:
            self._stable_since = now
        return now - self._stable_since >= duration

    def _invalidate(self) -> None:
        """撤销当前观测、计划和轨迹的执行资格."""
        self.inference.invalidate()
        self._plan_key = None
        if self._future is not None:
            self._future.cancel()
        self._anchor = self._window = self._trajectory = None
        self._arm_target = None
        self._last_send = None
        self.assembler.clear()

    def _halt_outputs(self) -> list[str]:
        """独立尝试停止机械臂与冻结夹爪，避免一次失败跳过另一执行器."""
        errors: list[str] = []
        if self.motion_enabled and self._arm_may_move:
            try:
                self.arm.stop()
                self._arm_may_move = False
            except Exception as exc:
                errors.append(f"停臂失败 {exc}")
        if self.motion_enabled and self._gripper_enabled:
            try:
                self.gripper.hold()
            except Exception as exc:
                self._gripper_enabled = False
                errors.append(f"夹爪保持失败 {exc}")
        return errors

    def _end_episode(self, now: float, reason: str) -> None:
        """记录当前回合已稳定完成的动作段数."""
        if self.state in _EPISODE_STATES:
            self.events.write(
                "episode_summary",
                episode_id=self.episode_id,
                chunks_completed=self._chunks_executed,
                duration_s=now - self._episode_started,
                reason=reason,
                monotonic=now,
            )

    def _fault(self, now: float, reason: str) -> None:
        """先停臂，再记录故障。故障会话必须退出后重启."""
        if self.state in (State.CLOSED, State.FAULT):
            return
        errors = self._halt_outputs()
        self._stop_gripper = True
        try:
            self._invalidate()
        except Exception as exc:
            errors.append(f"作废执行状态失败 {exc}")
        if errors:
            reason = "。".join([reason, *errors])
        self.fault_reason = reason
        try:
            self._end_episode(now, reason)
            self._transition(State.FAULT, now)
            self.events.write("fault", reason=reason, monotonic=now)
        finally:
            self.state = State.FAULT
        logger.error("RM75B 停止执行：%s", reason)

    def stop(self, now: float, reason: str = "operator_stop") -> None:
        """结束当前回合并清除待执行内容，夹爪保留最后安全目标."""
        if self.state in (State.CLOSED, State.FAULT):
            return
        errors = self._halt_outputs()
        if errors:
            self._fault(now, "。".join(errors))
            return
        try:
            self._invalidate()
            self._end_episode(now, reason)
            self._transition(State.READY, now)
            self.events.write("stopped", reason=reason, monotonic=now)
        except Exception as exc:
            self._fault(now, str(exc))

    def command(self, command: str, now: float) -> str:
        """处理显式命令，返回可显示的状态文字."""
        if command == "stop":
            self.stop(now)
            return "已请求停止"
        if command not in ("home", "start"):
            return "可用命令为 home、start、stop、quit"
        if self.state is not State.READY:
            return f"当前状态 {self.state}，无法执行 {command}"
        if self._future is not None or (command == "start" and self.inference.busy):
            return f"等待已作废的计算结束，请稍后重新输入 {command}"
        if command == "home" and not self.motion_enabled:
            return "dry-run 禁止回位，请在开始前将机械臂和夹爪置于起始状态"
        if self._gripper_server is not None:
            try:
                self._gripper_server.check_alive()
            except RuntimeError as exc:
                self._fault(now, str(exc))
                return str(exc)
        try:
            arm, gripper = self._feedback(now)
            self._tcp(arm.joints)
            self._tcp(self.assets.initial_joints)
        except RuntimeError as exc:
            return str(exc)
        if command == "start":
            if (
                np.max(np.abs(arm.joints - self.assets.initial_joints))
                > self.cfg.execution.start_joint_tolerance_rad
            ):
                return "机械臂未到初始关节位，请先回位"
            if gripper.openness < self.cfg.execution.start_gripper_min:
                return "夹爪未达到起始开度，请先张开"
            self._transition(State.STARTING, now)
            return "等待起始反馈稳定"
        if np.max(np.abs(arm.velocities)) > self.cfg.execution.stationary_velocity_rad_s:
            return "机械臂尚未静止"
        try:
            if self._planning_worker is not None:
                self._future = self._planning_worker.submit_home(arm.joints)
            else:
                assert self._executor is not None
                self._future = self._executor.submit(
                    self.planner.plan_home, arm.joints.copy(), self.assets.initial_joints.copy()
                )
            self._transition(State.HOME_PLANNING, now)
        except Exception as exc:
            self._fault(now, str(exc))
            return f"回位失败：{exc}"
        return "预检回位路径"

    def _send(self, joints: FloatArray, gripper: float | None, now: float) -> None:
        """维持相同底层隔离。dry-run 仅记录预定指令."""
        # 实际采样点可能落在预检网格之间，不能用实测 TCP 检查替代目标检查。
        self._tcp(joints)
        self._feedback(self._dispatch_time(now))
        actual = self._dispatch_time(now)
        if self.motion_enabled:
            self._arm_may_move = True
            self.arm.send_joint_target(joints)
        self._last_send = actual
        self._sent_this_tick = True
        self.events.write(
            "home_joint_command" if self.state is State.HOMING else "joint_command",
            episode_id=self.episode_id,
            request_id=self.request_id,
            joints=joints,
            gripper=gripper,
            applied=self.motion_enabled,
            monotonic=actual,
        )
        if self.motion_enabled and gripper is not None:
            self.gripper.set_target(gripper)
            self._gripper_enabled = True

    def _begin_episode(self, arm: ArmState, now: float) -> None:
        """将实测 TCP 冻结为原点，只在新的 start 之后执行一次."""
        self._invalidate()
        self.episode_id += 1
        self._episode_started, self._chunks_executed = now, 0
        self._anchor = self._tcp(arm.joints)
        self._arm_target = arm.joints.copy()
        if self.motion_enabled:
            self.gripper.hold()
            self._gripper_enabled = True
        self._not_before = now
        self.events.write("episode_start", episode_id=self.episode_id, anchor=self._anchor, monotonic=now)
        self._transition(State.OBSERVING, now)

    def _poll_inference(self, now: float) -> ActionChunk | None:
        """运行全过程检查进程存活，并过滤非当前请求的结果."""
        result = self.inference.poll()
        if result is None:
            return None
        window = self._window
        if (
            self.state is not State.INFERRING
            or window is None
            or (result.episode_id, result.request_id) != (window.episode_id, window.request_id)
            or result.observation_timestamp != window.timestamps[-1]
        ):
            self.events.write(
                "discard_inference",
                episode_id=result.episode_id,
                request_id=result.request_id,
                reason="stale_request",
                monotonic=now,
            )
            return None
        return result

    def _check_drift(self, tcp: FloatArray) -> None:
        """慢推理不裁剪动作，但观测期间发生实体位移会使整个结果失效."""
        if self._window is None:
            raise RuntimeError("当前没有可用于执行的观测窗口")
        captured = self._window.capture_tcp
        if (
            np.linalg.norm(tcp[:3, 3] - captured[:3, 3]) > self.cfg.safety.drift_translation_m
            or rotation_distance(tcp[:3, :3], captured[:3, :3]) > self.cfg.safety.drift_rotation_rad
        ):
            raise RuntimeError("推理或规划期间 TCP 相对观测发生漂移")

    def _observe(self, now: float) -> None:
        """完整窗口必须晚于本次稳定时刻，单次只提交一个推理请求."""
        if self._anchor is None:
            raise RuntimeError("回合起点未冻结")
        if now - self._phase_started > self.cfg.execution.settle_timeout_s:
            raise RuntimeError("等待同步观测窗口超时")
        if self._future is not None or self.inference.busy:
            return
        request_id = self.request_id + 1
        window = self.assembler.build(self._anchor, self.episode_id, request_id, self._not_before, now)
        if window is None:
            return
        # 存档数据留在主进程，模型仍只接收原有输入。
        model_window = replace(window, archive=None) if window.archive is not None else window
        self.inference.submit(model_window)
        if self._save_observation is not None:
            self._save_observation(window)
        self.request_id, self._window = request_id, model_window
        self.events.write(
            "observation",
            episode_id=self.episode_id,
            request_id=request_id,
            frame_ids=window.frame_ids,
            timestamps=window.timestamps,
            states=window.states,
            capture_tcp=window.capture_tcp,
            monotonic=now,
        )
        self._transition(State.INFERRING, now)

    def _plan(self, result: ActionChunk, arm: ArmState, gripper: GripperState, now: float) -> None:
        """向独立进程提交纯 IK 预检，控制循环不等待结果."""
        if self._anchor is None or self._future is not None:
            raise RuntimeError("轨迹规划状态不一致")
        self.events.write(
            "prediction",
            episode_id=result.episode_id,
            request_id=result.request_id,
            observation_timestamp=result.observation_timestamp,
            actions=result.actions,
            timings=result.timings,
            monotonic=now,
        )
        self._plan_key = result.episode_id, result.request_id
        started = time.monotonic()
        if self._planning_worker is not None:
            self._future = self._planning_worker.submit_policy(
                result.actions, self._anchor, arm.joints, gripper.openness
            )
        else:
            assert self._executor is not None
            self._future = self._executor.submit(
                self.planner.plan,
                result.actions.copy(),
                self._anchor.copy(),
                arm.joints.copy(),
                gripper.openness,
            )
        self.events.write(
            "planning_request",
            episode_id=result.episode_id,
            request_id=result.request_id,
            submit_s=time.monotonic() - started,
            monotonic=now,
        )
        self._transition(State.PLANNING, now)

    def _poll_plan(self, arm: ArmState, now: float) -> None:
        """只有与当前回合和请求一致的完成计划才可执行."""
        future = self._future
        if future is None or not future.done():
            return
        self._future = None
        if self._plan_key != (self.episode_id, self.request_id):
            raise RuntimeError("轨迹计划与当前请求不一致")
        try:
            trajectory = future.result()
        except ActionWorkspaceError as exc:
            if not self.motion_enabled:
                raise
            # 整段未下发，沿用手动停止流程作废本回合并等待操作员。
            logger.warning("RM75B 拒绝动作段：%s", exc)
            self.stop(now, reason=str(exc))
            return
        if trajectory is None:
            raise RuntimeError("规划进程未返回模型轨迹")
        self._plan_key = None
        if np.max(np.abs(arm.joints - trajectory.joints[0])) > self.cfg.execution.start_joint_tolerance_rad:
            raise RuntimeError("规划期间机械臂偏离关节种子")
        self._load_trajectory(trajectory)
        self.events.write(
            "trajectory",
            episode_id=self.episode_id,
            request_id=self.request_id,
            times=trajectory.times,
            joints=trajectory.joints,
            grippers=trajectory.grippers,
            tcp_targets=trajectory.tcp_targets,
            duration_s=trajectory.duration,
            monotonic=now,
        )
        self._transition(State.EXECUTING, now)

    def _load_trajectory(self, trajectory: JointTrajectory) -> None:
        """回位和模型执行共用插值进度与首点检查."""
        self._trajectory = trajectory
        self._elapsed, self._next_waypoint = 0.0, 1
        self._trajectory_started = False
        self._arm_target = trajectory.joints[0].copy()

    def _poll_home(self, now: float) -> None:
        """仅在整条回位路径预检通过后允许执行."""
        future = self._future
        if future is None or not future.done():
            return
        self._future = None
        try:
            trajectory = future.result()
        except ValueError as exc:
            logger.warning("RM75B 拒绝回位：%s", exc)
            self.events.write("home_rejected", reason=str(exc), monotonic=now)
            self._transition(State.READY, now)
            return
        if trajectory is None:
            raise RuntimeError("规划进程未返回回位轨迹")
        self._load_trajectory(trajectory)
        self.events.write("home_trajectory", trajectory=trajectory, monotonic=now)
        self._transition(State.HOMING, now)

    def _advance_trajectory(self, arm: ArmState, now: float, *, homing: bool = False) -> bool:
        """每次最多推进一个周期且停在经过的路点，不按墙钟跳过指令."""
        trajectory = self._trajectory
        if trajectory is None:
            raise RuntimeError("没有完成预检的轨迹")
        first = not self._trajectory_started
        if first:
            scale = self.cfg.execution.velocity_scale
            if homing:
                scale = min(scale, HOME_VELOCITY_SCALE)
                if np.max(np.abs(arm.velocities)) > self.cfg.execution.stationary_velocity_rad_s:
                    raise RuntimeError("预检回位后机械臂不再静止")
            first_step = self.kinematics.velocities * scale * self.period
            if np.any(np.abs(arm.joints - trajectory.joints[0]) > first_step):
                raise RuntimeError("发送前关节偏离轨迹起点")
            interval = 0.0
            self._trajectory_started = True
        else:
            interval = 0.0 if self._last_send is None else min(self.period, max(0.0, now - self._last_send))
        waypoint = float(trajectory.times[self._next_waypoint])
        self._elapsed = min(waypoint, self._elapsed + interval)
        joints, gripper = self.planner.sample(trajectory, self._elapsed)
        self._send(joints, None if homing and not first else gripper, now)
        if self.motion_enabled:
            self._arm_target = joints
        if self._elapsed == waypoint:
            self._next_waypoint += 1
        return self._next_waypoint == len(trajectory.times)

    def _home(self, arm: ArmState, gripper: GripperState, now: float) -> None:
        """执行预检后的回位轨迹，等待关节和夹爪同时稳定."""
        trajectory = self._trajectory
        if trajectory is None:
            raise RuntimeError("没有完成预检的回位轨迹")
        if self._next_waypoint < len(trajectory.times) and not self._advance_trajectory(
            arm, now, homing=True
        ):
            return
        target = trajectory.joints[-1]
        if not self._sent_this_tick:
            self._send(target, None, now)
        ready = self._at_target(arm, target) and gripper.openness >= self.cfg.execution.start_gripper_min
        if self._stable(ready, now, self.cfg.execution.start_settle_s):
            self._trajectory = self._arm_target = self._last_send = None
            self._transition(State.READY, now)

    def _step_active(self, result: ActionChunk | None, now: float) -> None:
        """验证最新反馈后推进当前阶段，执行间隙保持机械臂目标."""
        arm, gripper = self._feedback(now)
        tcp = self._tcp(arm.joints)
        phase_age = now - self._phase_started
        if self.state is State.HOME_PLANNING:
            if phase_age > self.cfg.execution.inference_timeout_s:
                raise RuntimeError("回位轨迹预检超时")
            self._poll_home(now)
        elif self.state is State.HOMING:
            if phase_age > self.cfg.execution.home_timeout_s:
                raise RuntimeError("回位超时")
            self._home(arm, gripper, now)
        elif self.state is State.STARTING:
            ready = self._at_target(arm, self.assets.initial_joints) and (
                gripper.openness >= self.cfg.execution.start_gripper_min
            )
            if phase_age > self.cfg.execution.settle_timeout_s:
                raise RuntimeError("起始状态未稳定")
            if self._stable(ready, now, self.cfg.execution.start_settle_s):
                self._begin_episode(arm, now)
        elif self.state is State.OBSERVING:
            self._observe(now)
        elif self.state in (State.INFERRING, State.PLANNING):
            self._check_drift(tcp)
            if phase_age > self.cfg.execution.inference_timeout_s:
                raise RuntimeError(f"阶段 {self.state} 超时")
            if self.state is State.INFERRING and result is not None:
                self._plan(result, arm, gripper, now)
            elif self.state is State.PLANNING:
                self._poll_plan(arm, now)
        elif self.state is State.EXECUTING:
            if not self._trajectory_started:
                self._check_drift(tcp)
            if self._advance_trajectory(arm, now):
                self._transition(State.SETTLING, now)
        elif self.state is State.SETTLING:
            if phase_age > self.cfg.execution.settle_timeout_s:
                raise RuntimeError("动作段末点未稳定")
            if self._arm_target is None or self._trajectory is None:
                raise RuntimeError("没有可保持的末点")
            # 空跑不下发夹爪指令，实机必须等各在线侧实测到达末点。
            grip_ready = not self.motion_enabled or self.gripper.at_target(
                gripper, float(self._trajectory.grippers[-1])
            )
            if self._stable(
                self._at_target(arm, self._arm_target) and grip_ready, now, self.cfg.execution.settle_s
            ):
                self._chunks_executed += 1
                self.events.write(
                    "chunk_completed",
                    episode_id=self.episode_id,
                    request_id=self.request_id,
                    trajectory_duration_s=self._elapsed,
                    monotonic=now,
                )
                if self.max_chunks is not None and self._chunks_executed >= self.max_chunks:
                    self.stop(now, reason="chunk_limit")
                else:
                    self._not_before = now
                    self._window = self._trajectory = None
                    self._transition(State.OBSERVING, now)
        if (
            self.state in _EPISODE_STATES - {State.EXECUTING}
            and self._arm_target is not None
            and self._last_send is not None
            and not self._sent_this_tick
        ):
            self._send(self._arm_target, None, now)

    def tick(self, now: float) -> None:
        """推进一次控制周期，未处理的错误进入故障状态."""
        if self.state in (State.CLOSED, State.FAULT):
            return
        self._tick_wall_started = time.monotonic()
        self._sent_this_tick = False
        try:
            if self._gripper_server is not None:
                self._gripper_server.check_alive()
            if self._planning_worker is not None:
                self._planning_worker.check_alive()
            if self.state in _ACTIVE_STATES and self._last_tick is not None:
                if now - self._last_tick > self.cfg.safety.max_send_gap_s:
                    raise RuntimeError("控制周期超时")
                if self._last_send is not None and now - self._last_send > self.cfg.safety.max_send_gap_s:
                    raise RuntimeError("机械臂命令间隔超时")
            if self._last_tick is None:
                self._phase_started = now
            self._last_tick = now
            self._read_feedback(now)
            result = self._poll_inference(now)
            if (
                self.state is not State.HOME_PLANNING
                and self._plan_key is None
                and self._future is not None
                and self._future.done()
            ):
                self._future = None
                self.events.write("discard_plan", reason="stale_request", monotonic=now)
            if self.state in _ACTIVE_STATES:
                self._step_active(result, now)
            if self.motion_enabled and self._gripper_enabled and not self._stop_gripper:
                actual = self._dispatch_time(now)
                if (
                    self._last_gripper_send is not None
                    and actual - self._last_gripper_send > self.cfg.safety.max_send_gap_s
                ):
                    raise RuntimeError("夹爪命令间隔超时")
                if self._gripper_state is None or not self._fresh(
                    self._gripper_state.timestamp, self.cfg.gripper.feedback_timeout_s, actual
                ):
                    raise RuntimeError("保持夹爪时反馈超时")
                self.gripper.tick(actual)
                self._last_gripper_send = actual
        except Exception as exc:
            self._fault(now, str(exc))

    def close(self, now: float | None = None) -> None:
        """停臂后关闭所有资源，并记录最后一次真实夹爪反馈."""
        if self.state is State.CLOSED:
            return
        if now is None:
            now = time.monotonic()
        errors = self._halt_outputs()
        try:
            if self._resources:
                self._invalidate()
            self._end_episode(now, "shutdown")
        except Exception as exc:
            errors.append(f"关闭任务失败 {exc}")
        self.state = State.CLOSED
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        for name, component in reversed(self._resources):
            try:
                component.close()
            except Exception as exc:
                errors.append(f"{name} 关闭失败 {exc}")
        self._resources.clear()
        try:
            self.events.write("closed", errors=errors, final_gripper=self._gripper_state, monotonic=now)
        except Exception as exc:
            errors.append(f"记录关闭状态失败 {exc}")
        for error in errors:
            logger.error("%s", error)
        if errors:
            raise RuntimeError("。".join(errors))
