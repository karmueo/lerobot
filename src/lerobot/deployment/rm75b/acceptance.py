"""RM75B 基础验收：离线规划或现场逐项检查，不启动模型."""

import argparse
import json
import math
import os
import select
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, DeploymentAssets, DeploymentConfig, load_assets, load_config
from .geometry import rotation_distance
from .kinematics import ArmKinematics
from .planning_worker import (
    AXES,
    AXIS_DISTANCES_MM,
    SEGMENT_DISTANCE_MM,
    PlanningPort,
    PlanningWorker,
    axis_plan as axis_plan,
    reverse_plan as reverse_plan,
)
from .telemetry import RunLog, build_manifest, json_value
from .trajectory import TrajectoryExecutor
from .types import (
    ArmInterface,
    ArmState,
    EventLogger,
    FloatArray,
    GripperInterface,
    GripperState,
    JointTrajectory,
)

POSITION_TOLERANCE_M = 0.0005
ANGLE_TOLERANCE_RAD = math.radians(0.2)
SPEED_PERCENT_CHOICES = (5, 20)
DEFAULT_SPEED_PERCENT = 5
TEST_FPS = 4.0
OBSERVATION_HOLD_S = 2.0


def acceptance_planner(
    cfg: DeploymentConfig,
    assets: DeploymentAssets,
    kin: ArmKinematics,
    speed_percent: int = DEFAULT_SPEED_PERCENT,
) -> TrajectoryExecutor:
    """按所选档位缩放验收时间轴，关节速度仍受部署配置上限约束."""
    if speed_percent not in SPEED_PERCENT_CHOICES:
        raise ValueError("验收速度只允许 5% 或 20%")
    execution = replace(
        cfg.execution,
        fps=TEST_FPS * speed_percent / DEFAULT_SPEED_PERCENT,
        velocity_scale=min(speed_percent / 100, cfg.execution.velocity_scale),
    )
    return TrajectoryExecutor(kin, assets.tcp_offset, execution, cfg.safety)


class BasicAcceptance:
    """单线程控制周期只处理一个步骤，等待现场输入期间继续监测反馈."""

    def __init__(
        self,
        cfg: DeploymentConfig,
        assets: DeploymentAssets,
        planner: TrajectoryExecutor,
        arm: ArmInterface,
        gripper: GripperInterface,
        events: EventLogger,
        *,
        planning_worker: PlanningPort,
        distance_mm: int = 5,
        speed_percent: int = DEFAULT_SPEED_PERCENT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """保存驱动与预检器，构造时不发指令."""
        self.cfg = cfg
        self.assets = assets
        self.planner = planner
        self.arm = arm
        self.gripper = gripper
        self.events = events
        if distance_mm not in AXIS_DISTANCES_MM:
            raise ValueError("轴向验收距离只允许 5 或 50 mm")
        self.distance_mm = distance_mm
        if speed_percent not in SPEED_PERCENT_CHOICES:
            raise ValueError("验收速度只允许 5% 或 20%")
        self.speed_percent = speed_percent
        self._planning_worker = planning_worker
        self._clock = clock
        self.period = 1 / cfg.execution.control_hz
        self.phase = "ready"
        self.homed = False
        self.baseline: FloatArray | None = None
        self.moved = False
        self.gripper_active = False
        self.last_tick: float | None = None
        self.stable_since: float | None = None
        self.started = 0.0
        self.target_grip = 0.0
        self.offset: JointTrajectory | None = None
        self.return_path: JointTrajectory | None = None
        self.trajectory: JointTrajectory | None = None
        self.elapsed = 0.0
        self.next_point = 1
        self.last_gripper: GripperState | None = None
        self._pending: Future[JointTrajectory | None] | None = None
        self._planned_step: str | None = None
        self._plan_seed: FloatArray | None = None
        self._last_arm_send: float | None = None
        self._last_gripper_send: float | None = None

    def _tcp(self, joints: FloatArray) -> FloatArray:
        """拒绝超限的实测或目标关节及其基座系 TCP."""
        if not self.planner.kinematics.within_limits(joints):
            raise RuntimeError("关节超出 URDF 限位")
        tcp = self.planner.kinematics.forward(joints) @ self.assets.tcp_offset
        if (
            not np.isfinite(tcp).all()
            or np.any(tcp[:3, 3] < self.cfg.safety.workspace_min)
            or np.any(tcp[:3, 3] > self.cfg.safety.workspace_max)
        ):
            raise RuntimeError("TCP 超出工作空间")
        return tcp

    def _feedback(self, now: float) -> tuple[ArmState, GripperState]:
        """每次发指令前确认新鲜且健康的两套反馈."""
        arm, grip = self.arm.read_state(), self.gripper.read_state()
        if (
            arm is None
            or arm.errors
            or not -self.period <= now - arm.timestamp <= self.cfg.safety.feedback_timeout_s
            or arm.velocities.shape != (7,)
            or not np.isfinite(arm.velocities).all()
        ):
            raise RuntimeError("机械臂反馈超时或故障")
        if (
            grip is None
            or not -self.period <= now - grip.timestamp <= self.cfg.gripper.feedback_timeout_s
            or not 0 <= grip.openness <= 1
        ):
            raise RuntimeError("夹爪反馈超时或故障")
        self._tcp(arm.joints)
        self.last_gripper = grip
        return arm, grip

    def _stationary(self, arm: ArmState) -> bool:
        """使用部署规定的关节速度阈值判断静止."""
        return bool(np.max(np.abs(arm.velocities)) <= self.cfg.execution.stationary_velocity_rad_s)

    def _stable(self, qualifying: bool, now: float, duration: float) -> bool:
        """容差必须连续满足，偏离时重新计时."""
        if not qualifying:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = now
        return now - self.stable_since >= duration

    def _phase(self, phase: str, now: float) -> None:
        """切换独立步骤并重置稳定计时."""
        self.phase, self.started, self.stable_since = phase, now, None
        self.events.write("phase", state=phase, monotonic=now)

    def _start_path(self, plan: JointTrajectory, phase: str, now: float) -> None:
        """记录完整预检轨迹，下一控制周期才会发首个目标."""
        self.trajectory, self.elapsed, self.next_point = plan, 0.0, 0
        if phase != "returning":
            self._last_arm_send = None
        self.events.write("trajectory", phase=phase, trajectory=plan)
        self._phase(phase, now)

    def _request_plan(self, step: str, joints: FloatArray, openness: float, now: float) -> None:
        """冻结实际状态并提交唯一的预检请求."""
        self._plan_seed = joints.copy()
        self._planned_step = step
        self._pending = self._planning_worker.submit(step, self._plan_seed, openness, self.distance_mm)
        submitted = self._clock()
        self.events.write("planning_request", step=step, monotonic=submitted, submit_s=submitted - now)
        self._phase("planning", submitted)

    def _dispatch_time(self, tick_started: float, previous_send: float | None) -> float:
        """核对当前周期与连续输出的发送截止时刻."""
        now = self._clock()
        limit = self.cfg.safety.max_send_gap_s
        if now - tick_started > limit or (previous_send is not None and now - previous_send > limit):
            raise RuntimeError("基础验收发送截止时刻已过")
        return now

    def _dispatch_feedback(
        self, now: float, previous_send: float | None
    ) -> tuple[float, ArmState, GripperState]:
        """阻塞或日志耗时后重新读取两套反馈，再核对发送时间."""
        arm, grip = self._feedback(self._dispatch_time(now, previous_send))
        actual = self._dispatch_time(now, previous_send)
        if (
            actual - arm.timestamp > self.cfg.safety.feedback_timeout_s
            or actual - grip.timestamp > self.cfg.gripper.feedback_timeout_s
        ):
            raise RuntimeError("发送前设备反馈已过期")
        return actual, arm, grip

    def _max_start_delta(self) -> FloatArray:
        """轨迹首点不能超过一控制周期允许的关节位移."""
        return self.planner.kinematics.velocities * self.planner.execution.velocity_scale * self.period

    def _validate_start(self, arm: ArmState, seed: FloatArray) -> None:
        """发送首点前检查静止及轨迹起点，限制低速入轨步长."""
        if not self._stationary(arm) or np.any(np.abs(arm.joints - seed) > self._max_start_delta()):
            raise RuntimeError("实测关节偏离预检轨迹起点")

    def _poll_plan(self, now: float) -> str | None:
        """只有健康反馈仍与预检起点一致才开始下发."""
        pending, step, seed = self._pending, self._planned_step, self._plan_seed
        assert pending is not None and step is not None and seed is not None
        if now - self.started > self.cfg.execution.settle_timeout_s:
            raise RuntimeError("基础验收预检超时")
        if not pending.done():
            return None
        plan = pending.result()
        self._pending = None
        actual, arm, grip = self._dispatch_feedback(now, None)
        self._validate_start(arm, seed)
        if step == "home":
            self.events.write("home_prepared", joints=self.assets.initial_joints, speed=self.speed_percent)
            actual, arm, _ = self._dispatch_feedback(now, None)
            self._validate_start(arm, seed)
            self.moved = True
            self.arm.home(self.assets.initial_joints, speed=self.speed_percent)
            self.events.write(
                "home_command", joints=self.assets.initial_joints, speed=self.speed_percent, monotonic=actual
            )
            self._phase("homing", actual)
            return "回位中"
        if plan is None or grip.openness < self.cfg.execution.start_gripper_min:
            raise RuntimeError("轴向计划无效或夹爪离开张开状态")
        self.offset = plan
        # 返程与去程沿同一组预检过的插值点，以反向时间轴保持各段速度。
        self.return_path = reverse_plan(plan)
        self._start_path(plan, "outbound", actual)
        return f"开始沿 base +{step} 移动 {self.distance_mm} mm"

    def command(self, command: str, now: float) -> str:
        """接收一条明确命令，出站稳定后需输入 back 才返程."""
        if command == "stop":
            self.halt()
            self._phase("stopped", now)
            return "已停止，本会话禁止继续运动"
        if self.phase == "offset" and command == "back":
            if now - self.started < OBSERVATION_HOLD_S:
                return "已到末点，至少停留 2 秒后才能返回"
            assert self.offset is not None and self.return_path is not None
            _, arm, _ = self._dispatch_feedback(now, self._last_arm_send)
            self._validate_offset(arm, self.offset)
            self._start_path(self.return_path, "returning", now)
            return "开始沿原轨迹返回基线"
        if self.phase != "ready":
            return f"当前 {self.phase}，请等待或输入 stop"
        if command not in ("home", "open", "close", *AXES):
            return "命令：home、open、close、x、y、z、back、stop、quit"
        arm, grip = self._feedback(now)
        if not self._stationary(arm):
            return "机械臂尚未静止"
        at_home = bool(
            np.max(np.abs(arm.joints - self.assets.initial_joints))
            <= self.cfg.execution.start_joint_tolerance_rad
        )
        if command == "home":
            # 基础验收的起点已在初始位附近，拒绝未知的大范围回位路径。
            if not at_home:
                return "当前位置不在配置初始位附近，拒绝基础验收回位"
            self.homed = False
            self.baseline = None
            self._request_plan("home", arm.joints, grip.openness, now)
            return "预检回位路径"
        if not (self.homed and at_home):
            return "请先完成回位"
        if command in ("open", "close"):
            self.target_grip = float(command == "open")
            self.gripper.set_target(self.target_grip)
            self.events.write("gripper_target_accepted", openness=self.target_grip)
            self.gripper_active = True
            self._phase(f"gripper_{command}", now)
            return "夹爪执行中"
        if grip.openness < self.cfg.execution.start_gripper_min:
            return "轴向检查前须确认夹爪张开"
        baseline = self.baseline
        assert baseline is not None
        baseline_tcp = self._tcp(baseline)
        current_tcp = self._tcp(arm.joints)
        if (
            np.linalg.norm(current_tcp[:3, 3] - baseline_tcp[:3, 3]) > POSITION_TOLERANCE_M
            or rotation_distance(current_tcp[:3, :3], baseline_tcp[:3, :3]) > ANGLE_TOLERANCE_RAD
        ):
            return "实测 TCP 偏离冻结的回位基线"
        self._request_plan(command, baseline, grip.openness, now)
        return f"预检 base +{command} 的 {self.distance_mm} mm 轨迹"

    def _track(self, arm: ArmState, tcp: FloatArray, target: FloatArray, plan: JointTrajectory) -> bool:
        """按实测 FK 核对末点的绝对位置和姿态误差."""
        goal = plan.tcp_targets[-1]
        return bool(
            self._stationary(arm)
            and np.max(np.abs(arm.joints - target)) <= self.cfg.execution.start_joint_tolerance_rad
            and np.linalg.norm(tcp[:3, 3] - goal[:3, 3]) <= POSITION_TOLERANCE_M
            and rotation_distance(tcp[:3, :3], goal[:3, :3]) <= ANGLE_TOLERANCE_RAD
        )

    def _validate_offset(self, arm: ArmState, plan: JointTrajectory) -> None:
        """以发送前的最新反馈拒绝末点漂移，不发纠偏或返程命令."""
        if not self._track(arm, self._tcp(arm.joints), plan.joints[-1], plan):
            raise RuntimeError("等待现场测量期间偏离末点")

    def tick(self, now: float) -> str | None:
        """更新反馈、限速下发与稳定判定，任何失效由上层停机."""
        if self.phase == "stopped":
            return None
        self._planning_worker.check_alive()
        gap = self.period if self.last_tick is None else now - self.last_tick
        if (self.phase != "ready" or self.gripper_active) and gap > self.cfg.safety.max_send_gap_s:
            raise RuntimeError("基础验收控制周期超时")
        self.last_tick = now
        arm, grip = self._feedback(now)
        tcp = self._tcp(arm.joints)
        self.events.write("feedback", monotonic=now, arm=arm, gripper=grip, tcp=tcp)
        if self.gripper_active:
            actual, _, _ = self._dispatch_feedback(now, self._last_gripper_send)
            self.gripper.tick(actual)
            self._last_gripper_send = actual
            self.events.write("gripper_tick", monotonic=actual)
        if self.phase == "planning":
            return self._poll_plan(now)
        age = now - self.started
        execution = self.cfg.execution
        if self.phase == "homing":
            if age > execution.home_timeout_s:
                raise RuntimeError("回位超时")
            reached = self._stationary(arm) and (
                np.max(np.abs(arm.joints - self.assets.initial_joints)) <= execution.start_joint_tolerance_rad
            )
            if self._stable(bool(reached), now, execution.start_settle_s):
                self.homed = True
                self.baseline = arm.joints.copy()
                self.events.write("baseline", joints=self.baseline, tcp=tcp)
                return self._completed("home", arm, grip, tcp, now)
        if self.phase in ("gripper_open", "gripper_close"):
            if age > execution.settle_timeout_s:
                raise RuntimeError(f"{self.phase} 开度未到目标，反馈 {grip.openness:.4f}")
            if self._stable(abs(grip.openness - self.target_grip) <= 0.05, now, execution.settle_s):
                return self._completed(self.phase, arm, grip, tcp, now)
        if self.phase in ("outbound", "returning", "offset"):
            plan = self.trajectory
            assert plan is not None
            if self.phase != "offset" and age > plan.duration + execution.settle_timeout_s:
                raise RuntimeError("轴向轨迹或末点稳定超时")
            first_point = self.next_point == 0
            if self.next_point < len(plan.times):
                point_time = float(plan.times[self.next_point])
                self.elapsed = min(point_time, self.elapsed + min(gap, self.period))
                if self.elapsed == point_time:
                    self.next_point += 1
            joints, _ = self.planner.sample(plan, self.elapsed)
            # 真实控制节拍会偏离预检网格，先检查本次实际下发的关节目标。
            self._tcp(joints)
            self.events.write("joint_command_prepared", joints=joints, elapsed=self.elapsed)
            actual, dispatch_arm, dispatch_grip = self._dispatch_feedback(now, self._last_arm_send)
            if dispatch_grip.openness < self.cfg.execution.start_gripper_min:
                raise RuntimeError("轴向运动期间夹爪离开张开状态")
            if first_point:
                self._validate_start(dispatch_arm, plan.joints[0])
            if self.phase == "offset":
                self._validate_offset(dispatch_arm, plan)
            self.moved = True
            self.arm.send_joint_target(joints)
            self._last_arm_send = actual
            self.events.write(
                "joint_command", joints=joints, elapsed=self.elapsed, applied=True, monotonic=actual
            )
            if (
                self.phase != "offset"
                and self.next_point == len(plan.times)
                and self._stable(self._track(arm, tcp, plan.joints[-1], plan), now, execution.settle_s)
            ):
                if self.phase == "returning":
                    return self._completed("return", arm, grip, tcp, now)
                actual_delta = tcp[:3, 3] - plan.tcp_targets[0, :3, 3]
                self.events.write(
                    "completed", step="outbound", arm=arm, gripper=grip, tcp=tcp, delta_m=actual_delta
                )
                self._phase("offset", now)
                return (
                    f"到达末点，实测位移 {np.round(1000 * actual_delta, 3).tolist()} mm。至少 2 秒后输入 back"
                )
        return None

    def _completed(self, step: str, arm: ArmState, grip: GripperState, tcp: FloatArray, now: float) -> str:
        """记录稳定后的实测值，然后等待下一条独立指令."""
        delta = None if self.baseline is None else tcp[:3, 3] - self._tcp(self.baseline)[:3, 3]
        self.events.write("completed", step=step, arm=arm, gripper=grip, tcp=tcp, delta_m=delta)
        self._phase("ready", now)
        return f"{step} 完成。TCP={np.round(tcp[:3, 3], 6).tolist()}，夹爪={grip.openness:.4f}"

    def halt(self) -> None:
        """结束所有后续下发；机械臂有过运动时请求停臂."""
        self.phase = "stopped"
        self.gripper_active = False
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        if self.moved:
            self.arm.stop()
            self.moved = False

    def close_planner(self) -> None:
        """停臂后释放独立规划进程和已作废的任务."""
        self._planning_worker.close()


def _interactive(session: BasicAcceptance) -> None:
    """终端只用非阻塞轮询，保持控制循环和停止响应."""
    fd = sys.stdin.fileno()
    pending = b""
    next_tick = time.monotonic()
    deadline = next_tick + session.cfg.safety.feedback_timeout_s
    while True:
        now = time.monotonic()
        try:
            session._feedback(now)
        except RuntimeError:
            if now >= deadline:
                raise
            time.sleep(0.02)
        else:
            break
    print("已连接并取得反馈。命令 home、close、open、x、y、z、back、stop、quit。", flush=True)
    while session.phase != "stopped":
        if select.select([fd], [], [], max(0.0, next_tick - time.monotonic()))[0]:
            data = os.read(fd, 4096)
            if not data:
                return
            lines = (pending + data).split(b"\n")
            pending = lines.pop()
            if len(pending) > 4096:
                raise RuntimeError("终端命令过长")
            commands = [line.decode(errors="replace").strip().lower() for line in lines if line.strip()]
            if "quit" in commands:
                return
            if "stop" in commands:
                commands = ["stop"]
            if len(commands) > 1:
                print("请逐条发送命令。", flush=True)
            elif commands:
                print(session.command(commands[0], time.monotonic()), flush=True)
        now = time.monotonic()
        if now >= next_tick:
            message = session.tick(now)
            if message is not None:
                print(message, flush=True)
            next_tick = now + session.period


def _execute(
    cfg: DeploymentConfig,
    assets: DeploymentAssets,
    planner: TrajectoryExecutor,
    output_dir: Path,
    distance_mm: int,
    speed_percent: int,
) -> int:
    """使用现场监督模式、独占锁和分步显式命令执行验收."""
    from .arm import ArmDriver
    from .cli import _session_signals, _SessionExit, conflicting_processes, deployment_lock
    from .gripper import GripperDriver

    if not sys.stdin.isatty():
        raise RuntimeError("基础验收运动须在交互终端执行")
    with _session_signals() as shutdown, deployment_lock(cfg.logging.lock_path):
        conflicts = conflicting_processes()
        if conflicts:
            raise RuntimeError(f"发现其他控制端：{conflicts}")
        manifest = build_manifest(cfg, assets, "basic-acceptance")
        manifest["acceptance_parameters"] = {
            "axis_frame": "base_link",
            "axis_distance_m": distance_mm / 1000,
            "axis_segment_distance_m": SEGMENT_DISTANCE_MM / 1000,
            "axis_waypoint_rate_hz": planner.execution.fps,
            "axis_velocity_scale": planner.execution.velocity_scale,
            "control_hz": planner.execution.control_hz,
            "requested_speed_percent": speed_percent,
            "home_speed_percent": speed_percent,
            "position_tolerance_m": POSITION_TOLERANCE_M,
            "angle_tolerance_rad": ANGLE_TOLERANCE_RAD,
            "minimum_observation_hold_s": OBSERVATION_HOLD_S,
            "planning_backend": "spawn",
            "planning_numeric_threads": 1,
        }
        with RunLog(output_dir, manifest) as events:
            print(f"基础验收记录 {events.directory}", flush=True)
            arm = ArmDriver(cfg.arm, motion_enabled=True)
            gripper = GripperDriver(cfg.gripper, motion_enabled=True)
            worker = PlanningWorker(
                cfg.arm.urdf, assets.tcp_offset, assets.initial_joints, planner.execution, cfg.safety
            )
            session = BasicAcceptance(
                cfg,
                assets,
                planner,
                arm,
                gripper,
                events,
                planning_worker=worker,
                distance_mm=distance_mm,
                speed_percent=speed_percent,
            )
            result = 0
            try:
                worker.start()
                events.write("planning_ready", pid=worker.pid)
                arm.connect()
                gripper.connect()
                _interactive(session)
            except _SessionExit:
                # 与正常退出使用相同顺序，先停臂再断开设备。
                pass
            except (Exception, KeyboardInterrupt) as error:
                shutdown.cleaning = True
                stop_error = None
                try:
                    session.halt()
                except Exception as cleanup_error:
                    stop_error = str(cleanup_error)
                reason = str(error) or type(error).__name__
                events.write("fault", reason=reason, stop_error=stop_error)
                print(f"基础验收故障：{reason}", file=sys.stderr, flush=True)
                result = 1
            finally:
                shutdown.cleaning = True
                errors = []
                for cleanup in (session.halt, gripper.close, arm.close, session.close_planner):
                    try:
                        cleanup()
                    except Exception as error:
                        errors.append(str(error))
                events.write("closed", errors=errors, final_gripper=session.last_gripper)
                if errors:
                    raise RuntimeError(f"基础验收关闭失败：{errors}")
        return result if shutdown.signum is None else 128 + shutdown.signum


def main() -> int:
    """默认仅计算预期轨迹，显式执行模式需要现场监督."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "docs/umi/rm75b_deployment.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/deploy/rm75b/acceptance")
    parser.add_argument("--axis-distance-mm", type=int, choices=AXIS_DISTANCES_MM, default=5)
    parser.add_argument(
        "--speed-percent", type=int, choices=SPEED_PERCENT_CHOICES, default=DEFAULT_SPEED_PERCENT
    )
    parser.add_argument("--execute", action="store_true", help="现场有人监督时，允许分步执行动作")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assets = load_assets(cfg)
    planner = acceptance_planner(cfg, assets, ArmKinematics.from_urdf(cfg.arm.urdf), args.speed_percent)
    if args.execute:
        return _execute(cfg, assets, planner, args.output_dir, args.axis_distance_mm, args.speed_percent)
    plans = {}
    for axis in AXES:
        out = axis_plan(planner, assets.initial_joints, 1.0, axis, args.axis_distance_mm)
        plans[axis] = {"outbound": out, "return": reverse_plan(out)}
    print(json.dumps(json_value(plans), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
