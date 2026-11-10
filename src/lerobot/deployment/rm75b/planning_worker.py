"""独立 CPU 进程预检基础验收轨迹，控制循环仅检查已完成的 Future."""

import math
import multiprocessing as mp
import os
import signal
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ExecutionConfig, SafetyConfig
from .geometry import invert, transform_to_pose, validate_transform
from .kinematics import ArmKinematics
from .trajectory import ActionWorkspaceError, TrajectoryExecutor
from .types import FloatArray, JointTrajectory

AXES = {"x": 0, "y": 1, "z": 2}
AXIS_DISTANCES_MM = (5, 50)
SEGMENT_DISTANCE_MM = 5
_NUMERIC_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_START_TIMEOUT_S = 10.0
_CLOSE_WAIT_S = 0.5


def axis_plan(
    planner: TrajectoryExecutor, joints: FloatArray, grip: float, axis: str, distance_mm: int = 5
) -> JointTrajectory:
    """从冻结基线沿基座轴平移，完整预检最多十段，每段 5 mm."""
    if distance_mm not in AXIS_DISTANCES_MM:
        raise ValueError("轴向验收距离只允许 5 或 50 mm")
    anchor = planner.kinematics.forward(joints) @ planner.tcp_offset
    anchor_inverse = invert(anchor)
    actions = np.empty((planner.execution.n_action_steps, 7))
    fractions = np.linspace(0, 1, len(actions) + 1)[1:]
    segments = []
    seed = joints
    for segment in range(distance_mm // SEGMENT_DISTANCE_MM):
        # 所有模型目标沿同一基座轴直线，分段只更新 IK 种子，保持原点和姿态。
        for index, fraction in enumerate(fractions):
            target = anchor.copy()
            target[AXES[axis], 3] += (segment + fraction) * SEGMENT_DISTANCE_MM / 1000
            actions[index, :6] = transform_to_pose(anchor_inverse @ target)
            actions[index, 6] = grip
        try:
            plan = planner.plan(actions, anchor, seed, grip)
        except ValueError as error:
            raise ValueError(f"base +{axis} 第 {segment + 1} 个 5 mm 段预检失败：{error}") from error
        segments.append(plan)
        seed = plan.joints[-1]
    if len(segments) == 1:
        return segments[0]

    # 跳过段首重复点，保留不同段根据关节速度上限计算的各自时长。
    elapsed = 0.0
    times = []
    for index, plan in enumerate(segments):
        times.append(elapsed + plan.times[int(index > 0) :])
        elapsed += plan.duration
    return JointTrajectory(
        np.concatenate(times),
        np.concatenate([plan.joints[int(index > 0) :] for index, plan in enumerate(segments)]),
        np.concatenate([plan.grippers[int(index > 0) :] for index, plan in enumerate(segments)]),
        np.concatenate([plan.tcp_targets[int(index > 0) :] for index, plan in enumerate(segments)]),
    )


def reverse_plan(trajectory: JointTrajectory) -> JointTrajectory:
    """反向复用同一组已预检路点，末点为精确的出发关节位置."""
    return JointTrajectory(
        trajectory.duration - trajectory.times[::-1],
        trajectory.joints[::-1].copy(),
        trajectory.grippers[::-1].copy(),
        trajectory.tcp_targets[::-1].copy(),
    )


def precheck_motion(
    planner: TrajectoryExecutor,
    initial_joints: FloatArray,
    step: str,
    start: FloatArray,
    openness: float,
    distance_mm: int,
) -> JointTrajectory | None:
    """纯计算预检回位关节路径或完整轴向轨迹."""
    if step != "home":
        return axis_plan(planner, start, openness, step, distance_mm)
    for fraction in np.linspace(0, 1, 51):
        q = (1 - fraction) * start + fraction * initial_joints
        if not planner.kinematics.within_limits(q):
            raise ValueError("回位路径超出 URDF 关节限位")
        tcp = planner.kinematics.forward(q) @ planner.tcp_offset
        if (
            not np.isfinite(tcp).all()
            or np.any(tcp[:3, 3] < planner.safety.workspace_min)
            or np.any(tcp[:3, 3] > planner.safety.workspace_max)
        ):
            raise ValueError("回位路径超出基座系 TCP 工作空间")
    return None


class PlanningPort(Protocol):
    """基础验收状态机与规划计算之间的非阻塞接口."""

    def submit(
        self, step: str, joints: FloatArray, openness: float, distance_mm: int
    ) -> Future[JointTrajectory | None]:
        """提交一个预检任务，不等待计算完成."""
        ...

    def check_alive(self) -> None:
        """检查进程是否仍健康，异常时禁止后续运动."""
        ...

    def close(self) -> None:
        """关闭已与执行器分离的规划资源."""
        ...


@dataclass(frozen=True)
class _AxisRequest:
    """固定轴向验收的纯数组输入."""

    step: str
    joints: FloatArray
    openness: float
    distance_mm: int


@dataclass(frozen=True)
class _PolicyRequest:
    """原生策略的完整动作段，不能包含运行对象."""

    actions: FloatArray
    anchor: FloatArray
    joints: FloatArray
    openness: float


@dataclass(frozen=True)
class _HomeRequest:
    """部署回位只需当前关节，目标由启动时的配置冻结."""

    joints: FloatArray


def _worker_main(
    commands: Connection,
    results: Connection,
    urdf: Path,
    tcp_offset: FloatArray,
    initial_joints: FloatArray,
    execution: ExecutionConfig,
    safety: SafetyConfig,
) -> None:
    """子进程只持有运动学与数组，硬件和终端中断由父进程处理."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        planner = TrajectoryExecutor(ArmKinematics.from_urdf(urdf), tcp_offset, execution, safety)
        # 提前触发首次 FK 与线性求解初始化，完成后才允许父进程连接硬件。
        precheck_motion(planner, initial_joints, "home", initial_joints, 1.0, 5)
        np.linalg.solve(np.eye(6), np.ones(6))
    except Exception as error:
        results.send((0, "startup_error", str(error)))
        return
    try:
        results.send((0, "ready", None))
        while True:
            try:
                request_id, request = commands.recv()
            except EOFError:
                return
            try:
                if isinstance(request, _AxisRequest):
                    plan = precheck_motion(
                        planner,
                        initial_joints,
                        request.step,
                        request.joints,
                        request.openness,
                        request.distance_mm,
                    )
                elif isinstance(request, _PolicyRequest):
                    plan = planner.plan(request.actions, request.anchor, request.joints, request.openness)
                elif isinstance(request, _HomeRequest):
                    plan = planner.plan_home(request.joints, initial_joints)
                else:
                    raise ValueError("未知规划请求类型")
            except ActionWorkspaceError as error:
                # 跨进程保留动作目标越界的类别，运行层据此结束回合。
                results.send((request_id, "action_outside_workspace", str(error)))
            except Exception as error:
                results.send((request_id, "error", f"{type(error).__name__}: {error}"))
            else:
                results.send((request_id, "result", plan))
    except BrokenPipeError:
        # 父进程已取消会话，后续结果无需交付。
        return
    finally:
        commands.close()
        results.close()


class PlanningWorker:
    """有界单请求 spawn 规划进程，专用读线程隔离完整轨迹的 IPC 读取."""

    def __init__(
        self,
        urdf: Path,
        tcp_offset: FloatArray,
        initial_joints: FloatArray,
        execution: ExecutionConfig,
        safety: SafetyConfig,
    ) -> None:
        """保存纯规划参数，构造时不启动任何后台资源."""
        self._params = (urdf, tcp_offset.copy(), initial_joints.copy(), execution, safety)
        self._process: BaseProcess | None = None
        self._commands: Connection | None = None
        self._results: Connection | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pending: tuple[int, Future[JointTrajectory | None]] | None = None
        self._next_id = 1
        self._closing = False
        self._failure: str | None = None

    @property
    def pid(self) -> int | None:
        """提供运行记录所需的独立规划进程 PID."""
        return None if self._process is None else self._process.pid

    def start(self) -> None:
        """限制子进程数值线程并等待就绪，只能在硬件连接前调用."""
        if self._process is not None or self._closing:
            raise RuntimeError("规划子进程已启动或关闭")
        context = mp.get_context("spawn")
        request_rx, self._commands = context.Pipe(duplex=False)
        self._results, result_tx = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_main,
            args=(request_rx, result_tx, *self._params),
            name="rm75b-planning",
        )
        saved = {name: os.environ.get(name) for name in _NUMERIC_ENV}
        try:
            try:
                os.environ.update(dict.fromkeys(_NUMERIC_ENV, "1"))
                process.start()
                self._process = process
            finally:
                request_rx.close()
                result_tx.close()
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            if not self._results.poll(_START_TIMEOUT_S):
                raise RuntimeError("规划子进程就绪超时")
            request_id, status, detail = self._results.recv()
            if request_id != 0 or status != "ready":
                raise RuntimeError(f"规划子进程初始化失败：{detail}")
            self._reader = threading.Thread(
                target=self._receive, args=(self._results,), name="rm75b-planning-reader", daemon=True
            )
            self._reader.start()
        except BaseException as error:
            try:
                self.close()
            except Exception as close_error:
                error.add_note(f"关闭规划子进程也失败：{close_error}")
            raise

    def check_alive(self) -> None:
        """快速拒绝子进程退出、关闭或读线程故障."""
        if self._failure is not None:
            raise RuntimeError(self._failure)
        process = self._process
        if self._closing or process is None or not process.is_alive():
            raise RuntimeError("规划子进程未运行或已退出")

    def submit(
        self, step: str, joints: FloatArray, openness: float, distance_mm: int
    ) -> Future[JointTrajectory | None]:
        """只发送七轴数组等小请求，上一请求未返回时立即拒绝."""
        if step not in ("home", *AXES) or distance_mm not in AXIS_DISTANCES_MM:
            raise ValueError("规划请求参数无效")
        return self._submit(_AxisRequest(step, self._check_seed(joints, openness), openness, distance_mm))

    def submit_policy(
        self, actions: FloatArray, anchor: FloatArray, joints: FloatArray, openness: float
    ) -> Future[JointTrajectory | None]:
        """非阻塞提交完整模型动作段，在子进程内执行同一套 IK 与保护."""
        targets = np.asarray(actions, dtype=np.float64)
        steps = self._params[3].n_action_steps
        if targets.shape != (steps, 7) or not np.isfinite(targets).all():
            raise ValueError("模型规划请求必须包含完整的有限动作段")
        return self._submit(
            _PolicyRequest(targets, validate_transform(anchor), self._check_seed(joints, openness), openness)
        )

    def submit_home(self, joints: FloatArray) -> Future[JointTrajectory | None]:
        """在规划子进程检查完整回位轨迹，保持控制循环可响应停止."""
        return self._submit(_HomeRequest(self._check_seed(joints, 1.0)))

    @staticmethod
    def _check_seed(joints: FloatArray, openness: float) -> FloatArray:
        """验证发送载荷的七轴维度和夹爪开度."""
        q = np.asarray(joints, dtype=np.float64)
        if (
            q.shape != (7,)
            or not np.isfinite(q).all()
            or not math.isfinite(openness)
            or not 0 <= openness <= 1
        ):
            raise ValueError("规划请求的关节或夹爪反馈无效")
        return q

    def _submit(
        self, request: _AxisRequest | _PolicyRequest | _HomeRequest
    ) -> Future[JointTrajectory | None]:
        """单请求队列仅含有界小数组，控制线程无等待结果路径."""
        self.check_alive()
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("规划子进程已有未完成请求")
            request_id = self._next_id
            self._next_id += 1
            future: Future[JointTrajectory | None] = Future()
            future.set_running_or_notify_cancel()
            self._pending = (request_id, future)
        try:
            # 始终只有一条在途请求，小数组远小于 IPC 管道容量。
            assert self._commands is not None
            self._commands.send((request_id, request))
        except (OSError, EOFError) as error:
            self._fail(f"无法提交规划请求：{error}")
            raise RuntimeError("规划子进程无法接收请求") from error
        return future

    def _fail(self, reason: str) -> None:
        """保存故障并唤醒等待结果的状态机."""
        with self._lock:
            pending, self._pending = self._pending, None
            if not self._closing:
                self._failure = reason
        if pending is not None:
            pending[1].set_exception(RuntimeError(reason))

    def _receive(self, connection: Connection) -> None:
        """完整读取并反序列化大结果，控制线程不接触阻塞 recv."""
        try:
            while True:
                request_id, status, result = connection.recv()
                with self._lock:
                    if self._closing:
                        return
                    pending = self._pending
                    if pending is None or pending[0] != request_id:
                        raise RuntimeError(f"规划响应编号不匹配：{request_id}")
                    if status not in ("error", "action_outside_workspace", "result"):
                        raise RuntimeError(f"未知规划响应类型：{status}")
                    self._pending = None
                future = pending[1]
                if status == "error":
                    future.set_exception(ValueError(str(result)))
                elif status == "action_outside_workspace":
                    future.set_exception(ActionWorkspaceError(str(result)))
                else:
                    future.set_result(result)
        except Exception as error:
            self._fail(f"规划子进程响应中断：{type(error).__name__}: {error}")

    def close(self) -> None:
        """先作废请求，再以有界等待回收进程、管道和读线程."""
        if self._closing:
            return
        self._closing = True
        self._fail("规划会话已关闭")
        if self._commands is not None:
            self._commands.close()
        process = self._process
        if process is not None:
            process.join(timeout=_CLOSE_WAIT_S)
            if process.is_alive():
                process.terminate()
                process.join(timeout=_CLOSE_WAIT_S)
            if process.is_alive():
                process.kill()
                process.join(timeout=_CLOSE_WAIT_S)
            if process.is_alive():
                raise RuntimeError("规划子进程关闭超时")
            process.close()
            self._process = None
        if self._results is not None:
            self._results.close()
        if self._reader is not None and self._reader.ident is not None:
            self._reader.join(timeout=_CLOSE_WAIT_S)
            if self._reader.is_alive():
                raise RuntimeError("规划 IPC 读线程关闭超时")
