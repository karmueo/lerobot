"""在下发前完成 TCP 转换、整段 IK 和统一限速的关节轨迹预检."""

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .geometry import invert, pose_to_transform, rotation_distance, validate_transform
from .kinematics import ArmKinematics
from .types import FloatArray, JointTrajectory

if TYPE_CHECKING:
    from .config import ExecutionConfig, SafetyConfig

# 单块最多预检 500 个控制点。固定 50 Hz 下相当于 10 秒，
# 避免极低限速使预检线程长期占用并延迟退出。
MAX_VALIDATION_TICKS = 500
# 回位同时受部署限速约束，且最多使用 URDF 关节速度的 10%。
HOME_VELOCITY_SCALE = 0.10


class ActionWorkspaceError(ValueError):
    """模型动作目标超出工作空间，整段计划尚未获准执行."""


class TrajectoryExecutor:
    """纯计算规划器，运行层独占驱动调用与控制时序."""

    def __init__(
        self,
        kinematics: ArmKinematics,
        tcp_offset: FloatArray,
        execution: "ExecutionConfig",
        safety: "SafetyConfig",
    ) -> None:
        """保存运动学、工具外参和执行约束，不访问任何设备."""
        self.kinematics = kinematics
        self.tcp_offset = validate_transform(tcp_offset)
        self._link7_from_tcp_inverse = invert(self.tcp_offset)
        self.execution = execution
        self.safety = safety
        self._workspace_min = np.asarray(safety.workspace_min, dtype=np.float64)
        self._workspace_max = np.asarray(safety.workspace_max, dtype=np.float64)

    def _workspace(self, tcp: FloatArray, label: str, *, error_type: type[ValueError] = ValueError) -> None:
        """工作空间约束明确作用于基座系 TCP 点."""
        position = tcp[:3, 3]
        if (
            not np.isfinite(tcp).all()
            or np.any(position < self._workspace_min)
            or np.any(position > self._workspace_max)
        ):
            raise error_type(f"{label} 超出基座系 TCP 工作空间：{position.tolist()}")

    def _step(self, previous: FloatArray, target: FloatArray, index: int) -> None:
        """防止模型目标在段首或相邻路点间出现意外位移和转角."""
        if (
            float(np.linalg.norm(target[:3, 3] - previous[:3, 3])) > self.safety.max_translation_m
            or rotation_distance(previous[:3, :3], target[:3, :3]) > self.safety.max_rotation_rad
        ):
            raise ValueError(f"动作点 {index} 超出 TCP 步长限制")

    def _check_segment(self, start: FloatArray, end: FloatArray, ticks: int, label: str) -> None:
        """在执行网格上检查插值 FK，包括实际关节终点."""
        for tick in range(1, ticks + 1):
            joints = end if tick == ticks else start + (end - start) * (tick / ticks)
            self._workspace(self.kinematics.forward(joints) @ self.tcp_offset, label)

    def plan_home(self, start: FloatArray, end: FloatArray) -> JointTrajectory:
        """按关节空间直线预检回位，返回控制循环实际采样的轨迹."""
        if not (self.kinematics.within_limits(start) and self.kinematics.within_limits(end)):
            raise ValueError("回位起点或终点超出 URDF 关节限位")
        start_tcp = self.kinematics.forward(start) @ self.tcp_offset
        self._workspace(start_tcp, "回位起点")
        velocity = self.kinematics.velocities * min(HOME_VELOCITY_SCALE, self.execution.velocity_scale)
        with np.errstate(over="ignore", divide="ignore"):
            required_ticks = float(np.max(np.abs(end - start) / velocity)) * self.execution.control_hz
        if not math.isfinite(required_ticks):
            raise ValueError("回位限速后所需时长不是有限值")
        ticks = max(1, math.ceil(required_ticks))
        duration = ticks / self.execution.control_hz
        if duration + self.execution.start_settle_s >= self.execution.home_timeout_s:
            raise ValueError("回位轨迹和稳定等待超出回位超时上限")
        self._check_segment(start, end, ticks, "回位路径")
        return JointTrajectory(
            np.array([0.0, duration]),
            np.stack([start, end]),
            np.ones(2),
            np.stack([start_tcp, self.kinematics.forward(end) @ self.tcp_offset]),
        )

    def plan(
        self, actions: ArrayLike, anchor: FloatArray, current_joints: FloatArray, current_gripper: float
    ) -> JointTrajectory:
        """使用固定回合原点还原所有目标，完整预检通过后返回九个路点."""
        values = np.asarray(actions, dtype=np.float64)
        if values.shape != (self.execution.n_action_steps, 7) or not np.isfinite(values).all():
            raise ValueError("模型动作须为完整且有限的八步七维目标")
        if not 0.0 <= current_gripper <= 1.0 or np.any((values[:, 6] < 0.0) | (values[:, 6] > 1.0)):
            raise ValueError("当前与目标夹爪开度须在 [0, 1] 内")
        frozen_anchor = validate_transform(anchor)
        joints = np.asarray(current_joints, dtype=np.float64)
        if not self.kinematics.within_limits(joints):
            raise ValueError("当前七轴关节超出 URDF 限位")
        previous_tcp = self.kinematics.forward(joints) @ self.tcp_offset
        self._workspace(previous_tcp, "实测起点")
        waypoints, tcp_targets = [joints.copy()], [previous_tcp]
        for index, action in enumerate(values, start=1):
            target = frozen_anchor @ pose_to_transform(action[:6])
            self._workspace(target, f"动作点 {index}", error_type=ActionWorkspaceError)
            self._step(previous_tcp, target, index)
            solution = self.kinematics.solve_ik(target @ self._link7_from_tcp_inverse, waypoints[-1])
            if solution is None or not self.kinematics.within_limits(solution):
                raise ValueError(f"动作点 {index} IK 不可达或超出关节限位")
            if np.any(np.abs(solution - waypoints[-1]) > self.safety.max_joint_step_rad):
                raise ValueError(f"动作点 {index} 出现关节跳变")
            waypoints.append(solution)
            tcp_targets.append(target)
            previous_tcp = target

        joint_array = np.stack(waypoints)
        velocity = self.kinematics.velocities * self.execution.velocity_scale
        if not np.isfinite(velocity).all() or np.any(velocity <= 0):
            raise ValueError("限速后的关节速度必须为正且有限")
        with np.errstate(over="ignore"):
            required_s = float(np.max(np.abs(np.diff(joint_array, axis=0)) / velocity))
        # 所有段采用相同倍率，向上对齐控制节拍以确保每个路点都能实际下发。
        required_ticks = max(1 / self.execution.fps, required_s) * self.execution.control_hz
        if not math.isfinite(required_s) or not math.isfinite(required_ticks):
            raise ValueError("轨迹限速后所需时长不是有限值")
        max_ticks_per_step = MAX_VALIDATION_TICKS // self.execution.n_action_steps
        if required_ticks > max_ticks_per_step:
            raise ValueError(f"轨迹限速后超过整块 {MAX_VALIDATION_TICKS} 个控制周期的预检上限")
        ticks = math.ceil(required_ticks)
        segment_s = ticks / self.execution.control_hz
        for index, (left, right) in enumerate(zip(joint_array[:-1], joint_array[1:], strict=True), start=1):
            self._check_segment(left, right, ticks, f"插值段 {index}")

        return JointTrajectory(
            np.arange(len(waypoints), dtype=np.float64) * segment_s,
            joint_array,
            np.concatenate(([current_gripper], values[:, 6])),
            np.stack(tcp_targets),
        )

    @staticmethod
    def sample(trajectory: JointTrajectory, elapsed: float) -> tuple[FloatArray, float]:
        """按同一时间轴插值关节和夹爪，端点之外保持端点值."""
        if not math.isfinite(elapsed):
            raise ValueError("轨迹采样时间必须有限")
        times = trajectory.times
        index = int(np.searchsorted(times, elapsed, side="right"))
        if index == 0:
            return trajectory.joints[0].copy(), float(trajectory.grippers[0])
        if index == len(times):
            return trajectory.joints[-1].copy(), float(trajectory.grippers[-1])
        left = index - 1
        fraction = (elapsed - times[left]) / (times[index] - times[left])
        joints = trajectory.joints[left] + fraction * (trajectory.joints[index] - trajectory.joints[left])
        grip = trajectory.grippers[left] + fraction * (trajectory.grippers[index] - trajectory.grippers[left])
        return joints, float(grip)
