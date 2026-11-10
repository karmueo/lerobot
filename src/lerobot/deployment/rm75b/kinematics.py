"""RM75 串联链正逆运动学，参考 FastUMI_Data 的 fastumi_bringup/kinematics.py."""

import math
import xml.etree.ElementTree as ET  # nosec B405 - 读取已由部署配置核对 SHA-256 的本地 URDF。
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .geometry import pose_to_transform, transform_to_pose, validate_transform
from .types import FloatArray

JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


@dataclass(frozen=True)
class RevoluteJoint:
    """关节固定变换、单位轴、位置与速度上限，单位为弧度和弧度/秒."""

    name: str
    origin: FloatArray
    axis: FloatArray
    lower: float
    upper: float
    velocity: float


def _vector(value: ArrayLike, size: int, label: str) -> FloatArray:
    """校验精确向量维度，避免 FK 误接受截短的关节数组."""
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} 须为长度 {size} 的有限向量")
    return result


def _origin(element: ET.Element | None) -> FloatArray:
    """URDF 的固定 rpy 使用绕固定轴的 Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    xyz = _vector((element.get("xyz", "0 0 0") if element is not None else "0 0 0").split(), 3, "xyz")
    roll, pitch, yaw = _vector(
        (element.get("rpy", "0 0 0") if element is not None else "0 0 0").split(), 3, "rpy"
    )
    result = (
        pose_to_transform([0, 0, 0, 0, 0, yaw])
        @ pose_to_transform([0, 0, 0, 0, pitch, 0])
        @ pose_to_transform([0, 0, 0, roll, 0, 0])
    )
    result[:3, 3] = xyz
    return result


class ArmKinematics:
    """用一致的 URDF 链计算 base_link 到 Link7 的刚体变换."""

    def __init__(self, joints: tuple[RevoluteJoint, ...]) -> None:
        """校验七轴限制并缓存 Rodrigues 公式中的固定关节轴项."""
        if tuple(joint.name for joint in joints) != JOINT_NAMES:
            raise ValueError("RM75 需要按 joint1..joint7 排列的七轴模型")
        self.joints = joints
        self.lower = np.array([joint.lower for joint in joints])
        self.upper = np.array([joint.upper for joint in joints])
        self.velocities = np.array([joint.velocity for joint in joints])
        limits = np.stack((self.lower, self.upper, self.velocities))
        if not np.isfinite(limits).all() or np.any(self.lower >= self.upper) or np.any(self.velocities <= 0):
            raise ValueError("URDF 关节限位或速度上限无效")
        self._skews = tuple(
            np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) for x, y, z in (joint.axis for joint in joints)
        )
        self._skew_squares = tuple(skew @ skew for skew in self._skews)

    @classmethod
    def from_urdf(cls, path: Path) -> "ArmKinematics":
        """校验七个旋转关节的父子链、轴和限制并构建运动学."""
        root = ET.parse(path).getroot()  # nosec B314 - 部署读取固定、已校验的本地 URDF。
        elements = root.findall("joint")
        by_name = {element.get("name"): element for element in elements}
        if len(by_name) != len(elements):
            raise ValueError("URDF 中存在重复关节名称")
        joints = []
        for index, name in enumerate(JOINT_NAMES, start=1):
            element = by_name.get(name)
            if element is None or element.get("type") != "revolute":
                raise ValueError(f"URDF 缺少旋转关节 {name}")
            parent, child = element.find("parent"), element.find("child")
            expected_parent = "base_link" if index == 1 else f"Link{index - 1}"
            if (
                parent is None
                or child is None
                or parent.get("link") != expected_parent
                or child.get("link") != f"Link{index}"
            ):
                raise ValueError(f"URDF 关节 {name} 未连接到预期的 RM75 父子链")
            axis_el, limit = element.find("axis"), element.find("limit")
            if axis_el is None or limit is None:
                raise ValueError(f"URDF 关节 {name} 缺少轴或限位")
            axis = _vector(axis_el.get("xyz", "").split(), 3, f"{name} 关节轴")
            norm = float(np.linalg.norm(axis))
            if norm == 0 or not math.isfinite(norm):
                raise ValueError(f"URDF 关节 {name} 的轴无效")
            joints.append(
                RevoluteJoint(
                    name,
                    _origin(element.find("origin")),
                    axis / norm,
                    float(limit.get("lower", "nan")),
                    float(limit.get("upper", "nan")),
                    float(limit.get("velocity", "nan")),
                )
            )
        return cls(tuple(joints))

    def _frames(self, joints: FloatArray) -> tuple[list[FloatArray], FloatArray]:
        """计算各关节旋转后的帧，复用于 FK 和雅可比矩阵."""
        transform = np.eye(4)
        frames = []
        for joint, angle, skew, squared in zip(
            self.joints, joints, self._skews, self._skew_squares, strict=True
        ):
            transform = transform @ joint.origin
            rotation = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * squared
            transform[:3, :3] = transform[:3, :3] @ rotation
            frames.append(transform)
        return frames, transform

    def forward(self, joints: FloatArray) -> FloatArray:
        """计算 T_base_Link7，输入为七轴关节弧度."""
        return self._frames(_vector(joints, 7, "关节角"))[1]

    def _jacobian(self, frames: list[FloatArray], end: FloatArray) -> FloatArray:
        """构造 base 系几何雅可比，上三行为平移、下三行为旋转."""
        jacobian = np.zeros((6, 7))
        for index, (joint, frame) in enumerate(zip(self.joints, frames, strict=True)):
            axis = frame[:3, :3] @ joint.axis
            jacobian[:3, index] = np.cross(axis, end[:3, 3] - frame[:3, 3])
            jacobian[3:, index] = axis
        return jacobian

    def within_limits(self, joints: FloatArray) -> bool:
        """判断七轴是否都在 URDF 允许的位置范围内."""
        angles = np.asarray(joints, dtype=np.float64)
        return angles.shape == (7,) and bool(np.all(angles >= self.lower) and np.all(angles <= self.upper))

    def solve_ik(self, target: FloatArray, seed: FloatArray) -> FloatArray | None:
        """以相邻路点作为种子求解阻尼最小二乘 IK，不可达时返回 None."""
        goal = validate_transform(target)
        start = _vector(seed, 7, "IK 种子")
        if not self.within_limits(start):
            return None
        angles = start.copy()
        best_error = math.inf
        stalled = 0
        relative = np.eye(4)
        for _ in range(300):
            frames, current = self._frames(angles)
            position_error = goal[:3, 3] - current[:3, 3]
            relative[:3, :3] = current[:3, :3].T @ goal[:3, :3]
            rotation_error = current[:3, :3] @ transform_to_pose(relative)[3:]
            position_norm = float(np.linalg.norm(position_error))
            rotation_norm = float(np.linalg.norm(rotation_error))
            if position_norm < 5e-4 and rotation_norm < math.radians(0.1):
                return angles
            total_error = position_norm + rotation_norm
            if total_error < 0.99 * best_error:
                best_error, stalled = total_error, 0
            else:
                stalled += 1
                if stalled >= 20:
                    return None
            jacobian = self._jacobian(frames, current)
            pseudo_inverse = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 0.02**2 * np.eye(6), np.eye(6)
            )
            # 零空间分量拉回原种子，减小冗余七轴在相邻目标间的构型变化。
            step = pseudo_inverse @ np.concatenate((position_error, rotation_error))
            step += (np.eye(7) - pseudo_inverse @ jacobian) @ (0.05 * (start - angles))
            largest = float(np.max(np.abs(step)))
            if largest > 0.2:
                step *= 0.2 / largest
            angles = np.clip(angles + step, self.lower, self.upper)
        return None
