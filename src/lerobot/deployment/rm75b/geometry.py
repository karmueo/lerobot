"""以米和弧度为单位的 TCP 刚体变换及 SO(3) 数值运算."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .types import FloatArray


def validate_transform(transform: ArrayLike) -> FloatArray:
    """验证有限的齐次变换、正交旋转和右手系，返回独立副本."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("齐次变换须为 4×4 有限矩阵")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0, abs_tol=1e-6)
    ):
        raise ValueError("齐次变换包含无效的旋转或底行")
    return matrix.copy()


def invert(transform: FloatArray) -> FloatArray:
    """求已验证刚体变换的逆，避免通用矩阵求逆."""
    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3].T
    result[:3, :3] = rotation
    result[:3, 3] = -(rotation @ transform[:3, 3])
    return result


def pose_to_transform(pose: ArrayLike) -> FloatArray:
    """将 xyz 和旋转向量组合为齐次变换."""
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("位姿须为有限的六维 xyz/rotvec")
    rx, ry, rz = values[3:]
    skew = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]], dtype=np.float64)
    theta = float(np.linalg.norm(values[3:]))
    # sinc 形式在接近零时也稳定，分别对应 sin(theta)/theta 和 (1-cos(theta))/theta²。
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] += np.sinc(theta / np.pi) * skew + 0.5 * np.sinc(theta / (2 * np.pi)) ** 2 * (skew @ skew)
    result[:3, 3] = values[:3]
    return result


def _rotation_vector(rotation: FloatArray) -> FloatArray:
    """SO(3) 对数，使用反对称项判别近 π 处的轴方向."""
    skew = 0.5 * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    sine = float(np.linalg.norm(skew))
    cosine = float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    angle = math.atan2(sine, cosine)
    if angle < 1e-8:
        return skew
    if math.pi - angle < 1e-5:
        # 对称部分的最大特征向量在 π 附近仍能稳定求出旋转轴。
        _, axes = np.linalg.eigh((rotation + rotation.T) / 2)
        axis = axes[:, -1]
        if float(axis @ skew) < 0:
            axis = -axis
        return np.asarray(axis * angle, dtype=np.float64)
    return skew * (angle / sine)


def transform_to_pose(transform: FloatArray) -> FloatArray:
    """将已验证的齐次变换转为 xyz 和主值旋转向量."""
    return np.concatenate((transform[:3, 3], _rotation_vector(transform[:3, :3])))


def rotation_distance(left: FloatArray, right: FloatArray) -> float:
    """求两个 3×3 旋转之间的最小角距离，单位弧度."""
    return float(np.linalg.norm(_rotation_vector(left.T @ right)))
