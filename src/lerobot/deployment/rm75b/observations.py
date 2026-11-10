"""按主机接收时间同步三帧相机、机械臂和夹爪观测."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import IMAGE_SHAPE, ExecutionConfig
from .geometry import invert, transform_to_pose, validate_transform
from .types import (
    ArmState,
    CameraFrame,
    FloatArray,
    GripperState,
    ImageArray,
    ObservationArchive,
    ObservationWindow,
)


class ForwardKinematics(Protocol):
    """观测层只需要根据实测关节计算基座至法兰的变换."""

    def forward(self, joints: FloatArray, /) -> FloatArray:
        """计算基座至法兰的齐次变换."""
        ...


@dataclass(frozen=True)
class _PreparedFrame:
    """缓存预处理图像，开启存档时同时保留原始压缩帧."""

    frame_id: int
    timestamp: float
    image: NDArray[np.float32]
    jpeg: bytes | None


def preprocess_rgb(rgb: ImageArray) -> NDArray[np.float32]:
    """复用 FastUMI 训练数据导出约定，居中裁剪再以 INTER_AREA 缩放."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) <= 0:
        raise ValueError("相机观测须为非空 HWC uint8 RGB 图像")
    _, height, width = IMAGE_SHAPE
    source_height, source_width = rgb.shape[:2]
    # 裁剪输入的过长边，奇数剩余像素留在图像右侧或底部。
    ratio = width / height
    if source_width / source_height > ratio:
        crop_width = round(source_height * ratio)
        start = (source_width - crop_width) // 2
        cropped = rgb[:, start : start + crop_width]
    else:
        crop_height = round(source_width / ratio)
        start = (source_height - crop_height) // 2
        cropped = rgb[start : start + crop_height]
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(np.moveaxis(resized, -1, 0), dtype=np.float32) / np.float32(255)


def _bracket[Feedback: (ArmState, GripperState)](
    history: deque[Feedback], timestamp: float, not_before: float, slop: float, now: float
) -> tuple[Feedback, Feedback, float] | None:
    """查找可插值的两侧实测反馈，禁止时间范围外推."""
    times = [sample.timestamp for sample in history]
    index = bisect_left(times, timestamp)
    if index == len(times):
        return None
    right = history[index]
    if right.timestamp > now:
        return None
    if right.timestamp == timestamp:
        return right, right, 0.0
    if index == 0:
        return None
    left = history[index - 1]
    if left.timestamp < not_before or max(timestamp - left.timestamp, right.timestamp - timestamp) > slop:
        return None
    fraction = (timestamp - left.timestamp) / (right.timestamp - left.timestamp)
    return left, right, fraction


class ObservationAssembler:
    """从有界缓存读取三帧独立图像和同时间的真实状态."""

    def __init__(
        self,
        kinematics: ForwardKinematics,
        tcp_offset: FloatArray,
        cfg: ExecutionConfig,
        *,
        save_inference: bool = False,
    ) -> None:
        """外参是 Link7 到 TCP 的固定刚体变换."""
        self._kinematics = kinematics
        self._tcp_offset = validate_transform(tcp_offset)
        self._cfg = cfg
        self._save_inference = save_inference
        self._arm: deque[ArmState] = deque(maxlen=256)
        self._gripper: deque[GripperState] = deque(maxlen=256)
        self._frames: deque[_PreparedFrame] = deque(maxlen=16)

    def clear(self) -> None:
        """清除旧动作段或旧回合的数据."""
        self._arm.clear()
        self._gripper.clear()
        self._frames.clear()

    def add_arm(self, sample: ArmState) -> None:
        """接收单调的七轴反馈，无效样本切断插值区间."""
        if (
            sample.errors
            or not math.isfinite(sample.timestamp)
            or sample.joints.shape != (7,)
            or not np.isfinite(sample.joints).all()
        ):
            self._arm.clear()
        elif not self._arm or sample.timestamp > self._arm[-1].timestamp:
            self._arm.append(sample)

    def add_gripper(self, sample: GripperState) -> None:
        """仅缓存有限的真实开度，反馈失效后等待新数据."""
        if not math.isfinite(sample.timestamp) or not 0 <= sample.openness <= 1:
            self._gripper.clear()
        elif not self._gripper or sample.timestamp > self._gripper[-1].timestamp:
            self._gripper.append(sample)

    def add_frame(self, frame: CameraFrame) -> None:
        """忽略重复和乱序帧，按帧缓存预处理图像."""
        if not math.isfinite(frame.timestamp):
            return
        if self._frames and (
            frame.timestamp <= self._frames[-1].timestamp or frame.frame_id <= self._frames[-1].frame_id
        ):
            return
        jpeg = frame.jpeg if self._save_inference else None
        self._frames.append(_PreparedFrame(frame.frame_id, frame.timestamp, preprocess_rgb(frame.rgb), jpeg))

    def _sample(
        self, timestamp: float, not_before: float, now: float
    ) -> tuple[FloatArray, float, FloatArray] | None:
        """在图像时间线性插值关节角和夹爪开度，再计算实际 TCP."""
        arm = _bracket(self._arm, timestamp, not_before, self._cfg.sync_slop_s, now)
        gripper = _bracket(self._gripper, timestamp, not_before, self._cfg.sync_slop_s, now)
        if arm is None or gripper is None:
            return None
        arm_left, arm_right, arm_fraction = arm
        grip_left, grip_right, grip_fraction = gripper
        joints = arm_left.joints + arm_fraction * (arm_right.joints - arm_left.joints)
        openness = grip_left.openness + grip_fraction * (grip_right.openness - grip_left.openness)
        return self._kinematics.forward(joints) @ self._tcp_offset, openness, joints

    def build(
        self, anchor: FloatArray, episode_id: int, request_id: int, not_before: float, now: float
    ) -> ObservationWindow | None:
        """寻找最新的完整窗口，历史缺失、过期或不同步时返回 None."""
        valid_after = max(not_before, now - self._cfg.input_timeout_s)
        frames = [frame for frame in self._frames if valid_after <= frame.timestamp <= now]
        if len(frames) < self._cfg.n_obs_steps:
            return None
        # 最新帧可能尚无右侧反馈，先找同时满足时间和真实反馈的帧，再匹配历史。
        synced = []
        for frame in frames:
            if sample := self._sample(frame.timestamp, not_before, now):
                synced.append((frame, *sample))
        start_from_base = invert(validate_transform(anchor))
        for current_index in range(len(synced) - 1, self._cfg.n_obs_steps - 2, -1):
            current = synced[current_index]
            picked = []
            previous_limit = 0
            for offset in range(self._cfg.n_obs_steps - 1, 0, -1):
                expected = current[0].timestamp - offset / self._cfg.fps
                candidates = range(previous_limit, current_index)
                nearest = min(
                    candidates,
                    key=lambda i: abs(synced[i][0].timestamp - expected),
                    default=None,
                )
                if (
                    nearest is None
                    or abs(synced[nearest][0].timestamp - expected) > self._cfg.history_tolerance_s
                ):
                    break
                picked.append(synced[nearest])
                previous_limit = nearest + 1
            if len(picked) != self._cfg.n_obs_steps - 1:
                continue
            picked.append(current)
            archive = None
            if self._save_inference:
                # 与模型使用完全相同的选帧顺序及插值关节角。
                jpegs = tuple(frame.jpeg for frame, *_ in picked if frame.jpeg is not None)
                if len(jpegs) != len(picked):
                    raise RuntimeError("推理存档缺少原始 JPEG 帧")
                archive = ObservationArchive(jpegs, np.stack([joints for _, _, _, joints in picked]))
            return ObservationWindow(
                episode_id=episode_id,
                request_id=request_id,
                frame_ids=tuple(frame.frame_id for frame, *_ in picked),
                timestamps=tuple(frame.timestamp for frame, *_ in picked),
                images=np.stack([frame.image for frame, *_ in picked]),
                states=np.asarray(
                    [
                        np.append(transform_to_pose(start_from_base @ tcp), openness)
                        for _, tcp, openness, _ in picked
                    ],
                    dtype=np.float32,
                ),
                anchor=anchor.copy(),
                capture_tcp=current[1].copy(),
                archive=archive,
            )
        return None
