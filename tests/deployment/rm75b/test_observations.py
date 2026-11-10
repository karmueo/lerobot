"""检查三帧按时间同步、预处理和防止失效反馈插值."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lerobot.deployment.rm75b.config import ExecutionConfig
from lerobot.deployment.rm75b.geometry import pose_to_transform
from lerobot.deployment.rm75b.observations import ObservationAssembler, preprocess_rgb
from lerobot.deployment.rm75b.telemetry import RunLog
from lerobot.deployment.rm75b.types import ActionChunk, ArmState, CameraFrame, FloatArray, GripperState


class _Kinematics:
    """将首关节值转换为基座 y 方向平移，便于独立计算期望."""

    def forward(self, joints: FloatArray) -> FloatArray:
        return pose_to_transform([0.3, joints[0], 0.4, 0, 0, math.pi / 2])


def _frame(index: int, timestamp: float) -> CameraFrame:
    """生成包含不同帧号和不同像素值的 RGB 图像."""
    return CameraFrame(index, timestamp, np.full((134, 240, 3), index, dtype=np.uint8))


def _feed(assembler: ObservationAssembler, times: list[float]) -> None:
    """提供两侧真实反馈，关节和夹爪均随时间线性变化."""
    for t in times:
        joints = np.zeros(7)
        joints[0] = t - 1
        assembler.add_arm(ArmState(t, joints, np.zeros(7)))
        assembler.add_gripper(GripperState(t, 0.5 + t - 1, 0.5 + t - 1))


def test_crop_preserves_rgb_channels_and_center() -> None:
    """目标宽高比裁剪去掉两侧不同颜色，保持 RGB 通道."""
    image = np.empty((134, 300, 3), dtype=np.uint8)
    image[:, :30] = [0, 0, 255]
    image[:, 30:270] = [255, 64, 0]
    image[:, 270:] = [0, 255, 0]
    result = preprocess_rgb(image)
    assert result.dtype == np.float32
    assert result.shape == (3, 134, 240)
    np.testing.assert_allclose(result[:, 0, 0], [1, 64 / 255, 0], atol=1e-7)
    np.testing.assert_allclose(result[:, -1, -1], [1, 64 / 255, 0], atol=1e-7)


def test_three_distinct_frames_and_bracketed_state() -> None:
    """用图像时刻插值真实关节和夹爪，并使用冻结原点."""
    offset = pose_to_transform([0.1, 0, 0, 0, 0, 0])
    anchor = _Kinematics().forward(np.zeros(7)) @ offset
    assembler = ObservationAssembler(_Kinematics(), offset, ExecutionConfig())
    _feed(assembler, [0.995, 1.005, 1.03, 1.04, 1.06, 1.07])
    for index, timestamp in enumerate([1.0, 1 + 1 / 30, 1 + 2 / 30]):
        assembler.add_frame(_frame(index, timestamp))
    assembler.add_frame(_frame(2, 1.10))
    window = assembler.build(anchor, episode_id=4, request_id=9, not_before=0.99, now=1.1)
    assert window is not None
    assert window.frame_ids == (0, 1, 2)
    assert (window.episode_id, window.request_id) == (4, 9)
    np.testing.assert_allclose(window.states[:, 0], [0, 1 / 30, 2 / 30], atol=1e-7)
    np.testing.assert_allclose(window.states[:, -1], [0.5, 0.5 + 1 / 30, 0.5 + 2 / 30], atol=1e-7)
    assert window.capture_tcp[1, 3] == pytest.approx(0.1 + 2 / 30)


def test_missing_and_stale_inputs_cannot_extrapolate_or_cross_boundary() -> None:
    """没有最新图像右侧反馈、历史跨稳定边界或输入过期均拒绝."""
    assembler = ObservationAssembler(_Kinematics(), np.eye(4), ExecutionConfig())
    for index in range(3):
        assembler.add_frame(_frame(index, 1 + index / 30))
    _feed(assembler, [0.99, 1.01, 1.04, 1.06])
    anchor = np.eye(4)
    assert assembler.build(anchor, 1, 1, 0.98, 1.08) is None
    _feed(assembler, [1.075])
    assert assembler.build(anchor, 1, 1, 1.01, 1.08) is None
    assert assembler.build(anchor, 1, 1, 0.98, 1.5) is None
    assert assembler.build(anchor, 1, 1, 0.98, 1.08) is not None
    assembler.clear()
    assert assembler.build(anchor, 2, 1, 0.98, 1.08) is None


def test_archive_pairs_original_frames_and_interpolated_joints(tmp_path: Path) -> None:
    """写盘后按同一请求核对原始帧、同步状态和动作."""
    kin = _Kinematics()
    anchor = kin.forward(np.zeros(7))
    assembler = ObservationAssembler(kin, np.eye(4), ExecutionConfig(), save_inference=True)
    _feed(assembler, [0.995, 1.005, 1.03, 1.04, 1.06, 1.07])
    originals = (b"original-0", b"original-1", b"original-2")
    for index, jpeg in enumerate(originals):
        assembler.add_frame(replace(_frame(index, 1 + index / 30), jpeg=jpeg))
    window = assembler.build(anchor, 4, 9, 0.99, 1.1)
    assert window is not None
    actions = np.arange(56, dtype=np.float32).reshape(8, 7)
    with RunLog(tmp_path, {"mode": "run"}, save_inference=True) as log:
        log.write_inference_input(window)
        log.write_inference_output(ActionChunk(4, 9, window.timestamps[-1], actions), False)
        log.write_inference_input(replace(window, request_id=10))
    name = "episode_000004_request_000009"
    folder = log.directory / "inference_inputs" / name
    metadata = json.loads((folder / "observation.json").read_text())
    assert tuple((folder / image).read_bytes() for image in metadata["images"]) == originals
    assert metadata["frame_ids"] == [0, 1, 2]
    assert metadata["timestamps"] == list(window.timestamps)
    np.testing.assert_allclose(np.asarray(metadata["joints_rad"])[:, 0], [0, 1 / 30, 2 / 30])
    np.testing.assert_array_equal(metadata["states"], window.states)
    outputs = log.directory / "inference_outputs"
    output = json.loads((outputs / f"{name}.json").read_text())
    np.testing.assert_array_equal(output["actions"], actions)
    assert (output["episode_id"], output["request_id"]) == (metadata["episode_id"], metadata["request_id"])
    assert output["discarded"] is False
    assert (log.directory / "inference_inputs" / "episode_000004_request_000010").is_dir()
    assert len(list(outputs.glob("*.json"))) == 1


def test_incorrect_history_spacing_cannot_fill_window() -> None:
    """即使三个不同帧均能同步，也必须满足 30 Hz 观测间隔."""
    assembler = ObservationAssembler(_Kinematics(), np.eye(4), ExecutionConfig())
    times = [1.0, 1.06, 1.09]
    for index, timestamp in enumerate(times):
        assembler.add_frame(_frame(index, timestamp))
    _feed(assembler, times)
    assert assembler.build(np.eye(4), 1, 1, 0.9, 1.1) is None
