"""验证物理端口定位、完整 MJPEG 检查与相机时效。"""

from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from typing import NamedTuple

import cv2
import numpy as np
import pytest

from lerobot.deployment.rm75b.camera import CameraSource, resolve_usb_location
from lerobot.deployment.rm75b.config import CameraConfig


class FakeCapture:
    """阻塞式模拟设备，复用真实 JPEG 解码器。"""

    def __init__(self, mode: tuple[int, int, int]) -> None:
        self.available_modes = [mode]
        self.frame_mode = None
        self.frames: Queue[SimpleNamespace | Exception] = Queue()
        self.closed = False

    def get_frame(self, timeout: float) -> SimpleNamespace:
        try:
            frame = self.frames.get(timeout=timeout)
        except Empty as error:
            raise TimeoutError from error
        if isinstance(frame, Exception):
            raise frame
        return frame

    def close(self) -> None:
        self.closed = True


class CameraMode(NamedTuple):
    """对应 Pupil Labs 1.0.4 的 CameraMode 字段。"""

    width: int
    height: int
    fps: int
    format_native: int
    format_name: str
    supported: bool


def fake_source(capture, config, clock, *, retain_jpeg: bool = False):
    """注入同一 USB 地址的唯一测试相机，避免读取宿主设备。"""
    uvc = SimpleNamespace(
        device_list=lambda: [{"bus_number": 1, "device_address": 2, "uid": "fake"}],
        Capture=lambda uid: capture,
    )
    return CameraSource(
        config, retain_jpeg=retain_jpeg, uvc_module=uvc, resolve_location=lambda _: (1, 2), clock=clock
    )


@pytest.mark.parametrize("retain_jpeg", [False, True])
def test_valid_mjpeg_only_and_bgr_converted_to_rgb(retain_jpeg: bool) -> None:
    mode = (32, 24, 30)
    capture = FakeCapture(mode)
    mjpeg = CameraMode(*mode, 7, "MJPG", True)
    capture.available_modes = [CameraMode(*mode, 3, "YUYV", True), mjpeg]
    config = replace(CameraConfig(), width=32, height=24)
    timestamp = [100.0]
    source = fake_source(capture, config, lambda: timestamp[0], retain_jpeg=retain_jpeg)
    bgr = np.full((24, 32, 3), [30, 80, 180], dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    capture.frames.put(SimpleNamespace(jpeg_buffer=encoded.tobytes(), data_fully_received=False))
    capture.frames.put(SimpleNamespace(jpeg_buffer=b"\xff\xd8incomplete", data_fully_received=True))
    capture.frames.put(SimpleNamespace(jpeg_buffer=encoded.tobytes(), data_fully_received=True))
    source.connect()
    try:
        assert capture.frame_mode == mjpeg
        frames = source.read_frames()
        assert len(frames) == 1
        assert frames[0].frame_id == 0 and frames[0].timestamp == 100.0
        assert frames[0].rgb.shape == (24, 32, 3)
        assert frames[0].jpeg == (encoded.tobytes() if retain_jpeg else None)
        np.testing.assert_allclose(frames[0].rgb[0, 0], [180, 80, 30], atol=3)
        assert not source.read_frames()
        timestamp[0] += 0.21
        with pytest.raises(RuntimeError, match="没有新鲜"):
            source.read_frames()
    finally:
        source.close()
    assert capture.closed


def test_mode_must_match_exactly_and_failure_closes_device():
    capture = FakeCapture((1920, 1080, 25))
    source = fake_source(capture, CameraConfig(), lambda: 0.0)
    with pytest.raises(RuntimeError, match="不支持精确"):
        source.connect()
    assert capture.closed


def test_capture_error_propagates_and_closes_device():
    capture = FakeCapture((1920, 1080, 30))
    capture.frames.put(OSError("disconnected"))
    source = fake_source(capture, CameraConfig(), lambda: 0.0)
    with pytest.raises(RuntimeError, match="线程退出") as exc:
        source.connect()
    assert isinstance(exc.value.__cause__, OSError)
    assert capture.closed


def test_resolve_by_path_selects_usb_address(tmp_path: Path):
    dev = tmp_path / "dev"
    by_path = dev / "v4l" / "by-path"
    by_path.mkdir(parents=True)
    node = dev / "video7"
    node.touch()
    physical = by_path / "wrist-port-video-index0"
    physical.symlink_to(node)
    usb_device = tmp_path / "usb" / "1-4.2"
    interface = usb_device / "1-4.2:1.0"
    interface.mkdir(parents=True)
    (usb_device / "busnum").write_text("1\n")
    (usb_device / "devnum").write_text("15\n")
    sysfs = tmp_path / "video4linux"
    (sysfs / "video7").mkdir(parents=True)
    (sysfs / "video7" / "device").symlink_to(interface)
    assert resolve_usb_location(str(physical), by_path_root=by_path, device_root=dev, video_sysfs=sysfs) == (
        1,
        15,
    )
    with pytest.raises(ValueError, match="index0"):
        resolve_usb_location(str(node), by_path_root=by_path, device_root=dev, video_sysfs=sysfs)
