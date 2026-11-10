"""腕部 UVC MJPEG 采集，按物理端口选设备并分离采集和解码.

设备定位与采集参考 FastUMI_Data 的 fastumi_usb_camera/capture.py。
"""

import re
import shutil
import subprocess  # nosec B404 - 仅在端口枚举时运行 udevadm，不启动 shell。
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.utils.import_utils import _uvc_available, require_package

from .types import CameraFrame

if TYPE_CHECKING:
    from .config import CameraConfig

if TYPE_CHECKING or _uvc_available:
    import uvc

BY_PATH = Path("/dev/v4l/by-path")
USB_SYSFS = Path("/sys/bus/usb/devices")
VIDEO_SYSFS = Path("/sys/class/video4linux")


def _usb_address(device: Path) -> tuple[int, int]:
    """读取 sysfs 上的 USB 总线号和设备地址."""
    return int((device / "busnum").read_text()), int((device / "devnum").read_text())


def _usb_location_from_udev(path: Path, usb_sysfs: Path) -> tuple[int, int]:
    """视频节点缺失时按相同物理端口匹配唯一的 UVC 控制接口."""
    udevadm = shutil.which("udevadm")
    if udevadm is None:
        raise RuntimeError("相机视频节点缺失，且未找到用于定位物理端口的 udevadm")
    expected = path.name.removesuffix("-video-index0")
    matches = []
    for interface in usb_sysfs.glob("*:*"):
        try:
            if (interface / "bInterfaceClass").read_text().strip() != "0e" or (
                interface / "bInterfaceSubClass"
            ).read_text().strip() != "01":
                continue
            output = subprocess.run(  # nosec B603 - 固定程序及参数，设备路径来自 sysfs。
                [udevadm, "info", "--query=property", f"--path={interface}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout
            if f"ID_PATH={expected}" in output.splitlines():
                matches.append(_usb_address(interface.resolve(strict=True).parent))
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"未能在物理端口 {path} 找到唯一在线 UVC 相机")
    return matches[0]


def resolve_usb_location(
    video_device: str,
    *,
    by_path_root: Path = BY_PATH,
    device_root: Path = Path("/dev"),
    video_sysfs: Path = VIDEO_SYSFS,
    usb_sysfs: Path = USB_SYSFS,
) -> tuple[int, int]:
    """校验 udev 主节点路径，再沿 sysfs 定位对应的 USB 设备."""
    path = Path(video_device)
    if path.parent != by_path_root or not re.fullmatch(r"[^/]+-video-index0", path.name):
        raise ValueError(f"相机设备应为 /dev/v4l/by-path/*-video-index0：{path}")
    try:
        node = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return _usb_location_from_udev(path, usb_sysfs)
    if node.parent != device_root.resolve() or not re.fullmatch(r"video[0-9]+", node.name):
        raise ValueError(f"相机设备路径必须指向 /dev/video*：{path}")
    interface = (video_sysfs / node.name / "device").resolve(strict=True)
    for parent in (interface, *interface.parents):
        try:
            return _usb_address(parent)
        except (OSError, ValueError):
            continue
    raise RuntimeError(f"未能为 {path} 找到 USB 总线号和设备地址")


class CameraSource:
    """保持一个待解码 MJPEG 和有界 RGB 历史，提供非阻塞读取."""

    def __init__(
        self,
        config: "CameraConfig",
        *,
        retain_jpeg: bool = False,
        uvc_module: Any = None,
        clock: Callable[[], float] = time.monotonic,
        resolve_location: Callable[[str], tuple[int, int]] = resolve_usb_location,
    ) -> None:
        """准备设备配置及有界缓存，在 connect 阶段打开相机."""
        self.config = config
        self._retain_jpeg = retain_jpeg
        self._uvc = uvc_module
        self._clock = clock
        self._resolve_location = resolve_location
        self._capture: Any = None
        self._threads: list[threading.Thread] = []
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._pending: tuple[int, float, bytes] | None = None
        self._history: deque[CameraFrame] = deque(maxlen=config.history_size)
        self._error: Exception | None = None
        self._latest_timestamp: float | None = None

    def connect(self) -> None:
        """选择唯一 USB 设备、精确设置采集模式，并等待首张有效图像."""
        if self._capture is not None:
            raise RuntimeError("腕部相机已连接")
        if self._uvc is None:
            try:
                require_package("pupil-labs-uvc", extra="rm75b", import_name="uvc")
            except ImportError as error:
                raise ImportError("缺少 UVC 支持，请先运行 scripts/setup_rm75b_deployment.sh") from error
            self._uvc = uvc
        location = self._resolve_location(self.config.video_device)
        devices = [
            item
            for item in self._uvc.device_list()
            if (item.get("bus_number"), item.get("device_address")) == location
        ]
        if len(devices) != 1:
            raise RuntimeError(f"USB 地址 {location} 未匹配唯一 UVC 设备")
        self._capture = self._uvc.Capture(devices[0]["uid"])
        try:
            requested = (self.config.width, self.config.height, self.config.fps)
            # Pupil Labs 的 CameraMode 包含 format_name 和 supported。
            matching = [
                mode
                for mode in self._capture.available_modes
                if tuple(mode[:3]) == requested
                and getattr(mode, "supported", True)
                and getattr(mode, "format_name", "MJPG").upper() in {"MJPG", "MJPEG", "MJPEGFORMAT"}
            ]
            if not matching:
                raise RuntimeError(f"UVC 设备不支持精确 MJPEG 采集模式 {requested}")
            self._capture.frame_mode = matching[0]
            self._stop.clear()
            self._error = None
            self._threads = [
                threading.Thread(target=self._capture_loop, name="rm75b-uvc", daemon=True),
                threading.Thread(target=self._decode_loop, name="rm75b-mjpeg", daemon=True),
            ]
            for thread in self._threads:
                thread.start()
            # 初始化允许 UVC 协商首帧，进入控制循环后严格使用配置的帧时效。
            with self._condition:
                ready = self._condition.wait_for(
                    lambda: self._latest_timestamp is not None or self._error is not None, timeout=2.0
                )
                self._raise_error()
                if not ready:
                    raise RuntimeError("腕部相机未在 2 秒内产生有效 MJPEG 图像")
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"关闭腕部相机时也出现错误：{cleanup_error}")
            raise

    def _set_error(self, error: Exception) -> None:
        """在读线程出错时唤醒解码线程和主进程的连接检查."""
        with self._condition:
            if self._error is None:
                self._error = error
            self._stop.set()
            self._condition.notify_all()

    def _raise_error(self) -> None:
        """将后台错误传播给调用者，禁止重复返回旧图像."""
        if self._error is not None:
            raise RuntimeError("腕部相机采集或解码线程退出") from self._error

    def _capture_loop(self) -> None:
        """主机收到完整 JPEG 时打时间戳，覆盖未消费的旧 JPEG."""
        frame_id = 0
        try:
            while not self._stop.is_set():
                try:
                    frame = self._capture.get_frame(timeout=0.1)
                except TimeoutError:
                    continue
                timestamp = self._clock()
                jpeg = getattr(frame, "jpeg_buffer", None)
                if jpeg is None or not getattr(frame, "data_fully_received", True):
                    continue
                data = bytes(jpeg)
                if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                    continue
                with self._condition:
                    self._pending = (frame_id, timestamp, data)
                    frame_id += 1
                    self._condition.notify_all()
        except Exception as error:
            self._set_error(error)

    def _decode_loop(self) -> None:
        """消费最新待解码帧，将 OpenCV BGR 转为模型输入所需 RGB."""
        try:
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(lambda: self._pending is not None or self._stop.is_set())
                    if self._stop.is_set():
                        return
                    pending, self._pending = self._pending, None
                if pending is None:
                    continue
                frame_id, timestamp, data = pending
                bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                if bgr.shape != (self.config.height, self.config.width, 3):
                    raise RuntimeError(f"腕部相机帧尺寸 {bgr.shape} 与协商模式不符")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self._condition:
                    self._history.append(
                        CameraFrame(frame_id, timestamp, rgb, data if self._retain_jpeg else None)
                    )
                    self._latest_timestamp = timestamp
                    self._condition.notify_all()
        except Exception as error:
            self._set_error(error)

    def read_frames(self) -> list[CameraFrame]:
        """交付尚未消费的历史，超过图像时效时报告采集故障."""
        with self._condition:
            self._raise_error()
            if (
                self._latest_timestamp is None
                or self._clock() - self._latest_timestamp > self.config.frame_timeout_s
            ):
                raise RuntimeError("腕部相机没有新鲜有效图像")
            frames = list(self._history)
            self._history.clear()
            return frames

    def close(self) -> None:
        """停止并等待后台线程，再释放原生 UVC 句柄."""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            if thread.ident is not None:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    raise RuntimeError(f"相机线程 {thread.name} 未能退出")
        self._threads.clear()
        if self._capture is not None:
            self._capture.close()
            self._capture = None
        self._pending = None
        self._history.clear()
        self._latest_timestamp = None
