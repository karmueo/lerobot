"""RM_API2 七轴适配器，SDK 入口接收度，部署侧始终使用弧度."""

import importlib
import importlib.util
import math
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from .types import ArmState, FloatArray

if TYPE_CHECKING:
    from .config import ArmConfig


def load_rm_sdk(root: Path) -> ModuleType:
    """按配置加载厂商源码目录，无需把 SDK 安装成 Python 发行包."""
    package = Path(root).resolve() / "Robotic_Arm"
    module_name = "Robotic_Arm.rm_robot_interface"
    if not (package / "rm_robot_interface.py").is_file():
        raise ImportError(f"RM SDK 未找到：{package / 'rm_robot_interface.py'}")
    loaded = sys.modules.get("Robotic_Arm")
    if loaded is not None:
        if Path(loaded.__file__ or "").resolve().parent != package:
            raise ImportError(f"已加载其他版本的 RM SDK：{loaded.__file__}")
        return importlib.import_module(module_name)

    spec = importlib.util.spec_from_file_location(
        "Robotic_Arm", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 RM SDK：{package}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return importlib.import_module(module_name)
    except BaseException:
        # 加载失败时移除本次导入的半初始化 SDK，便于诊断后重试。
        for name in tuple(sys.modules):
            if name == spec.name or name.startswith(f"{spec.name}."):
                sys.modules.pop(name)
        raise


def check_udp_endpoint(ip: str, port: int) -> None:
    """绑定探测本机地址和端口，进入 SDK 前拒绝明显的反馈通道冲突."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind((ip, port))
    except OSError as error:
        raise RuntimeError(f"RM UDP 反馈端点 {ip}:{port} 不可绑定：{error}") from error


def _require_ok(result: int, operation: str) -> None:
    """保留厂商返回码，区分控制器拒绝与正常发送."""
    if result != 0:
        raise RuntimeError(f"{operation} 失败，RM SDK 返回 {result}")


class ArmDriver:
    """独占一套 RM SDK 三线程连接，不在回调内调用 SDK 或运行运动学."""

    def __init__(
        self,
        config: "ArmConfig",
        *,
        motion_enabled: bool = False,
        sdk: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """保存连接配置和运动授权，SDK 在 connect 阶段初始化."""
        self.config = config
        self._motion_enabled = motion_enabled
        self._sdk = sdk
        self._clock = clock
        self._arm: Any = None
        self._connected = False
        self._handle_valid = False
        self._callback: Any = None
        self._old_udp: Any = None
        self._stop_needed = False
        self._feedback: tuple[float, tuple[float, ...], tuple[float, ...]] | None = None
        self._feedback_fault: tuple[int, ...] = ()
        self._decode_error: int | None = None

    def connect(self) -> None:
        """核对 RM75 型号并订阅 7 轴反馈，连接阶段不调用执行器."""
        if self._arm is not None:
            raise RuntimeError("RM SDK 已初始化")
        check_udp_endpoint(self.config.udp_ip, self.config.udp_port)
        sdk = self._sdk if self._sdk is not None else load_rm_sdk(self.config.sdk_root)
        self._sdk = sdk
        self._feedback = None
        self._feedback_fault = ()
        self._decode_error = None
        arm = sdk.RoboticArm(sdk.rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self._arm = arm
        try:
            # SDK 默认命令超时 500 ms；控制停止不能等待完整默认超时。
            arm.rm_set_timeout(100)
            handle = arm.rm_create_robot_arm(self.config.ip, self.config.port)
            self._handle_valid = handle is not None and handle.id >= 0
            if not self._handle_valid:
                raise RuntimeError(f"无法连接 RM75B {self.config.ip}:{self.config.port}")
            result, info = arm.rm_get_robot_info()
            _require_ok(result, "读取机械臂型号")
            if info.get("arm_model") != "RM_75" or info.get("arm_dof") != 7:
                raise RuntimeError(f"控制器不是七轴 RM_75：{info}")
            result, previous = arm.rm_get_realtime_push()
            _require_ok(result, "读取 RM UDP 配置")
            # 厂商 SDK 的 cycle 单位为 5 ms，和 FastUMI 驱动一致。
            cycle = self.config.udp_cycle_ms // 5
            custom = previous["custom_config"]
            wanted = {
                **previous,
                "cycle": cycle,
                "enable": True,
                "ip": self.config.udp_ip,
                "port": self.config.udp_port,
                "custom_config": {**custom, "joint_speed": 1},
            }
            self._callback = sdk.rm_realtime_arm_state_callback_ptr(self._receive)
            arm.rm_realtime_arm_state_call_back(self._callback)
            if wanted != previous:
                # 关闭时还原原有反馈配置，避免影响其他控制器的后续会话。
                self._old_udp = self._udp_config(previous)
                _require_ok(arm.rm_set_realtime_push(self._udp_config(wanted)), "设置 RM UDP 反馈")
                result, actual = arm.rm_get_realtime_push()
                _require_ok(result, "核对 RM UDP 配置")
                if (
                    any(actual.get(key) != wanted[key] for key in ("cycle", "enable", "ip", "port"))
                    or actual["custom_config"].get("joint_speed") != 1
                ):
                    raise RuntimeError(f"RM 控制器未接受所需 UDP 反馈配置：{actual}")
            self._connected = True
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"关闭 RM SDK 时也出现错误：{cleanup_error}")
            raise

    def _udp_config(self, values: dict[str, Any]) -> Any:
        """从 SDK 查询字典重建结构体，保留其他反馈项."""
        sdk = self._sdk
        custom = sdk.rm_udp_custom_config_t(**values["custom_config"])
        return sdk.rm_realtime_push_config_t(
            cycle=values["cycle"],
            enable=values["enable"],
            port=values["port"],
            force_coordinate=values["force_coordinate"],
            ip=values["ip"],
            custom_config=custom,
        )

    def _receive(self, packet: Any) -> None:
        """只复制反馈和主机接收时刻，锁存故障以免被后续数据覆盖."""
        received = self._clock()
        try:
            if packet.arm_ip.decode("ascii") != self.config.ip:
                return
            if packet.errCode:
                self._decode_error = int(packet.errCode)
                return
            status = packet.joint_status
            position, velocity = tuple(status.joint_position), tuple(status.joint_speed)
            if (
                len(position) != 7
                or len(velocity) != 7
                or not all(math.isfinite(value) for value in (*position, *velocity))
                or not 0 <= packet.err.err_len <= len(packet.err.err)
            ):
                self._decode_error = -1
                return
            errors = tuple(int(code) for code in status.joint_err_code if code)
            errors += tuple(int(code) for code in packet.err.err[: packet.err.err_len] if code)
            # 负值为部署层反馈故障，和厂商正错误码区分。
            errors += tuple(-100 - index for index, enabled in enumerate(status.joint_en_flag) if not enabled)
            if errors:
                self._feedback_fault = errors
            self._feedback = (received, position, velocity)
        except (AttributeError, TypeError, ValueError, IndexError):
            self._decode_error = -1

    def read_state(self) -> ArmState | None:
        """将最新一帧真实 UDP 反馈转换为部署单位，保留原接收时间."""
        if self._decode_error is not None:
            raise RuntimeError(f"RM UDP 反馈解析失败，错误码 {self._decode_error}")
        if self._feedback is None:
            return None
        received, position, velocity = self._feedback
        return ArmState(received, np.deg2rad(position), np.deg2rad(velocity), self._feedback_fault)

    def _actuator(self, *, stopping: bool = False) -> Any:
        """在 SDK 边界拒绝 dry-run 调用和无有效反馈时的新运动."""
        if not self._motion_enabled:
            raise RuntimeError("RM 运动接口未启用")
        if not self._connected or self._arm is None:
            raise RuntimeError("RM SDK 未连接")
        if not stopping:
            state = self.read_state()
            if state is None or state.errors or not 0.0 <= self._clock() - state.timestamp <= 0.25:
                raise RuntimeError("RM 运动前需要新鲜、无故障的七轴反馈")
        return self._arm

    @staticmethod
    def _joint_degrees(joints: FloatArray) -> list[float]:
        """只接受有限七轴关节角，并在 SDK 边界转为度."""
        radians = np.asarray(joints, dtype=np.float64)
        if radians.shape != (7,) or not np.isfinite(radians).all():
            raise ValueError("RM 关节目标必须为七个有限的弧度值")
        return np.rad2deg(radians).tolist()

    def send_joint_target(self, joints: FloatArray) -> None:
        """发送一个低跟随 CANFD 目标，由主循环按 50 Hz 调用."""
        arm = self._actuator()
        target = self._joint_degrees(joints)
        # 即使发送失败也可能已到达控制器，退出时仍需请求停臂。
        self._stop_needed = True
        _require_ok(arm.rm_movej_canfd(target, follow=False), "发送低跟随关节目标")

    def home(self, joints: FloatArray, speed: int = 10) -> None:
        """显式提交控制器规划的回位命令，由主循环监测到位."""
        arm = self._actuator()
        if type(speed) is not int or not 1 <= speed <= 100:
            raise ValueError("回位速度百分比应为 1～100 的整数")
        target = self._joint_degrees(joints)
        self._stop_needed = True
        _require_ok(arm.rm_movej(target, speed, 0, 0, 0), "回初始关节位")

    def stop(self) -> None:
        """中止此前发送的运动，在下一条显式运动命令前保持幂等."""
        arm = self._actuator(stopping=True)
        if self._stop_needed:
            _require_ok(arm.rm_set_arm_stop(), "停止机械臂")
            self._stop_needed = False

    def close(self) -> None:
        """还原 UDP 配置，释放句柄和 SDK 线程，不发送新的运动目标."""
        arm = self._arm
        if arm is None:
            return
        failures: list[Exception] = []
        if self._handle_valid:
            if self._old_udp is not None:
                try:
                    _require_ok(arm.rm_set_realtime_push(self._old_udp), "还原 RM UDP 配置")
                except Exception as error:
                    failures.append(error)
            try:
                _require_ok(arm.rm_delete_robot_arm(), "释放 RM 连接")
            except Exception as error:
                failures.append(error)
        try:
            _require_ok(arm.rm_destroy(), "关闭 RM SDK 线程")
        except Exception as error:
            failures.append(error)
        else:
            # C 回调的 Python 引用要保留到厂商线程完全退出之后。
            self._callback = None
            self._arm = None
        self._handle_valid = False
        self._connected = False
        self._feedback = None
        self._feedback_fault = ()
        self._decode_error = None
        self._old_udp = None
        self._stop_needed = False
        if failures:
            raise ExceptionGroup("释放 RM SDK 失败", failures)
