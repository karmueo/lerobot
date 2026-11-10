"""Unitree Dex1 夹爪适配器，连续双向映射和逐侧闭环限速."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _unitree_sdk_available, require_package

from .types import GripperState

if TYPE_CHECKING:
    from .config import GripperConfig

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_

CONTROL_PERIOD_S = 1 / 50
TARGET_TOLERANCE = 0.05  # 沿用基础验收的训练刻度开度容差。


@dataclass(frozen=True)
class DdsBindings:
    """将厂商 DDS 工厂集中注入，便于以真实字段结构模拟读写."""

    initialize: Callable[[int, str], None]
    publisher: Callable[[str, Any], Any]
    subscriber: Callable[[str, Any], Any]
    motor_cmd: Callable[..., Any]
    motor_cmds: Callable[..., Any]
    motor_states: Any


def _load_dds() -> DdsBindings:
    """延后初始化 DDS，允许纯模型和配置检查不安装夹爪依赖."""
    try:
        require_package("unitree-sdk2py", extra="rm75b", import_name="unitree_sdk2py")
    except ImportError as error:
        raise ImportError("缺少 Unitree SDK，请先运行 scripts/setup_rm75b_deployment.sh") from error
    return DdsBindings(
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber, MotorCmd_, MotorCmds_, MotorStates_
    )


class GripperDriver:
    """依据真实反馈控制启动时发现的电机集合，禁止运行中自动纳入新侧."""

    def __init__(
        self,
        config: "GripperConfig",
        *,
        motion_enabled: bool = False,
        dds: DdsBindings | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """保存标定、反馈缓存和运动授权，构造阶段不访问 DDS."""
        self.config = config
        self._motion_enabled = motion_enabled
        self._dds = dds
        self._clock = clock
        self._sleep = sleep
        self._feedback: dict[str, tuple[float, float]] = {}
        self._subscribers: dict[str, Any] = {}
        self._publishers: dict[str, Any] = {}
        self._messages: dict[str, Any] = {}
        self._last_sent: dict[str, float] = {}
        self._target_positions: dict[str, float] = {}
        self._last_tick: float | None = None
        self._online: tuple[str, ...] = ()
        self._mapping = np.asarray(config.mapping, dtype=np.float64)

    @property
    def online_sides(self) -> tuple[str, ...]:
        """当前会话在启动阶段冻结的有反馈侧."""
        return self._online

    def _receiver(self, side: str) -> Callable[[Any], None]:
        """为一侧绑定回调，只接收有效的真实电机位置."""

        def receive(message: Any) -> None:
            received = self._clock()
            try:
                position = float(message.states[0].q)
            except (AttributeError, IndexError, TypeError, ValueError):
                return
            if math.isfinite(position):
                self._feedback[side] = (received, position)

        return receive

    def connect(self) -> None:
        """读反馈完成侧别发现，连接阶段不发送开合或保持指令."""
        if self._subscribers:
            raise RuntimeError("夹爪已连接")
        dds = self._dds if self._dds is not None else _load_dds()
        dds.initialize(self.config.domain_id, self.config.network_interface)
        try:
            for side in self.config.sides:
                subscriber = dds.subscriber(f"rt/dex1/{side}/state", dds.motor_states)
                self._subscribers[side] = subscriber
                subscriber.Init(self._receiver(side))
            deadline = self._clock() + self.config.discovery_s
            while True:
                now = self._clock()
                found = tuple(side for side in self.config.sides if self._fresh(side, now) is not None)
                if len(found) == len(self.config.sides) or now >= deadline:
                    break
                self._sleep(min(0.01, deadline - now))
            if not found:
                raise RuntimeError("未收到 Unitree DDS 夹爪电机反馈，请检查厂商服务与网卡")
            self._online = found
            if self._motion_enabled:
                for side in found:
                    publisher = dds.publisher(f"rt/dex1/{side}/cmd", dds.motor_cmds)
                    self._publishers[side] = publisher
                    publisher.Init()
                    self._messages[side] = dds.motor_cmds(
                        cmds=[dds.motor_cmd(1, 0.0, 0.0, 0.0, self.config.kp, self.config.kd, [0, 0, 0])]
                    )
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"关闭夹爪 DDS 时也出现错误：{cleanup_error}")
            raise

    def _fresh(self, side: str, now: float) -> tuple[float, float] | None:
        """检查反馈时间窗口，绝不以目标值替换缺失的实测值."""
        feedback = self._feedback.get(side)
        # DDS 回调可能在主循环本次取时之后几毫秒到达。
        if feedback is not None and -CONTROL_PERIOD_S <= now - feedback[0] <= self.config.feedback_timeout_s:
            return feedback
        return None

    def _snapshot(self, now: float) -> dict[str, tuple[float, float]] | None:
        """只有冻结集合中所有电机均新鲜才生成整夹爪反馈."""
        if not self._online:
            return None
        snapshot = {}
        for side in self._online:
            feedback = self._fresh(side, now)
            if feedback is None:
                return None
            snapshot[side] = feedback
        return snapshot

    def _normalize(self, position: float) -> float:
        """将电机弧度反馈投影到端点限定的归一化开度."""
        width = self.config.pos_open_rad - self.config.pos_close_rad
        return float(np.clip((position - self.config.pos_close_rad) / width, 0.0, 1.0))

    def _openness(self, raw: float) -> float:
        """将归一化电机位置转换回模型使用的开度刻度."""
        return float(np.interp(raw, self._mapping[:, 1], self._mapping[:, 0]))

    def read_state(self) -> GripperState | None:
        """逆向映射实际电机反馈，时间戳采用在线侧中最早值."""
        feedback = self._snapshot(self._clock())
        if feedback is None:
            return None
        positions = {side: sample[1] for side, sample in feedback.items()}
        raw = sum(self._normalize(q) for q in positions.values()) / len(positions)
        return GripperState(
            min(sample[0] for sample in feedback.values()), self._openness(raw), raw, positions
        )

    def at_target(self, feedback: GripperState, openness: float) -> bool:
        """逐侧比较实测开度，避免平均值掩盖单侧未到位."""
        positions = feedback.motor_positions
        return bool(self._online) and all(
            side in positions
            and abs(self._openness(self._normalize(positions[side])) - openness) <= TARGET_TOLERANCE
            for side in self._online
        )

    def _check_motion(self) -> None:
        """在驱动边界禁止 dry-run 的任何夹爪指令."""
        if not self._motion_enabled:
            raise RuntimeError("夹爪运动接口未启用")
        if not self._online:
            raise RuntimeError("夹爪未连接")

    def set_target(self, openness: float) -> None:
        """更新模型刻度开度目标，下一控制周期才会下发."""
        self._check_motion()
        if not math.isfinite(openness) or not 0.0 <= openness <= 1.0:
            raise ValueError("夹爪开度必须为 [0, 1] 内的有限值")
        raw = float(np.interp(openness, self._mapping[:, 0], self._mapping[:, 1]))
        position = self.config.pos_close_rad + raw * (self.config.pos_open_rad - self.config.pos_close_rad)
        self._target_positions = dict.fromkeys(self._online, position)

    def hold(self) -> None:
        """冻结每侧最后下发位置，首条命令前用真实反馈建立保持目标."""
        self._check_motion()
        feedback = self._snapshot(self._clock())
        if feedback is None:
            raise RuntimeError("夹爪反馈失效，无法建立保持目标")
        self._target_positions = {
            side: self._last_sent.get(side, sample[1]) for side, sample in feedback.items()
        }

    def tick(self, now: float) -> None:
        """检查所有侧的速度与误差限制后逐侧发送连续目标."""
        self._check_motion()
        if not self._target_positions:
            return
        feedback = self._snapshot(now)
        if feedback is None:
            raise RuntimeError("夹爪反馈失效，停止发送目标")
        dt = CONTROL_PERIOD_S if self._last_tick is None else min(now - self._last_tick, CONTROL_PERIOD_S)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("夹爪控制周期必须为正数")
        max_step = self.config.max_speed_rad_s * dt
        max_error = self.config.max_hold_error_rad
        commands: dict[str, float] = {}
        for side, (_, actual) in feedback.items():
            previous = self._last_sent.get(side, actual)
            lower = max(previous - max_step, actual - max_error)
            upper = min(previous + max_step, actual + max_error)
            if lower > upper:
                raise RuntimeError(f"{side} 夹爪跟踪偏差无法同时满足限速和误差约束")
            commands[side] = float(np.clip(self._target_positions[side], lower, upper))
        self._last_tick = now
        for side, command in commands.items():
            message = self._messages[side]
            message.cmds[0].q = command
            if not self._publishers[side].Write(message):
                raise RuntimeError(f"发送 {side} 夹爪指令失败")
            self._last_sent[side] = command

    def close(self) -> None:
        """关闭 DDS 端点，不发送打开、闭合或停机命令."""
        errors: list[Exception] = []
        for channel in (*self._subscribers.values(), *self._publishers.values()):
            try:
                channel.Close()
            except Exception as error:
                errors.append(error)
        self._subscribers.clear()
        self._publishers.clear()
        self._messages.clear()
        self._feedback.clear()
        self._target_positions.clear()
        self._last_sent.clear()
        self._last_tick = None
        self._online = ()
        if errors:
            raise ExceptionGroup("释放夹爪 DDS 失败", errors)
