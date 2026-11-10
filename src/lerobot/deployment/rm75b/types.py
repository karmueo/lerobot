"""部署模块共享的数据契约；内部统一使用米、弧度和单调时钟秒."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class ArmState:
    """来自机械臂真实反馈的七轴状态；故障由驱动解码，不伪造反馈."""

    timestamp: float
    joints: FloatArray
    velocities: FloatArray
    errors: tuple[int, ...] = ()


@dataclass(frozen=True)
class GripperState:
    """训练刻度开度及原始电机归一化开度，均来自在线侧反馈."""

    timestamp: float
    openness: float
    raw_openness: float
    motor_positions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraFrame:
    """主机收到完整图像时打时间戳；rgb 为 HWC uint8 原始图像."""

    frame_id: int
    timestamp: float
    rgb: ImageArray
    jpeg: bytes | None = None  # 仅存档时保留相机交付的原始 JPEG。


@dataclass(frozen=True)
class ObservationArchive:
    """存档专用的原始帧与对应关节角，仅在主进程中使用."""

    jpegs: tuple[bytes, ...]
    joints: FloatArray  # 按帧时间插值的七轴关节角，单位为弧度。


@dataclass(frozen=True)
class ObservationWindow:
    """已同步的三帧观测，不包含虚构的重复帧或运行期硬件对象."""

    episode_id: int
    request_id: int
    frame_ids: tuple[int, ...]
    timestamps: tuple[float, ...]
    images: NDArray[np.float32]  # (3, 3, 134, 240)，RGB、CHW、0～1。
    states: NDArray[np.float32]  # (3, 7)，回合起始 TCP 系的 xyz/rotvec/gripper。
    anchor: FloatArray  # T_base_start，整回合固定。
    capture_tcp: FloatArray  # 窗口最后一帧的 T_base_TCP，用于推理漂移检查。
    archive: ObservationArchive | None = None


@dataclass(frozen=True)
class ActionChunk:
    """原生后处理器输出的八步目标；不是关节角或逐步增量."""

    episode_id: int
    request_id: int
    observation_timestamp: float
    actions: NDArray[np.float32]  # (8, 7)。
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class JointTrajectory:
    """含实测起点和所有路点的完整预检轨迹，时间从执行开始计."""

    times: FloatArray
    joints: FloatArray
    grippers: FloatArray
    tcp_targets: FloatArray

    @property
    def duration(self) -> float:
        """整段执行时长，包含按关节速度限制增加的时间."""
        return float(self.times[-1])


class ArmInterface(Protocol):
    """驱动层严格区分反馈读取和产生运动的调用."""

    def connect(self) -> None:
        """建立反馈连接，不产生运动."""
        ...

    def read_state(self) -> ArmState | None:
        """返回最近真实反馈，未收到反馈时返回 None."""
        ...

    def send_joint_target(self, joints: FloatArray) -> None:
        """发送七轴弧度目标，禁用运动时拒绝调用."""
        ...

    def home(self, joints: FloatArray, speed: int = 10) -> None:
        """启动非阻塞回位，速度参数为控制器百分比."""
        ...

    def stop(self) -> None:
        """停止机械臂；只读模式不得调用."""
        ...

    def close(self) -> None:
        """释放反馈和连接资源，不执行回位."""
        ...


class GripperInterface(Protocol):
    """主控制循环负责 tick，目标更新不直接堵塞执行线程."""

    def connect(self) -> None:
        """发现有反馈的电机，不自动张开或闭合."""
        ...

    def read_state(self) -> GripperState | None:
        """返回真实反馈的训练刻度开度."""
        ...

    def set_target(self, openness: float) -> None:
        """更新训练刻度目标，实际发送在 tick 中完成."""
        ...

    def at_target(self, feedback: GripperState, openness: float) -> bool:
        """确认所有在线侧的实测开度均在目标容差内."""
        ...

    def tick(self, now: float) -> None:
        """按单调时间更新速度受限的电机命令."""
        ...

    def hold(self) -> None:
        """停止开度轨迹推进，保留当前保持目标."""
        ...

    def close(self) -> None:
        """关闭通道，不自动释放物体."""
        ...


class CameraInterface(Protocol):
    """读取自上次调用以来尚未消费的有界图像缓存."""

    def connect(self) -> None:
        """协商采集模式并启动采集线程."""
        ...

    def read_frames(self) -> list[CameraFrame]:
        """取出本次尚未消费的有界图像缓存."""
        ...

    def close(self) -> None:
        """终止采集并释放设备."""
        ...


class InferenceInterface(Protocol):
    """只有一个在途请求；poll 非阻塞，错误应显式抛出."""

    @property
    def busy(self) -> bool:
        """包括已经失效但仍需收取结果的在途请求."""
        ...

    def start(self) -> None:
        """运动开始前加载模型，启动失败必须报告."""
        ...

    def submit(self, window: ObservationWindow) -> None:
        """提交唯一在途观测窗口，不阻塞控制周期."""
        ...

    def poll(self) -> ActionChunk | None:
        """非阻塞接收有效结果或报告子进程异常."""
        ...

    def invalidate(self) -> None:
        """废弃在途结果，防止停止后执行旧动作."""
        ...

    def close(self) -> None:
        """限时关闭子进程及队列."""
        ...


class EventLogger(Protocol):
    """运行日志接口，复杂数组由实现转换为可序列化内容."""

    def write(self, event: str, **values: Any) -> None:
        """记录有时间戳的事件和关联值."""
        ...
