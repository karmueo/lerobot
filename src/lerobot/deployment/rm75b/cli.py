"""本地 RM75B 部署命令入口."""

from __future__ import annotations

import argparse
import errno
import fcntl
import importlib
import json
import logging
import os
import select
import signal
import socket
import stat
import sys
import termios
import time
import tty
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

from .config import REPO_ROOT, load_assets, load_config
from .telemetry import RunLog, build_manifest, json_value

if TYPE_CHECKING:
    from .config import DeploymentAssets, DeploymentConfig
    from .runtime import SerialDeployment

logger = logging.getLogger(__name__)
_DEFAULT_CONFIG = REPO_ROOT / "docs/umi/rm75b_deployment.yaml"
# TTY 不报告松键，持续到来的同键字节只延长静默窗口。
_MOTION_KEY_QUIET_S = 1.0


class _SessionExit(KeyboardInterrupt):
    """把终端挂断、终止和 Ctrl+C 转成会话退出."""

    def __init__(self, signum: int) -> None:
        """保留退出状态所需的信号编号."""
        self.signum = signum


class _SessionSignals:
    """清理阶段只记录中断，避免二次信号打断设备与进程回收."""

    def __init__(self) -> None:
        """初始化会话信号记录."""
        self.signum: int | None = None
        self.cleaning = False

    def receive(self, signum: int, frame: FrameType | None) -> None:
        """信号处理不做硬件调用或阻塞等待."""
        if self.signum is None:
            self.signum = signum
            if not self.cleaning:
                raise _SessionExit(signum)

    @contextmanager
    def defer_launch(self) -> Iterator[None]:
        """先接收新进程句柄，再处理启动瞬间到来的退出信号."""
        previous = self.cleaning
        self.cleaning = True
        try:
            yield
        finally:
            self.cleaning = previous
        if self.signum is not None and not previous:
            raise _SessionExit(self.signum)


@contextmanager
def _session_signals() -> Iterator[_SessionSignals]:
    """临时接管交互会话的三种退出信号，完成后恢复上游处理器."""
    state = _SessionSignals()
    saved = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            saved[sig] = signal.signal(sig, state.receive)
        yield state
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


# 可能占用机械臂、夹爪或相机的本机进程。厂商 DDS 服务端不在此列表中。
_CONTROLLER_NAMES = frozenset(
    {
        "rm_driver",
        "arm_control_service",
        "rm_75_initial_pose",
        "rm75_joint_controller",
        "rm75_placo_controller",
        "dex1_gripper_node",
        "gripper_node",
        "gripper_bridge",
        "gripper_control.py",
        "gripper_control_node.py",
        "canfd_angle_demo.py",
        "canfd_pose_demo.py",
        "usb_camera_node",
        "usb_camera_ffmpeg",
        "lerobot-deploy-rm75b",
        "lerobot.deployment.rm75b.cli",
    }
)


@contextmanager
def deployment_lock(path: Path) -> Iterator[None]:
    """锁定文件描述符，已有部署持锁时不连接硬件."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(f"部署锁文件类型或所有者无效 {path}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有 RM75B 部署进程持锁 {path}") from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(fd)


def _own_ancestry(proc_root: Path) -> set[int]:
    """识别当前进程及 uv 等上游启动器，避免将自己判为冲突."""
    own = {os.getpid()}
    pid = os.getpid()
    while pid > 1:
        try:
            lines = (proc_root / str(pid) / "status").read_text().splitlines()
            parent = next(line for line in lines if line.startswith("PPid:"))
            pid = int(parent.split()[1])
        except (OSError, ValueError, IndexError, StopIteration):
            break
        if pid <= 0 or pid in own:
            break
        own.add(pid)
    return own


def conflicting_processes(proc_root: Path = Path("/proc")) -> list[str]:
    """读取进程命令行识别已知的硬件占用进程."""
    conflicts = []
    own = _own_ancestry(proc_root)
    for entry in sorted(proc_root.iterdir()):
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            # 进程可能已退出或属于其他用户。
            continue
        names = {Path(arg).name for arg in argv if arg}
        matched = names & _CONTROLLER_NAMES
        offline = any(arg in ("--mode=inspect", "--mode=benchmark") for arg in argv) or any(
            flag == "--mode" and value in ("inspect", "benchmark")
            for flag, value in zip(argv, argv[1:], strict=False)
        )
        if offline:
            matched -= {"lerobot-deploy-rm75b", "lerobot.deployment.rm75b.cli"}
        if matched:
            conflicts.append(f"PID {entry.name}: {', '.join(sorted(matched))}")
    return conflicts


def _probe(check: Callable[[], object]) -> dict[str, Any]:
    """允许检查模式报告当前机器缺少的设备和依赖."""
    try:
        check()
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}
    return {"ok": True}


def inspect_hardware(cfg: DeploymentConfig) -> dict[str, Any]:
    """检查本机 SDK、设备路径与网口，不创建机器人或 DDS 会话."""
    from lerobot.utils.import_utils import _unitree_sdk_available, _uvc_available

    from .arm import check_udp_endpoint, load_rm_sdk

    def check_camera() -> tuple[int, int]:
        from .camera import resolve_usb_location

        return resolve_usb_location(cfg.camera.video_device)

    def check_unitree() -> None:
        if not _unitree_sdk_available:
            raise ImportError("缺少 unitree-sdk2py")
        importlib.import_module("unitree_sdk2py.core.channel")

    def check_uvc() -> None:
        if not _uvc_available:
            raise ImportError("缺少 pupil-labs-uvc")
        importlib.import_module("uvc")

    checks = {
        "rm_sdk": _probe(lambda: load_rm_sdk(cfg.arm.sdk_root)),
        "rm_udp_endpoint": _probe(lambda: check_udp_endpoint(cfg.arm.udp_ip, cfg.arm.udp_port)),
        "unitree_sdk": _probe(check_unitree),
        "gripper_network_interface": _probe(lambda: socket.if_nametoindex(cfg.gripper.network_interface)),
        "uvc": _probe(check_uvc),
        "camera_port": _probe(check_camera),
    }
    return {"ready": all(check["ok"] for check in checks.values()), "checks": checks}


@contextmanager
def _immediate_keys(fd: int) -> Iterator[None]:
    """切到无回显的逐键模式，保留 Ctrl+C，并在退出时还原终端."""
    if not os.isatty(fd):
        raise RuntimeError("部署交互模式需要 TTY，请在终端直接运行")
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd, termios.TCSAFLUSH)
        yield
    finally:
        try:
            # 先清输入，再立即恢复；退出路径不等待终端输出排空。
            try:
                termios.tcflush(fd, termios.TCIFLUSH)
            finally:
                termios.tcsetattr(fd, termios.TCSANOW, previous)
        except termios.error as exc:
            # 终端已断开时没有可恢复的终端会话。
            if exc.args[0] not in (errno.EIO, errno.ENOTTY):
                raise


def _key_command(data: bytes, now: float, last_keys: dict[str, float]) -> str | None:
    """从一次输入中选择一个动作，退出和停止优先于运动键."""
    previous = last_keys.copy()
    if b" " in data:
        last_keys["home"] = now
    if b"\r" in data or b"\n" in data:
        last_keys["start"] = now
    if b"q" in data or b"\x04" in data:
        return "quit"
    if b"\x08" in data or b"\x7f" in data:
        return "stop"
    if data and not data.strip(b" "):
        command = "home"
    elif data and not data.strip(b"\r\n"):
        command = "start"
    else:
        # 混合输入、方向键序列等都不能授权运动。
        return None
    if now - previous.get(command, -float("inf")) < _MOTION_KEY_QUIET_S:
        return None
    return command


def _terminal_loop(runtime: SerialDeployment) -> int:
    """在逐键模式内轮询输入和控制周期，退出前恢复 TTY."""
    fd = sys.stdin.fileno()
    with _immediate_keys(fd):
        return _run_key_loop(runtime, fd)


def _run_key_loop(runtime: SerialDeployment, fd: int) -> int:
    """一次读取最多执行一条命令，保持现有控制节拍."""
    from .runtime import State

    last_keys: dict[str, float] = {}
    next_tick = time.monotonic()
    runtime.tick(next_tick)
    next_tick += runtime.period
    print("已连接。空格回位，回车开始，退格停止，q 退出。", flush=True)
    while runtime.state not in (State.FAULT, State.CLOSED):
        timeout = max(0.0, next_tick - time.monotonic())
        readable, _, _ = select.select([fd], [], [], timeout)
        if readable:
            try:
                data = os.read(fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    return 0
                raise
            if not data:
                return 0
            now = time.monotonic()
            command = _key_command(data, now, last_keys)
            if command == "quit":
                return 0
            if command is not None:
                print(runtime.command(command, now), flush=True)
        now = time.monotonic()
        if now >= next_tick:
            previous = runtime.state
            runtime.tick(now)
            if previous is not State.READY and runtime.state is State.READY:
                print("状态 ready。", flush=True)
            # 发生迟到时从实际周期重新计时，避免集中补发。
            next_tick = now + runtime.period
    return 1 if runtime.state is State.FAULT else 0


def _run(
    cfg: DeploymentConfig,
    assets: DeploymentAssets,
    mode: str,
    max_chunks: int | None,
    *,
    save_inference: bool = False,
) -> int:
    """锁住本地资源后构造控制器，退出时先停臂再关闭日志."""
    if not sys.stdin.isatty():
        raise RuntimeError("run 和 dry-run 需要交互终端，不能通过管道发送按键")

    from .arm import ArmDriver
    from .camera import CameraSource
    from .gripper import GripperDriver
    from .gripper_server import GripperServer
    from .kinematics import ArmKinematics
    from .observations import ObservationAssembler
    from .policy import InferenceWorker
    from .runtime import SerialDeployment
    from .trajectory import TrajectoryExecutor

    with _session_signals() as shutdown, deployment_lock(cfg.logging.lock_path):
        conflicts = conflicting_processes()
        if conflicts:
            raise RuntimeError(f"发现可能占用部署硬件的进程，请先在对应终端退出：{', '.join(conflicts)}")
        hardware = inspect_hardware(cfg)
        if not hardware["ready"]:
            failures = {
                name: result["detail"] for name, result in hardware["checks"].items() if not result["ok"]
            }
            raise RuntimeError(f"本机硬件预检失败 {failures}")
        manifest = build_manifest(cfg, assets, mode)
        manifest["hardware_preflight"] = hardware
        manifest["max_chunks"] = max_chunks
        motion_enabled = mode == "run"
        kinematics = ArmKinematics.from_urdf(cfg.arm.urdf)
        with RunLog(cfg.logging.output_dir, manifest, save_inference=save_inference) as events:
            print(f"运行记录 {events.directory}", flush=True)
            server = (
                GripperServer(cfg.gripper, events.directory, events, launch_guard=shutdown.defer_launch)
                if motion_enabled and cfg.gripper.server_mode == "auto"
                else None
            )
            runtime = SerialDeployment(
                cfg,
                assets,
                ArmDriver(cfg.arm, motion_enabled=motion_enabled),
                GripperDriver(cfg.gripper, motion_enabled=motion_enabled),
                CameraSource(cfg.camera, retain_jpeg=save_inference),
                InferenceWorker(
                    cfg.policy, save_result=events.write_inference_output if save_inference else None
                ),
                ObservationAssembler(
                    kinematics, assets.tcp_offset, cfg.execution, save_inference=save_inference
                ),
                kinematics,
                TrajectoryExecutor(kinematics, assets.tcp_offset, cfg.execution, cfg.safety),
                events,
                motion_enabled=motion_enabled,
                max_chunks=max_chunks,
                gripper_server=server,
                save_observation=events.write_inference_input if save_inference else None,
            )
            try:
                runtime.connect(cleanup_on_error=False)
                result = _terminal_loop(runtime)
            finally:
                shutdown.cleaning = True
                try:
                    runtime.close()
                finally:
                    if server is not None:
                        server.close()
        if shutdown.signum is not None:
            raise _SessionExit(shutdown.signum)
        return result


def _parser() -> argparse.ArgumentParser:
    """定义离线检查、性能验证和显式执行模式."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG, help="部署 YAML 路径")
    parser.add_argument("--mode", choices=("inspect", "benchmark", "dry-run", "run"), default="dry-run")
    parser.add_argument("--num-inference-steps", type=int, choices=range(1, 101), metavar="1..100")
    parser.add_argument("--verify-native", action="store_true", help="测速时对照 10 个窗口与 3 个随机种子")
    parser.add_argument("--warmup", type=int, default=2, help="benchmark 预热次数")
    parser.add_argument("--samples", type=int, default=20, help="benchmark 样本数")
    parser.add_argument("--max-chunks", type=int, help="每回合最多执行的完整动作段数")
    parser.add_argument(
        "--save-inference", action="store_true", help="保存每次推理的原始 JPEG、观测状态和输出"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """为 CLI 模式按需加载实现，并返回可检查的退出状态."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.samples < 1:
        parser.error("预热次数不得为负且样本数必须为正")
    if args.verify_native and args.mode != "benchmark":
        parser.error("--verify-native 仅适用于 benchmark")
    if args.max_chunks is not None and (args.max_chunks < 1 or args.mode not in ("dry-run", "run")):
        parser.error("--max-chunks 仅适用于 dry-run/run 且必须为正")
    if args.save_inference and args.mode not in ("dry-run", "run"):
        parser.error("--save-inference 仅适用于 dry-run/run")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_config(args.config)
        if args.num_inference_steps is not None:
            cfg = replace(cfg, policy=replace(cfg.policy, num_inference_steps=args.num_inference_steps))
        assets = load_assets(cfg)
        if args.mode == "inspect":
            manifest = build_manifest(cfg, assets, args.mode)
            manifest["hardware_preflight"] = inspect_hardware(cfg)
            manifest["conflicting_processes"] = conflicting_processes()
            print(json.dumps(json_value(manifest), ensure_ascii=False, allow_nan=False, indent=2))
            return 0
        if args.mode == "benchmark":
            from .policy import benchmark

            manifest = build_manifest(cfg, assets, args.mode)
            with RunLog(cfg.logging.output_dir, manifest) as events:
                print(f"测速记录 {events.directory}", file=sys.stderr, flush=True)
                report = benchmark(
                    cfg, assets, warmup=args.warmup, samples=args.samples, verify_native=args.verify_native
                )
                events.write("benchmark", report=report)
                print(json.dumps(json_value(report), ensure_ascii=False, allow_nan=False, indent=2))
            return 0
        return _run(cfg, assets, args.mode, args.max_chunks, save_inference=args.save_inference)
    except _SessionExit as exc:
        logger.info("会话收到信号 %d，已结束运行", exc.signum)
        return 128 + exc.signum
    except KeyboardInterrupt:
        logger.info("已结束运行")
        return 130
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
