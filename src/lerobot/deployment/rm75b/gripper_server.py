"""仅在实机模式管理本次部署启动的 Unitree 厂商进程."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import REPO_ROOT

if TYPE_CHECKING:
    from .config import GripperConfig
    from .types import EventLogger

_SERVER = "dex1_1_gripper_server"
_READY = b"Dex1-1 Gripper Server started"
_SCRIPT = REPO_ROOT / "scripts/run_rm75b_gripper_server.sh"
_TERM_TIMEOUT_S = 2.0
_KILL_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class ServerProcess:
    """按 PID 与启动时间绑定服务身份，避免进程号复用."""

    pid: int
    start_time: str
    network: str | None


def _start_time(proc: Path) -> str:
    """读取 /proc stat 第 22 列，跳过可能包含空格的进程名."""
    return (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]


def _group_has_children(group: int) -> bool:
    """清理时核对同组存活后代，跳过已退出但尚未回收的进程."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == group:
            continue
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] not in ("Z", "X") and int(fields[2]) == group:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def find_server_processes(proc_root: Path = Path("/proc")) -> list[ServerProcess]:
    """优先以真实可执行文件识别服务，兼容其他用户的受限 proc."""
    servers: list[ServerProcess] = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            try:
                exe = Path(os.readlink(proc / "exe").removesuffix(" (deleted)")).name
            except PermissionError:
                # 不能读 exe 时仍需匹配内核进程名及 argv[0]，避免仅凭命令参数判定。
                if (proc / "comm").read_text().strip() != _SERVER[:15]:
                    continue
                exe = None
            if exe not in (_SERVER, None):
                continue
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if exe is None and Path(argv[0]).name != _SERVER:
                raise RuntimeError(f"无法确认可能已运行的夹爪服务 PID {proc.name}")
            network = next((value for flag, value in zip(argv, argv[1:], strict=False) if flag == "-n"), None)
            servers.append(ServerProcess(int(proc.name), _start_time(proc), network))
        except (FileNotFoundError, ProcessLookupError):
            # 枚举时已退出的进程不参与占用判定。
            continue
    return sorted(servers, key=lambda entry: entry.pid)


class GripperServer:
    """在部署独占锁内启动、复用或回收厂商服务."""

    def __init__(
        self,
        cfg: GripperConfig,
        directory: Path,
        events: EventLogger,
        *,
        launch_guard: Callable[[], AbstractContextManager[None]] = nullcontext,
    ) -> None:
        """保存路径和配置；创建对象时不启动进程."""
        self.cfg = cfg
        self.log_path = directory / "gripper-server.log"
        self.events = events
        self._launch_guard = launch_guard
        self._process: subprocess.Popen[bytes] | None = None
        self._existing: ServerProcess | None = None
        self._pidfd: int | None = None

    def _reuse(self, server: ServerProcess) -> None:
        """绑定已有实例的 pidfd，仅监控，退出时绝不发信号."""
        if server.network != self.cfg.network_interface:
            raise RuntimeError(
                f"已有夹爪服务 PID {server.pid} 的网卡 {server.network!r} 与配置 "
                f"{self.cfg.network_interface!r} 不一致"
            )
        fd = os.pidfd_open(server.pid)
        try:
            if _start_time(Path(f"/proc/{server.pid}")) != server.start_time:
                raise RuntimeError("夹爪服务在复用期间退出，请检查服务")
            self._pidfd = fd
            self._existing = server
            self.check_alive()
        except BaseException:
            os.close(fd)
            self._pidfd = None
            self._existing = None
            raise
        self.events.write("gripper_server_reused", pid=server.pid, network=server.network, owned=False)

    def start(self) -> None:
        """在 DDS 连接前等待服务就绪；启动失败由会话清理自己创建的进程."""
        if self._process is not None or self._existing is not None:
            raise RuntimeError("当前会话已初始化夹爪服务")
        if self.cfg.domain_id != 0:
            raise RuntimeError("厂商服务使用 DDS 域 0；其他域请设 gripper.server_mode: external")
        existing = find_server_processes()
        if len(existing) > 1:
            raise RuntimeError(f"发现多个夹爪服务，请在原终端检查 PID {[p.pid for p in existing]}")
        if existing:
            self._reuse(existing[0])
            return

        env = {
            **os.environ,
            "RM75B_GRIPPER_VENDOR": str(self.cfg.server_vendor_dir),
            "RM75B_GRIPPER_NETWORK": self.cfg.network_interface,
        }
        with self.log_path.open("xb") as output, self._launch_guard():
            # 单独进程组容纳 wrapper 的依赖检查，退出只向本次创建的组发信号。
            self._process = subprocess.Popen(  # nosec B603 - 固定本地脚本，无 shell 参数展开。
                ["bash", str(_SCRIPT), "--run"],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        pid = self._process.pid
        self.events.write(
            "gripper_server_started",
            pid=pid,
            network=self.cfg.network_interface,
            owned=True,
            log=self.log_path,
        )
        deadline = time.monotonic() + self.cfg.server_start_timeout_s
        tail = b""
        with self.log_path.open("rb") as output:
            while (remaining := deadline - time.monotonic()) > 0:
                self.check_alive()
                data = tail + output.read(65536)
                if _READY in data:
                    self.check_alive()
                    others = [item.pid for item in find_server_processes() if item.pid != pid]
                    if others:
                        raise RuntimeError(f"启动期间发现其他夹爪服务 PID {others}，请检查重复启动")
                    self.events.write("gripper_server_ready", pid=pid, owned=True)
                    return
                tail = data[-len(_READY) :]
                time.sleep(min(0.02, remaining))
        raise TimeoutError(
            f"夹爪服务启动超时（{self.cfg.server_start_timeout_s:g} 秒），日志 {self.log_path}"
        )

    def check_alive(self) -> None:
        """不阻塞控制周期；已退出的服务立即使会话进入故障."""
        if self._process is not None:
            if (code := self._exit_code(self._process)) is not None:
                raise RuntimeError(
                    f"夹爪服务 PID {self._process.pid} 已退出（状态 {code}），日志 {self.log_path}"
                )
        elif self._existing is not None:
            assert self._pidfd is not None
            poller = select.poll()
            poller.register(self._pidfd, select.POLLIN)
            if poller.poll(0):
                raise RuntimeError(f"已有夹爪服务 PID {self._existing.pid} 已退出，请查看其原终端")

    def close(self) -> None:
        """释放监控或限时关闭自有服务；由调用方在断开控制设备后执行."""
        if self._pidfd is not None:
            os.close(self._pidfd)
            self._pidfd = None
        if self._existing is not None:
            pid, self._existing = self._existing.pid, None
            self.events.write("gripper_server_released", pid=pid, owned=False)
        process, self._process = self._process, None
        if process is None:
            return

        action = "already_exited"
        try:
            if self._group_alive(process):
                action = "terminate"
                self._signal_group(process, signal.SIGTERM)
                try:
                    self._wait_group(process, _TERM_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    action = "kill"
                    self._signal_group(process, signal.SIGKILL)
                    self._wait_group(process, _KILL_TIMEOUT_S)
            # 最后才回收组长，使此前发送信号时组号不可能被复用。
            process.wait(timeout=0)
        except Exception as exc:
            self.events.write("gripper_server_close_failed", pid=process.pid, action=action, reason=str(exc))
            raise RuntimeError(f"关闭夹爪服务失败，PID {process.pid}，日志 {self.log_path}") from exc
        self.events.write(
            "gripper_server_closed", pid=process.pid, owned=True, action=action, returncode=process.returncode
        )

    @staticmethod
    def _exit_code(process: subprocess.Popen[bytes]) -> int | None:
        """查询组长退出状态，不回收 PID，以便退出时安全清理同组后代."""
        status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if status is None:
            return None
        return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status

    def _group_alive(self, process: subprocess.Popen[bytes]) -> bool:
        """组长已退出时仍检查 wrapper 可能留下的依赖检查进程."""
        return self._exit_code(process) is None or _group_has_children(process.pid)

    def _wait_group(self, process: subprocess.Popen[bytes], timeout: float) -> None:
        """用有界等待回收整组，组长退出不等于同组进程已结束."""
        deadline = time.monotonic() + timeout
        while self._group_alive(process):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(_SERVER, timeout)
            time.sleep(min(0.02, remaining))

    @staticmethod
    def _signal_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        """只向 Popen 创建的新进程组发送信号，处理自然退出竞态."""
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
