"""验证厂商服务生命周期，仅使用进程和时钟替身."""

import os
import signal
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lerobot.deployment.rm75b import gripper_server
from lerobot.deployment.rm75b.config import GripperConfig, load_config
from lerobot.deployment.rm75b.gripper_server import GripperServer, ServerProcess


class _Events:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def write(self, event: str, **values: Any) -> None:
        self.records.append((event, values))


class _Child:
    pid = 4777

    def __init__(
        self, *, hang_on_term: bool = False, code: int | None = None, descendant: bool = False
    ) -> None:
        self.returncode = code
        self.signals: list[signal.Signals] = []
        self.waits: list[float] = []
        self.hang_on_term = hang_on_term
        self.descendant = descendant

    def wait(self, timeout: float) -> int:
        self.waits.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-vendor", timeout)
        return self.returncode


def _fake_child(
    monkeypatch: pytest.MonkeyPatch,
    child: _Child,
    *,
    ready: bool = True,
) -> None:
    monkeypatch.setattr(gripper_server, "find_server_processes", list)

    def launch(args: list[str], **options: Any) -> _Child:
        assert args[0] == "bash" and args[-1] == "--run"
        assert options["start_new_session"] is True
        assert options["stdin"] == subprocess.DEVNULL
        assert options["env"]["RM75B_GRIPPER_NETWORK"] == "testnic"
        assert options["env"]["RM75B_GRIPPER_VENDOR"] == "/tmp/test-vendor"
        if ready:
            options["stdout"].write(b"Dex1-1 Gripper Server started.\n")
            options["stdout"].flush()
        return child

    def send_group(pid: int, sig: signal.Signals) -> None:
        assert pid == child.pid
        child.signals.append(sig)
        if sig == signal.SIGKILL or not child.hang_on_term:
            if child.returncode is None:
                child.returncode = -sig
            child.descendant = False

    monkeypatch.setattr(gripper_server.subprocess, "Popen", launch)
    monkeypatch.setattr(gripper_server.os, "killpg", send_group)
    monkeypatch.setattr(GripperServer, "_exit_code", staticmethod(lambda process: process.returncode))
    monkeypatch.setattr(gripper_server, "_group_has_children", lambda group: child.descendant)
    monkeypatch.setattr(gripper_server, "_TERM_TIMEOUT_S", 0.01)


@pytest.fixture
def cfg() -> GripperConfig:
    return GripperConfig(network_interface="testnic", server_vendor_dir=Path("/tmp/test-vendor"))


def test_owned_child_ready_then_terminate_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: GripperConfig
) -> None:
    child = _Child()
    _fake_child(monkeypatch, child)
    events = _Events()
    service = GripperServer(cfg, tmp_path, events)
    service.start()
    service.check_alive()
    assert "Gripper Server started" in (tmp_path / "gripper-server.log").read_text()
    service.close()
    service.close()
    assert child.signals == [signal.SIGTERM]
    assert child.waits == [0]
    assert events.records[-1][1]["action"] == "terminate"


def test_owned_child_escalates_only_after_term_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: GripperConfig
) -> None:
    child = _Child(hang_on_term=True)
    _fake_child(monkeypatch, child)
    events = _Events()
    service = GripperServer(cfg, tmp_path, events)
    service.start()
    service.close()
    assert child.signals == [signal.SIGTERM, signal.SIGKILL]
    assert child.waits == [0]
    assert events.records[-1][1]["action"] == "kill"


def test_startup_timeout_closes_owned_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: GripperConfig
) -> None:
    child = _Child()
    _fake_child(monkeypatch, child, ready=False)
    now = 0.0

    def sleep(duration: float) -> None:
        nonlocal now
        now += duration

    monkeypatch.setattr(gripper_server, "time", SimpleNamespace(monotonic=lambda: now, sleep=sleep))
    service = GripperServer(replace(cfg, server_start_timeout_s=0.04), tmp_path, _Events())
    try:
        with pytest.raises(TimeoutError, match="gripper-server.log"):
            service.start()
    finally:
        service.close()
    assert child.signals == [signal.SIGTERM]


def test_child_early_exit_reports_log_and_does_not_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: GripperConfig
) -> None:
    child = _Child(code=9)
    _fake_child(monkeypatch, child, ready=False)
    events = _Events()
    service = GripperServer(cfg, tmp_path, events)
    try:
        with pytest.raises(RuntimeError, match=r"状态 9.*gripper-server.log"):
            service.start()
    finally:
        service.close()
    assert not child.signals
    assert events.records[-1][1]["action"] == "already_exited"


def test_wrapper_exits_before_same_group_descendant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: GripperConfig
) -> None:
    child = _Child(code=9, descendant=True, hang_on_term=True)
    _fake_child(monkeypatch, child, ready=False)
    service = GripperServer(cfg, tmp_path, _Events())
    try:
        with pytest.raises(RuntimeError, match="状态 9"):
            service.start()
    finally:
        service.close()
    assert child.signals == [signal.SIGTERM, signal.SIGKILL]
    assert not child.descendant and child.waits == [0]


def test_reused_child_death_detected_without_signals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: GripperConfig
) -> None:
    fd, peer = os.pipe()
    monkeypatch.setattr(
        gripper_server, "find_server_processes", lambda: [ServerProcess(9900, "12", "testnic")]
    )
    monkeypatch.setattr(gripper_server, "_start_time", lambda path: "12")
    monkeypatch.setattr(gripper_server.os, "pidfd_open", lambda pid: fd)
    monkeypatch.setattr(gripper_server.os, "killpg", lambda *args: pytest.fail("复用服务不应发信号"))
    monkeypatch.setattr(
        gripper_server.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("不应重复启动")
    )
    events = _Events()
    service = GripperServer(cfg, tmp_path, events)
    try:
        service.start()
        service.check_alive()
        os.write(peer, b"exit")
        with pytest.raises(RuntimeError, match="9900.*已退出"):
            service.check_alive()
    finally:
        service.close()
        os.close(peer)
    assert events.records[-1] == ("gripper_server_released", {"pid": 9900, "owned": False})
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize(
    ("running", "message"),
    [
        ([ServerProcess(9900, "12", "other")], "网卡"),
        ([ServerProcess(9900, "12", "testnic"), ServerProcess(9901, "14", "testnic")], "多个"),
    ],
)
def test_conflicting_existing_services_fail_without_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cfg: GripperConfig,
    running: list[ServerProcess],
    message: str,
) -> None:
    monkeypatch.setattr(gripper_server, "find_server_processes", lambda: running)
    monkeypatch.setattr(gripper_server.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("不应启动"))
    with pytest.raises(RuntimeError, match=message):
        GripperServer(cfg, tmp_path, _Events()).start()


def test_proc_detection_requires_executable_and_reads_interface(tmp_path: Path) -> None:
    for pid, name in ((12, "dex1_1_gripper_server"), (14, "bash")):
        path = tmp_path / str(pid)
        path.mkdir()
        (path / "exe").symlink_to(f"/vendor/{name}")
        (path / "cmdline").write_bytes(b"/vendor/dex1_1_gripper_server\0-n\0testnic\0")
        (path / "stat").write_text(f"{pid} (proc with space) S " + " ".join(["0"] * 18 + ["123"]))
    assert gripper_server.find_server_processes(tmp_path) == [ServerProcess(12, "123", "testnic")]


def test_legacy_yaml_defaults_to_auto_and_external_opt_out(tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text("gripper:\n  network_interface: testnic\n")
    assert load_config(path).gripper.server_mode == "auto"
    path.write_text("gripper:\n  server_mode: external\n")
    assert load_config(path).gripper.server_mode == "external"
