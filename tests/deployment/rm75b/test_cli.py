"""验证 CLI 模式隔离与本机互斥."""

import json
import os
import pty
import signal
import termios
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lerobot.deployment.rm75b import cli, policy
from lerobot.deployment.rm75b.config import DeploymentAssets, DeploymentConfig
from lerobot.deployment.rm75b.runtime import State


def _fake_assets() -> DeploymentAssets:
    return DeploymentAssets(
        np.eye(4),
        np.zeros(7),
        {
            "noise_scheduler_type": "DDPM",
            "num_train_timesteps": 100,
            "n_obs_steps": 3,
            "n_action_steps": 8,
        },
        {},
        {"config": "sha"},
    )


def test_inspect_reports_preflight_without_actuators(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: DeploymentConfig())
    monkeypatch.setattr(cli, "load_assets", lambda cfg: _fake_assets())
    monkeypatch.setattr(cli, "_run", lambda *args: pytest.fail("inspect 进入了运行路径"))
    monkeypatch.setattr(cli, "inspect_hardware", lambda cfg: {"ready": False, "checks": {}})
    monkeypatch.setattr(cli, "conflicting_processes", list)
    assert cli.main(["--mode", "inspect"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["hardware_preflight"]["ready"] is False
    assert manifest["policy"]["num_inference_steps"] == 100


def test_benchmark_routes_options_and_stays_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cfg = DeploymentConfig()
    cfg = replace(cfg, logging=replace(cfg.logging, output_dir=tmp_path))
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "load_assets", lambda cfg: _fake_assets())
    monkeypatch.setattr(cli, "_run", lambda *args: pytest.fail("benchmark 进入了运行路径"))
    monkeypatch.setattr(cli, "inspect_hardware", lambda *args: pytest.fail("benchmark 探测了硬件"))

    def benchmark(cfg: DeploymentConfig, assets: DeploymentAssets, **kwargs: Any) -> dict[str, Any]:
        assert cfg.policy.num_inference_steps == 16
        assert kwargs == {"warmup": 2, "samples": 20, "verify_native": True}
        return {"mock_report": 1}

    monkeypatch.setattr(policy, "benchmark", benchmark)
    assert cli.main(["--mode", "benchmark", "--verify-native", "--num-inference-steps", "16"]) == 0
    assert json.loads(capsys.readouterr().out) == {"mock_report": 1}
    manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
    assert manifest["policy"]["num_inference_steps"] == 16
    assert "benchmark" in next(tmp_path.glob("*/events.jsonl")).read_text()


@pytest.mark.parametrize("enabled", [False, True])
def test_default_mode_is_dry_run(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: DeploymentConfig())
    monkeypatch.setattr(cli, "load_assets", lambda cfg: _fake_assets())

    def run(
        cfg: DeploymentConfig,
        assets: DeploymentAssets,
        mode: str,
        max_chunks: int | None,
        *,
        save_inference: bool,
    ) -> int:
        assert mode == "dry-run"
        assert max_chunks == 1
        assert save_inference is enabled
        return 0

    monkeypatch.setattr(cli, "_run", run)
    assert cli.main(["--max-chunks", "1", *(["--save-inference"] if enabled else [])]) == 0


@pytest.mark.parametrize("mode", ["benchmark", "inspect"])
def test_archive_flag_requires_interactive_mode(mode: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--mode", mode, "--save-inference"])


def test_lock_rejects_second_controller(tmp_path: Path) -> None:
    lock = tmp_path / "control.lock"
    with cli.deployment_lock(lock), pytest.raises(RuntimeError, match="持锁"), cli.deployment_lock(lock):
        pytest.fail("第二个控制器获得了同一锁")
    with cli.deployment_lock(lock):
        assert lock.read_text().strip().isdigit()


def test_conflict_scan_leaves_vendor_server_allowed(tmp_path: Path) -> None:
    for pid, command in (
        (10001, b"/vendor/dex1_1_gripper_server\0-n\0wlP1p1s0\0"),
        (10002, b"/usr/bin/python3\0/opt/ros/bin/rm_driver\0"),
        (10003, b".venv/bin/python\0-m\0lerobot.deployment.rm75b.cli\0"),
        (10004, b"/ros/install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_node\0--ros-args\0"),
        (10005, b"/ros/install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_ffmpeg\0"),
    ):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(command)
    assert cli.conflicting_processes(tmp_path) == [
        "PID 10002: rm_driver",
        "PID 10003: lerobot.deployment.rm75b.cli",
        "PID 10004: usb_camera_node",
        "PID 10005: usb_camera_ffmpeg",
    ]


def test_conflict_scan_excludes_own_uv_and_offline_jobs(tmp_path: Path) -> None:
    for pid, parent, command in (
        (os.getpid(), 10001, b"lerobot-deploy-rm75b\0"),
        (10001, 1, b"uv\0run\0--no-sync\0lerobot-deploy-rm75b\0"),
        (10002, 1, b"uv\0run\0lerobot-deploy-rm75b\0--mode\0benchmark\0"),
        (10003, 1, b"lerobot-deploy-rm75b\0--mode=run\0"),
    ):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(command)
        (directory / "status").write_text(f"PPid:\t{parent}\n")
    assert cli.conflicting_processes(tmp_path) == ["PID 10003: lerobot-deploy-rm75b"]


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[int, int]]:
    master, slave = pty.openpty()
    try:
        with os.fdopen(os.dup(slave)) as stdin:
            monkeypatch.setattr(cli.sys, "stdin", stdin)
            yield master, slave
    finally:
        os.close(master)
        os.close(slave)


class _KeyRuntime:
    period = 0.0
    state = State.READY

    def __init__(self, master: int, slave: int, keys: list[bytes]) -> None:
        self.master, self.slave = master, slave
        self.keys = iter(keys)
        self.commands: list[str] = []

    def tick(self, now: float) -> None:
        flags = termios.tcgetattr(self.slave)[3]
        assert not flags & (termios.ICANON | termios.ECHO)
        assert flags & termios.ISIG
        os.write(self.master, next(self.keys))

    def command(self, name: str, now: float) -> str:
        self.commands.append(name)
        return name


def test_terminal_shortcuts_are_immediate_and_restore_tty(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    original = termios.tcgetattr(slave)
    # 进入逐键模式时必须丢弃模型加载期间缓存在终端里的命令。
    os.write(master, b" \r")
    runtime = _KeyRuntime(master, slave, [b" ", b" ", b"\r", b"\n", b"\x08", b"\x7f", b"q"])
    assert cli._terminal_loop(runtime) == 0
    assert runtime.commands == ["home", "start", "stop", "stop"]
    assert termios.tcgetattr(slave) == original


def test_terminal_restores_tty_on_interrupt(terminal: tuple[int, int]) -> None:
    class InterruptingRuntime(_KeyRuntime):
        def tick(self, now: float) -> None:
            super().tick(now)
            raise KeyboardInterrupt

    master, slave = terminal
    original = termios.tcgetattr(slave)
    with pytest.raises(KeyboardInterrupt):
        cli._terminal_loop(InterruptingRuntime(master, slave, [b" "]))
    assert termios.tcgetattr(slave) == original


def test_terminal_eof_closes_session_and_restores_tty(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    original = termios.tcgetattr(slave)
    runtime = _KeyRuntime(master, slave, [b"\x04"])
    assert cli._terminal_loop(runtime) == 0
    assert not runtime.commands
    assert termios.tcgetattr(slave) == original


def test_key_priority_and_held_motion_rearm() -> None:
    last: dict[str, float] = {}
    assert cli._key_command(b" \r\x7f", 1.0, last) == "stop"
    assert cli._key_command(b" ", 1.5, last) is None
    assert cli._key_command(b" ", 2.0, last) is None
    assert cli._key_command(b" ", 3.1, last) == "home"
    assert cli._key_command(b" \x08q\r", 4.0, last) == "quit"
    assert cli._key_command(b"\x1b[A", 5.0, last) is None


@pytest.mark.parametrize("mode", ["run", "dry-run"])
def test_non_tty_rejected_before_device_preflight(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli, "deployment_lock", lambda *args: pytest.fail("非终端进入了设备路径"))
    with pytest.raises(RuntimeError, match="交互终端"):
        cli._run(DeploymentConfig(), _fake_assets(), mode, 1)


@pytest.mark.parametrize("failure", [None, "connect", "connect_signal", "close", "interrupt", "term", "hup"])
def test_cli_managed_service_closes_after_runtime_on_every_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str | None
) -> None:
    order = _patch_service_session(monkeypatch, failure=failure)
    cfg = DeploymentConfig()
    cfg = replace(
        cfg, logging=replace(cfg.logging, lock_path=tmp_path / "lock", output_dir=tmp_path / "runs")
    )
    if failure is None:
        assert cli._run(cfg, _fake_assets(), "run", 1) == 0
    elif failure in ("interrupt", "term", "hup"):
        with pytest.raises(KeyboardInterrupt):
            cli._run(cfg, _fake_assets(), "run", 1)
    else:
        with pytest.raises(RuntimeError, match="测试异常"):
            cli._run(cfg, _fake_assets(), "run", 1)
    assert order == ["server_create", "server_start", "runtime_close", "server_close"]


@pytest.mark.parametrize(("mode", "server_mode"), [("dry-run", "auto"), ("run", "external")])
def test_non_managed_modes_do_not_construct_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, server_mode: str
) -> None:
    order = _patch_service_session(monkeypatch)
    cfg = DeploymentConfig()
    cfg = replace(
        cfg,
        gripper=replace(cfg.gripper, server_mode=server_mode),
        logging=replace(cfg.logging, lock_path=tmp_path / "lock", output_dir=tmp_path / "runs"),
    )
    assert cli._run(cfg, _fake_assets(), mode, 1) == 0
    assert order == ["runtime_close"]


def _patch_service_session(monkeypatch: pytest.MonkeyPatch, *, failure: str | None = None) -> list[str]:
    from lerobot.deployment.rm75b import (
        arm,
        camera,
        gripper,
        gripper_server,
        kinematics,
        observations,
        runtime,
        trajectory,
    )

    order: list[str] = []
    for module, name in (
        (arm, "ArmDriver"),
        (gripper, "GripperDriver"),
        (camera, "CameraSource"),
        (policy, "InferenceWorker"),
        (observations, "ObservationAssembler"),
        (trajectory, "TrajectoryExecutor"),
    ):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: object())
    monkeypatch.setattr(kinematics.ArmKinematics, "from_urdf", lambda path: object())
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "conflicting_processes", list)
    monkeypatch.setattr(cli, "inspect_hardware", lambda cfg: {"ready": True})

    class Server:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            order.append("server_create")

        def close(self) -> None:
            order.append("server_close")

    class Runtime:
        def __init__(self, *args: Any, gripper_server: Server | None, **kwargs: Any) -> None:
            self.server = gripper_server

        def connect(self, *, cleanup_on_error: bool) -> None:
            assert cleanup_on_error is False
            if self.server is not None:
                order.append("server_start")
            if failure in ("connect", "connect_signal"):
                raise RuntimeError("测试异常：连接失败")

        def close(self) -> None:
            if failure == "connect_signal":
                # 普通连接错误后的首次信号也不能打断统一清理。
                os.kill(os.getpid(), signal.SIGHUP)
            order.append("runtime_close")
            if failure == "close":
                raise RuntimeError("测试异常：关闭失败")

    def loop(controller: Runtime) -> int:
        if failure == "interrupt":
            os.kill(os.getpid(), signal.SIGINT)
        elif failure == "term":
            os.kill(os.getpid(), signal.SIGTERM)
        elif failure == "hup":
            os.kill(os.getpid(), signal.SIGHUP)
        return 0

    monkeypatch.setattr(gripper_server, "GripperServer", Server)
    monkeypatch.setattr(runtime, "SerialDeployment", Runtime)
    monkeypatch.setattr(cli, "_terminal_loop", loop)
    return order
