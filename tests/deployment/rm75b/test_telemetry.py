"""验证日志保存真实反馈的快照与结束证据."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from lerobot.deployment.rm75b import telemetry
from lerobot.deployment.rm75b.telemetry import RunLog
from lerobot.deployment.rm75b.types import ActionChunk, GripperState


def test_snapshot_and_episode_summary(tmp_path: Path) -> None:
    joints = np.arange(7, dtype=np.float64)
    feedback = GripperState(10.0, 0.4, 0.3, {"right": 1.5})
    with RunLog(tmp_path, {"mode": "dry-run", "local_asset": tmp_path}) as log:
        log.write("joint_command", joints=joints, applied=False, monotonic=10.0)
        joints[:] = -1
        log.write("episode_summary", episode_id=1, chunks_completed=1)
        log.write("fault", reason="传感器超时")
        log.write("closed", final_gripper=feedback, errors=[])
    manifest = json.loads((log.directory / "manifest.json").read_text())
    assert manifest["local_asset"] == str(tmp_path)
    assert manifest["save_inference"] is False
    assert not (log.directory / "inference_inputs").exists()
    assert not (log.directory / "inference_outputs").exists()
    command = json.loads((log.directory / "events.jsonl").read_text().splitlines()[0])
    assert command["joints"] == list(range(7))
    assert command["applied"] is False
    summary = json.loads((log.directory / "summary.json").read_text())
    assert summary["episodes"][0]["chunks_completed"] == 1
    assert summary["faults"] == ["传感器超时"]
    assert summary["shutdown"]["final_gripper"]["motor_positions"] == {"right": 1.5}


def test_invalid_json_data_reports_log_failure(tmp_path: Path) -> None:
    log = RunLog(tmp_path, {"mode": "run"})
    log.write("arm_feedback", invalid=float("nan"))
    with pytest.raises(RuntimeError, match="写入运行日志失败"):
        log.close()


def test_shanghai_date_rollover_and_same_second_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """上海时间跨日后仍可为同一秒的两个会话创建独立目录."""

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            return datetime(2026, 11, 10, 16, 5, 9, 123456, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(telemetry, "datetime", FixedDatetime)
    with RunLog(tmp_path, {"mode": "run"}) as first, RunLog(tmp_path, {"mode": "run"}) as second:
        assert first.directory.name == "2026_1111__0005_09-run"
        assert second.directory.name == "2026_1111__0005_09_02-run"


def test_archive_write_failure_is_reported(tmp_path: Path) -> None:
    """无法写入输出文件时，退出报告磁盘错误."""
    log = RunLog(tmp_path, {"mode": "run"}, save_inference=True)
    manifest = json.loads((log.directory / "manifest.json").read_text())
    assert manifest["save_inference"] is True
    (log.directory / "inference_outputs").rmdir()
    log.write_inference_output(ActionChunk(1, 1, 1.0, np.zeros((8, 7), dtype=np.float32)), False)
    with pytest.raises(RuntimeError, match="写入运行日志失败") as error:
        log.close()
    assert isinstance(error.value.__cause__, OSError)
