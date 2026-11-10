"""测试标定 CLI 的输出隔离、拒绝证据和只读采集路径."""

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from lerobot.deployment.rm75b import calibration_cli, calibration_vision, camera, cli
from lerobot.deployment.rm75b.config import CameraConfig
from lerobot.deployment.rm75b.types import CameraFrame, ImageArray


def _report(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "report.json").read_text())


def test_template_rejects_unfilled_measurements_then_solves(tmp_path: Path) -> None:
    template = tmp_path / "template"
    assert calibration_cli.main(["template", "--output-dir", str(template)]) == 0
    measurements = template / "measurements.yaml"
    template_text = measurements.read_text()
    values = yaml.safe_load(template_text)
    assert values["marker_size_mm"] is None
    assert values["samples"] == [{"x_mm": None, "y_mm": None, "z_mm": None}] * 3

    rejected = tmp_path / "rejected"
    args = ["solve", "--measurements", str(measurements), "--output-dir"]
    assert calibration_cli.main([*args, str(rejected)]) == 1
    assert _report(rejected)["status"] == "rejected"
    assert not (rejected / "candidate.yaml").exists()
    assert measurements.read_text() == template_text

    values.update(marker_size_mm=20.0, gripper_openness=0.5)
    values["samples"] = [{"x_mm": 1.0, "y_mm": 30.0, "z_mm": -10.0}] * 3
    measurements.write_text(yaml.safe_dump(values))
    solved = tmp_path / "solved"
    assert calibration_cli.main([*args, str(solved)]) == 0
    candidate = yaml.safe_load((solved / "candidate.yaml").read_text())
    assert candidate == _report(solved)
    assert candidate["verified"] is False
    np.testing.assert_allclose(np.asarray(candidate["T_d_tag_tcp"])[:3, 3], [0.001, 0.03, -0.01])


def test_existing_output_dir_keeps_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "measurements.yaml"
    path.write_text("保留")
    assert calibration_cli.main(["template", "--output-dir", str(tmp_path)]) == 1
    assert path.read_text() == "保留"
    assert not (tmp_path / "report.json").exists()


def _image_args(tmp_path: Path, output: Path) -> list[str]:
    return [
        "image",
        "--image",
        str(tmp_path / "source.png"),
        "--camera-calibration",
        str(tmp_path / "camera.yaml"),
        "--marker-size-mm",
        "20",
        "--candidate",
        str(tmp_path / "candidate.yaml"),
        "--measured-tag-distance-mm",
        "54.2",
        "--tcp-pixel",
        "20",
        "30",
        "--fastumi-root",
        str(tmp_path),
        "--output-dir",
        str(output),
    ]


def test_image_writes_overlay_and_forwards_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def analyze(image_path: Path, camera_path: Path, **kwargs: Any) -> tuple[dict[str, Any], ImageArray]:
        assert image_path == tmp_path / "source.png"
        assert camera_path == tmp_path / "camera.yaml"
        assert kwargs == {
            "fastumi_root": tmp_path,
            "marker_size_mm": 20.0,
            "candidate_path": tmp_path / "candidate.yaml",
            "measured_tag_distance_mm": 54.2,
            "observed_tcp_px": (20.0, 30.0),
        }
        return {"accepted": True}, np.full((4, 5, 3), [1, 2, 3], dtype=np.uint8)

    monkeypatch.setattr(calibration_vision, "analyze_image", analyze)
    output = tmp_path / "image"
    assert calibration_cli.main(_image_args(tmp_path, output)) == 0
    assert _report(output) == {"accepted": True}
    np.testing.assert_array_equal(cv2.imread(str(output / "overlay.png"))[0, 0], [1, 2, 3])


def test_rejected_image_keeps_partial_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def analyze(*args: Any, **kwargs: Any) -> None:
        raise calibration_vision.CalibrationImageError("缺少 ID 1", {"detected_ids": [0]})

    monkeypatch.setattr(calibration_vision, "analyze_image", analyze)
    output = tmp_path / "rejected"
    assert calibration_cli.main(_image_args(tmp_path, output)) == 1
    report = _report(output)
    assert report["status"] == "rejected"
    assert report["reason"] == "缺少 ID 1"
    assert report["detected_ids"] == [0]
    assert not (output / "overlay.png").exists()


@pytest.fixture
def capture_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"logging": {"lock_path": str(tmp_path / "devices.lock")}}))
    monkeypatch.setattr(cli, "conflicting_processes", list)
    return path


@pytest.mark.parametrize("timeout", [False, True])
def test_capture_uses_only_mock_camera_and_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capture_config: Path, timeout: bool
) -> None:
    from lerobot.deployment.rm75b import arm, gripper

    def reject_motor(*args: Any, **kwargs: Any) -> None:
        pytest.fail("相机采集不应创建执行器")

    monkeypatch.setattr(arm, "ArmDriver", reject_motor)
    monkeypatch.setattr(gripper, "GripperDriver", reject_motor)
    rgb = np.full((20, 36, 3), [17, 43, 232], dtype=np.uint8)
    calls: list[str] = []

    class FakeCamera:
        def __init__(self, config: CameraConfig) -> None:
            calls.append("create")

        def connect(self) -> None:
            calls.append("connect")

        def read_frames(self) -> list[CameraFrame]:
            return [CameraFrame(5, time.monotonic(), rgb)]

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(camera, "CameraSource", FakeCamera)
    if timeout:
        monkeypatch.setattr(calibration_cli, "_CAPTURE_TIMEOUT_S", 0.0)
    output = tmp_path / "capture"
    result = calibration_cli.main(["capture", "--config", str(capture_config), "--output-dir", str(output)])
    assert calls == ["create", "connect", "close"]
    report = _report(output)
    if timeout:
        assert result == 1
        assert report["status"] == "rejected"
        assert not (output / "raw_rgb.png").exists()
    else:
        assert result == 0
        assert report["frame_id"] == 5
        assert report["raw_rgb"]["shape_hwc"] == [20, 36, 3]
        assert report["model_rgb"]["shape_hwc"] == [134, 240, 3]
        np.testing.assert_array_equal(cv2.imread(str(output / "raw_rgb.png"))[..., ::-1], rgb)
        np.testing.assert_array_equal(cv2.imread(str(output / "model_rgb.png"))[0, 0, ::-1], rgb[0, 0])


def test_capture_rejects_camera_conflict_before_opening_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capture_config: Path
) -> None:
    monkeypatch.setattr(cli, "conflicting_processes", lambda: ["PID 123: usb_camera_node"])
    monkeypatch.setattr(camera, "CameraSource", lambda *args: pytest.fail("已占用的相机不应打开"))
    output = tmp_path / "conflict"
    assert (
        calibration_cli.main(["capture", "--config", str(capture_config), "--output-dir", str(output)]) == 1
    )
    assert "usb_camera_node" in _report(output)["reason"]
