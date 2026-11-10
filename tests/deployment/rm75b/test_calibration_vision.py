"""验证离线标签质量检查、候选外参与独立像素证据."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from lerobot.deployment.rm75b import calibration_vision as vision
from lerobot.deployment.rm75b.types import FloatArray


class _Camera:
    """提供独立的合成像素投影，便于核对残差符号和单位."""

    width = 120
    height = 80
    K = np.array([[100.0, 0.0, 60.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    D = np.zeros(4)

    def project(self, points: FloatArray) -> FloatArray:
        pixels = points @ self.K.T
        return pixels[:, :2] / pixels[:, 2:]


def _synthetic_pose() -> FloatArray:
    """标签正面朝向相机，中心位于相机光轴前 300 mm."""
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 0.3
    return pose


@pytest.fixture
def synthetic_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """构建原分辨率图像与独立可调的双标签检测结果."""
    image = tmp_path / "raw.png"
    assert cv2.imwrite(str(image), np.zeros((80, 120, 3), dtype=np.uint8))
    intrinsics = tmp_path / "camera.yaml"
    intrinsics.write_text("cam0: {}\n", encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    dtag_from_tcp = np.eye(4)
    dtag_from_tcp[1, 3] = 0.018
    payload = {
        "T_d_tag_tcp": dtag_from_tcp.tolist(),
        "frame": "deployment_dual_tag",
        "verified": False,
        "tag_ids": [0, 1],
        "marker_size_mm": 16,
        "gripper_openness": 1.0,
        "measured_tag_distance_mm": 132.5,
    }
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    tags = []
    for tag_id, x in enumerate((-0.0665, 0.0665)):
        pose = _synthetic_pose()
        pose[0, 3] = x
        tags.append(
            SimpleNamespace(
                tag_id=tag_id,
                camera_from_tag=pose,
                corners_px=np.ones((4, 2)),
                reprojection_rms_px=0.5,
            )
        )
    observation = SimpleNamespace(
        tag0=tags[0],
        tag1=tags[1],
        reprojection_rms_px=0.5,
        normal_angle_deg=1.0,
        coplanarity_deg=0.5,
        tag_distance_m=0.133,
        camera_from_dtag=_synthetic_pose(),
    )
    board = SimpleNamespace(load_fisheye_camera=lambda path: _Camera())
    dual = SimpleNamespace(
        DualTagEstimator=lambda *args: SimpleNamespace(estimate=lambda image: observation),
        tag_object_points=lambda size: np.array(
            [
                [-size / 2, size / 2, 0],
                [size / 2, size / 2, 0],
                [size / 2, -size / 2, 0],
                [-size / 2, -size / 2, 0],
            ]
        ),
        draw_observation=lambda image, *args: image.copy(),
    )
    monkeypatch.setattr(vision, "_load_fastumi", lambda root, report: (board, dual))
    return SimpleNamespace(
        image=image,
        intrinsics=intrinsics,
        candidate=candidate,
        payload=payload,
        observation=observation,
        root=tmp_path,
    )


def _analyze(files: SimpleNamespace, **kwargs: Any) -> tuple[dict[str, Any], np.ndarray]:
    """调用正式接口，保持所测外部资源均由测试提供."""
    return vision.analyze_image(
        files.image, files.intrinsics, fastumi_root=files.root, marker_size_mm=16.0, **kwargs
    )


def test_candidate_projection_matches_independent_geometry(synthetic_files: SimpleNamespace) -> None:
    """由合成几何独立算出投影与人工像素的差，标记仍为未物理验证."""
    report, overlay = _analyze(
        synthetic_files, candidate_path=synthetic_files.candidate, observed_tcp_px=(57.0, 38.0)
    )
    assert report["accepted"]
    assert report["physical_tcp_verified"] is False
    assert report["gripper_openness_match"] == "unknown"
    assert report["candidate_gripper_openness"] == 1.0
    np.testing.assert_allclose(report["projected_tcp_px"], [60.0, 34.0])
    np.testing.assert_allclose(report["T_cam_tcp"][0:3], [[1, 0, 0, 0], [0, -1, 0, -0.018], [0, 0, -1, 0.3]])
    comparison = report["tcp_pixel_comparison"]
    np.testing.assert_allclose(comparison["projected_minus_observed_px"], [3, -4])
    assert comparison["norm_px"] == pytest.approx(5)
    assert report["tag_distance_comparison"]["estimate_minus_measured_mm"] == pytest.approx(0.5)
    assert np.any(np.all(overlay == [255, 255, 0], axis=-1))
    assert json.loads(json.dumps(report, allow_nan=False))["accepted"]


def test_tag_only_analysis_does_not_invent_tcp_projection(synthetic_files: SimpleNamespace) -> None:
    """缺少候选时允许双 tag 复核，测距依然明确来源."""
    report, overlay = _analyze(synthetic_files, measured_tag_distance_mm=133.0)
    assert report["accepted"] and not report["physical_tcp_verified"]
    assert report["tag_distance_comparison"]["source"] == "operator_measurement"
    assert report["tag_distance_comparison"]["estimate_minus_measured_mm"] == pytest.approx(0)
    assert "projected_tcp_px" not in report
    assert overlay.shape == (80, 120, 3)


@pytest.mark.parametrize("pixel", [(120, 40), (60, -1), (float("nan"), 40)])
def test_pixel_annotations_require_finite_raw_coordinates(
    synthetic_files: SimpleNamespace, pixel: tuple[float, float]
) -> None:
    """防止拿裁剪图或非法坐标作为独立像素证据."""
    with pytest.raises(vision.CalibrationImageError, match="原始图像范围") as failure:
        _analyze(synthetic_files, candidate_path=synthetic_files.candidate, observed_tcp_px=pixel)
    assert failure.value.report["accepted"] is False
    assert "tcp_pixel_comparison" not in failure.value.report
    json.dumps(failure.value.report, allow_nan=False)


def test_pixel_annotation_without_candidate_is_rejected(synthetic_files: SimpleNamespace) -> None:
    """人工像素应与明确的候选外参配对."""
    with pytest.raises(vision.CalibrationImageError, match="候选外参"):
        _analyze(synthetic_files, observed_tcp_px=(50, 30))


def test_reject_different_resolution(synthetic_files: SimpleNamespace) -> None:
    """模型尺寸或其他缩放图不得代替原始标定分辨率."""
    assert cv2.imwrite(str(synthetic_files.image), np.zeros((40, 60, 3), dtype=np.uint8))
    with pytest.raises(vision.CalibrationImageError, match="鱼眼内参") as failure:
        _analyze(synthetic_files)
    assert failure.value.report["image_shape_bgr"] == [40, 60, 3]
    assert failure.value.report["camera_resolution_wh"] == [120, 80]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("marker_size_mm", 17, "标签边长"),
        ("verified", True, "未验证"),
        ("frame", "demonstration_pair", "部署端 d_tag"),
        ("T_d_tag_tcp", np.diag([1, 1, -1, 1]).tolist(), "无效的旋转"),
    ],
)
def test_candidate_rejects_incompatible_metadata(
    synthetic_files: SimpleNamespace, key: str, value: Any, message: str
) -> None:
    """拒绝示教轴、镜像刚体和与实测边长冲突的候选."""
    payload = {**synthetic_files.payload, key: value}
    synthetic_files.candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(vision.CalibrationImageError, match=message):
        _analyze(synthetic_files, candidate_path=synthetic_files.candidate)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("reprojection_rms_px", 1.6, "重投影"),
        ("normal_angle_deg", 11.0, "法向"),
        ("coplanarity_deg", 10.1, "共面"),
        ("tag_distance_m", float("inf"), "质量指标无效"),
    ],
)
def test_quality_failure_keeps_json_evidence(
    synthetic_files: SimpleNamespace, metric: str, value: float, message: str
) -> None:
    """不接受低质量或非有限测量，拒绝报告仍可写入 JSON."""
    setattr(synthetic_files.observation, metric, value)
    with pytest.raises(vision.CalibrationImageError, match=message) as failure:
        _analyze(synthetic_files)
    assert failure.value.report["physical_tcp_verified"] is False
    assert failure.value.report["rejection_reason"]
    json.dumps(failure.value.report, allow_nan=False)


def test_rejects_nonpositive_tag_corner_depth(synthetic_files: SimpleNamespace) -> None:
    """即使重投影较低，也要求标签四角在相机前方."""
    synthetic_files.observation.tag1.camera_from_tag[2, 3] = -0.01
    with pytest.raises(vision.CalibrationImageError, match="非正深度"):
        _analyze(synthetic_files)


def test_external_loader_executes_source_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """只装载需要的三个源码并还原 sys.modules，不留下外部字节缓存."""
    src = tmp_path / "ros2_ws/src"
    board_dir = src / "hand_eye_calibration_macpdata/hand_eye_calibration_macpdata"
    transforms_dir = src / "fastumi_bringup/fastumi_bringup"
    board_dir.mkdir(parents=True)
    transforms_dir.mkdir(parents=True)
    (transforms_dir / "transforms.py").write_text("SENTINEL = 42\n", encoding="utf-8")
    for name in ("board.py", "dual_tag.py"):
        (board_dir / name).write_text("from fastumi_bringup.transforms import SENTINEL\n", encoding="utf-8")
    existing = ModuleType("fastumi_bringup.transforms")
    monkeypatch.setitem(sys.modules, existing.__name__, existing)
    report: dict[str, Any] = {}
    board, dual = vision._load_fastumi(tmp_path, report)
    assert board.SENTINEL == dual.SENTINEL == 42
    assert sys.modules[existing.__name__] is existing
    assert len(report["fastumi_sources"]) == 3
    assert all(len(entry["sha256"]) == 64 for entry in report["fastumi_sources"])
    assert not list(src.rglob("__pycache__"))
