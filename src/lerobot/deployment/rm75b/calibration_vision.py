"""在已保存的原始腕部图像上复算双标签姿态和候选 TCP 投影."""

from __future__ import annotations

import hashlib
import math
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from threading import Lock
from types import CodeType, ModuleType
from typing import Any

import cv2
import numpy as np
import yaml

from .geometry import validate_transform
from .types import FloatArray, ImageArray

_DICTIONARY = "DICT_4X4_50"
_TAG_IDS = (0, 1)
_MAX_REPROJECTION_PX = 1.5
_MAX_NORMAL_DEG = 10.0
_MAX_COPLANARITY_DEG = 10.0
_IMPORT_LOCK = Lock()


class CalibrationImageError(ValueError):
    """离线分析拒绝，附带可持久化的部分证据."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        """在报告中保留拒绝原因，供调用端写入 JSON."""
        super().__init__(message)
        self.report = {**report, "accepted": False, "rejection_reason": message}


class _ReadOnlySourceLoader(SourceFileLoader):
    """直接执行已计算摘要的源码字节，不读写外部 pycache."""

    def __init__(self, name: str, path: Path, source: bytes) -> None:
        """保留读取的原始源码，保证报告摘要与执行内容一致."""
        super().__init__(name, str(path))
        self._source = source

    def get_code(self, fullname: str) -> CodeType:
        """编译这次读取的源码，绕过默认缓存流程."""
        return compile(self._source, self.path, "exec", dont_inherit=True)


def _load_fastumi(root: Path, report: dict[str, Any]) -> tuple[ModuleType, ModuleType]:
    """仅加载三个已知的本地数学模块，不导入 ROS 包或运动节点."""
    src = root / "ros2_ws/src"
    tag_dir = src / "hand_eye_calibration_macpdata/hand_eye_calibration_macpdata"
    modules = {
        "fastumi_bringup.transforms": src / "fastumi_bringup/fastumi_bringup/transforms.py",
        "_lerobot_rm75b_fastumi_board": tag_dir / "board.py",
        "_lerobot_rm75b_fastumi_dual_tag": tag_dir / "dual_tag.py",
    }
    loaded: dict[str, ModuleType] = {}
    fingerprints: list[dict[str, str]] = []
    report["fastumi_sources"] = fingerprints
    # 外部两个模块从固定名称导入 transforms，临时注册后还原调用者的模块表。
    with _IMPORT_LOCK:
        previous = {name: sys.modules.get(name) for name in modules}
        try:
            for name, path in modules.items():
                source = path.read_bytes()
                fingerprints.append(
                    {"path": str(path.resolve()), "sha256": hashlib.sha256(source).hexdigest()}
                )
                loader = _ReadOnlySourceLoader(name, path, source)
                spec = spec_from_file_location(name, path, loader=loader)
                if spec is None:
                    raise ImportError(f"无法加载 FastUMI 数学模块: {path}")
                module = module_from_spec(spec)
                sys.modules[name] = module
                loader.exec_module(module)
                loaded[name] = module
        finally:
            for name, prior in previous.items():
                if prior is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior
    return loaded["_lerobot_rm75b_fastumi_board"], loaded["_lerobot_rm75b_fastumi_dual_tag"]


def _asset(path: Path) -> dict[str, str]:
    """生成已提供输入文件的来源摘要."""
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _positive_number(value: float, label: str) -> float:
    """标签边长和中心距都必须是有限的正毫米值."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} 必须是有限正数")
    return float(value)


def _read_candidate(path: Path, marker_size_mm: float) -> tuple[FloatArray, dict[str, Any]]:
    """只读解析候选外参，拒绝非刚体变换或不匹配的标签规格."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "T_d_tag_tcp" not in payload:
        raise ValueError("候选文件必须包含 T_d_tag_tcp")
    if payload.get("verified", False) is not False:
        raise ValueError("图像分析只接收未验证的 TCP 候选")
    if payload.get("frame", "deployment_dual_tag") != "deployment_dual_tag":
        raise ValueError("候选文件必须使用部署端 d_tag 坐标系")
    if "tag_ids" in payload and payload["tag_ids"] != list(_TAG_IDS):
        raise ValueError("候选文件 tag_ids 必须为 [0, 1]")
    if "marker_size_mm" in payload and not math.isclose(
        _positive_number(payload["marker_size_mm"], "候选 marker_size_mm"), marker_size_mm, abs_tol=1e-9
    ):
        raise ValueError("候选文件的标签边长与检测参数不一致")
    return validate_transform(payload["T_d_tag_tcp"]), payload


def _image_pixel(value: Any, shape: tuple[int, ...], label: str) -> FloatArray:
    """核对原始图像上的有限二维坐标，避免使用裁剪图上的像素."""
    pixel = np.asarray(value, dtype=np.float64)
    height, width = shape[:2]
    if (
        pixel.shape != (2,)
        or not np.isfinite(pixel).all()
        or not 0 <= pixel[0] < width
        or not 0 <= pixel[1] < height
    ):
        raise ValueError(f"{label} 必须在原始图像范围内，尺寸 {width}×{height}")
    return pixel


def _check_observation(observation: Any, dual: ModuleType, marker_size_m: float) -> None:
    """按历史重投影和法向阈值拒绝坏帧，并限制明显的非共面结果."""
    values = (
        observation.reprojection_rms_px,
        observation.normal_angle_deg,
        observation.coplanarity_deg,
        observation.tag_distance_m,
    )
    if not np.isfinite(values).all() or observation.tag_distance_m <= 0:
        raise ValueError("双标签质量指标无效")
    checks = (
        (observation.reprojection_rms_px, _MAX_REPROJECTION_PX, "双标签重投影 RMS"),
        (observation.normal_angle_deg, _MAX_NORMAL_DEG, "双标签法向夹角"),
        (observation.coplanarity_deg, _MAX_COPLANARITY_DEG, "双标签共面偏差"),
    )
    for value, limit, label in checks:
        if not 0 <= value <= limit:
            raise ValueError(f"{label} {value:.4f} 超过允许阈值 {limit}")
    points = dual.tag_object_points(marker_size_m)
    for tag in (observation.tag0, observation.tag1):
        if not np.isfinite(tag.corners_px).all() or not math.isfinite(tag.reprojection_rms_px):
            raise ValueError(f"标签 {tag.tag_id} 的角点或重投影误差无效")
        camera_from_tag = validate_transform(tag.camera_from_tag)
        depths = points @ camera_from_tag[2, :3] + camera_from_tag[2, 3]
        if not np.all(depths > 0):
            raise ValueError(f"标签 {tag.tag_id} 存在非正深度角点")


def analyze_image(
    image_path: Path,
    camera_path: Path,
    *,
    fastumi_root: Path,
    marker_size_mm: float,
    candidate_path: Path | None = None,
    measured_tag_distance_mm: float | None = None,
    observed_tcp_px: tuple[float, float] | None = None,
) -> tuple[dict[str, Any], ImageArray]:
    """分析原始鱼眼图，返回 JSON 证据和 BGR 叠加图."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "accepted_scope": "image_analysis_only",
        "image_path": str(image_path.resolve()),
        "camera_path": str(camera_path.resolve()),
        "candidate_path": str(candidate_path.resolve()) if candidate_path is not None else None,
        "fastumi_root": str(fastumi_root.resolve()),
        "dictionary": _DICTIONARY,
        "tag_ids": list(_TAG_IDS),
        "d_tag_axes": {"x": "id0_to_id1", "y": "z_cross_x_toward_gripper_tip", "z": "front_normal"},
        "thresholds": {
            "max_reprojection_px": _MAX_REPROJECTION_PX,
            "max_normal_angle_deg": _MAX_NORMAL_DEG,
            "max_coplanarity_deg": _MAX_COPLANARITY_DEG,
        },
        "physical_tcp_verified": False,
        # 相机帧不含电机反馈，画面不能证明测量时和拍照时的夹爪开度一致。
        "gripper_openness_match": "unknown",
    }
    try:
        marker_size_mm = _positive_number(marker_size_mm, "marker_size_mm")
        report["marker_size_mm"] = marker_size_mm
        report["marker_size_source"] = "operator_supplied_black_border_edge"
        if observed_tcp_px is not None and candidate_path is None:
            raise ValueError("人工 TCP 像素需与候选外参一起提供")
        report["image"] = _asset(image_path)
        report["camera_intrinsics"] = _asset(camera_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f"无法解码输入图像: {image_path}")
        report["image_shape_bgr"] = list(image.shape)
        board, dual = _load_fastumi(fastumi_root, report)
        camera = board.load_fisheye_camera(camera_path)
        report["camera_resolution_wh"] = [camera.width, camera.height]
        if image.shape[:2] != (camera.height, camera.width):
            raise ValueError(
                f"原始图像 {image.shape[1]}×{image.shape[0]} 与鱼眼内参 {camera.width}×{camera.height} 不一致"
            )
        if (
            not np.isfinite(camera.K).all()
            or not np.isfinite(camera.D).all()
            or min(camera.K[0, 0], camera.K[1, 1]) <= 0
        ):
            raise ValueError("鱼眼内参或畸变系数无效")

        estimator = dual.DualTagEstimator(camera, _DICTIONARY, marker_size_mm / 1000, _TAG_IDS)
        observation = estimator.estimate(image)
        report["quality"] = {
            "reprojection_rms_px": float(observation.reprojection_rms_px),
            "normal_angle_deg": float(observation.normal_angle_deg),
            "coplanarity_deg": float(observation.coplanarity_deg),
            "estimated_tag_distance_mm": float(observation.tag_distance_m) * 1000,
        }
        _check_observation(observation, dual, marker_size_mm / 1000)
        camera_from_dtag = validate_transform(observation.camera_from_dtag)
        report["T_cam_dtag"] = camera_from_dtag.tolist()
        report["tags"] = [
            {
                "id": int(tag.tag_id),
                "corners_px": tag.corners_px.tolist(),
                "reprojection_rms_px": float(tag.reprojection_rms_px),
                "T_cam_tag": tag.camera_from_tag.tolist(),
            }
            for tag in (observation.tag0, observation.tag1)
        ]

        dtag_from_tcp = None
        distance_source = "operator_measurement"
        if candidate_path is not None:
            report["candidate"] = _asset(candidate_path)
            dtag_from_tcp, candidate = _read_candidate(candidate_path, marker_size_mm)
            openness = candidate.get("gripper_openness")
            if openness is not None and (
                type(openness) not in (int, float) or not math.isfinite(openness) or not 0 <= openness <= 1
            ):
                raise ValueError("候选 gripper_openness 必须是 [0, 1] 内的有限数值")
            report["candidate_gripper_openness"] = openness
            camera_from_tcp = camera_from_dtag @ dtag_from_tcp
            projected = _image_pixel(
                camera.project(camera_from_tcp[:3, 3][None])[0], image.shape, "候选 TCP 投影"
            )
            report["T_cam_tcp"] = camera_from_tcp.tolist()
            report["projected_tcp_px"] = projected.tolist()
            if measured_tag_distance_mm is None and candidate.get("measured_tag_distance_mm") is not None:
                measured_tag_distance_mm = candidate["measured_tag_distance_mm"]
                distance_source = "candidate_measurement"

        if measured_tag_distance_mm is not None:
            measured = _positive_number(measured_tag_distance_mm, "measured_tag_distance_mm")
            report["tag_distance_comparison"] = {
                "measured_mm": measured,
                "source": distance_source,
                "estimate_minus_measured_mm": report["quality"]["estimated_tag_distance_mm"] - measured,
            }

        overlay = dual.draw_observation(image, camera, observation, dtag_from_tcp)
        if observed_tcp_px is not None:
            observed = _image_pixel(observed_tcp_px, image.shape, "人工标注 TCP")
            residual = projected - observed
            report["tcp_pixel_comparison"] = {
                "observed_px": observed.tolist(),
                "source": "operator_annotation_on_raw_image",
                "projected_minus_observed_px": residual.tolist(),
                "norm_px": float(np.linalg.norm(residual)),
            }
            # 洋红十字是模型投影，青色圆是独立提供的同图二维标注。
            cv2.circle(overlay, tuple(int(round(v)) for v in observed), 12, (255, 255, 0), 3)
        report["accepted"] = True
        return report, overlay
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        AttributeError,
        SyntaxError,
        cv2.error,
        yaml.YAMLError,
    ) as error:
        # 外部源的 DualTagError 是 ValueError，直接保留其缺失/重复标签等详细原因。
        # 失效的数值不会进入 CLI JSON，避免二次序列化错误掩盖原始拒绝。
        if "quality" in report:
            report["quality"] = {
                key: value if math.isfinite(value) else None for key, value in report["quality"].items()
            }
        raise CalibrationImageError(str(error), report) from error
