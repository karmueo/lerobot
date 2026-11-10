"""从重复手测生成双 Tag 到抓取中心的平移候选."""

from __future__ import annotations

import math
import statistics
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

_AXES = ("x_mm", "y_mm", "z_mm")
_AXIS_DEFINITIONS = {
    "origin": "双 Tag 中心连线的中点",
    "+X": "沿中心连线从 ID 0 指向 ID 1",
    "+Y": "Z × X，安装约定为朝指尖",
    "+Z": "平均 Tag 正面法向在垂直于 X 的平面上的投影方向",
}


def measurement_template() -> dict[str, Any]:
    """返回待填的毫米制模板，空字段代表未测量."""
    return {
        "schema_version": 1,
        "units": "mm",
        "frame": "deployment_dual_tag",
        "tcp_axes": "aligned_with_dtag",
        "tag_ids": [0, 1],
        "marker_size_mm": None,
        "measured_tag_distance_mm": None,
        "gripper_openness": None,
        "measurement_notes": "",
        "samples": [dict.fromkeys(_AXES) for _ in range(3)],
    }


def _finite_number(value: object, field: str) -> float:
    """拒绝未测值、布尔值和非有限数值."""
    if type(value) not in (int, float):
        raise ValueError(f"{field} 须填写有限数值")
    assert isinstance(value, (int, float))
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} 须填写有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} 须填写有限数值")
    return number


def solve_measurements(path: Path) -> dict[str, Any]:
    """读取 YAML 或 JSON，统计手测平移并保留未验收状态."""
    content = path.read_bytes()
    try:
        data = yaml.safe_load(content)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取手测文件 {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("手测文件须为键值映射")
    contract = measurement_template()
    for key in ("schema_version", "units", "frame", "tcp_axes", "tag_ids"):
        value = data.get(key)
        expected = contract[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"{key} 必须为 {expected!r}")
    if any(type(tag_id) is not int for tag_id in data["tag_ids"]):
        raise ValueError("tag_ids 必须使用整数 ID [0, 1]")

    marker_size = _finite_number(data.get("marker_size_mm"), "marker_size_mm")
    if marker_size <= 0:
        raise ValueError("marker_size_mm 必须大于零")
    tag_distance = data.get("measured_tag_distance_mm")
    if tag_distance is not None:
        tag_distance = _finite_number(tag_distance, "measured_tag_distance_mm")
        if tag_distance <= 0:
            raise ValueError("measured_tag_distance_mm 必须大于零")
    openness = _finite_number(data.get("gripper_openness"), "gripper_openness")
    if not 0 <= openness <= 1:
        raise ValueError("gripper_openness 必须在 [0, 1] 范围内")
    notes = data.get("measurement_notes")
    if not isinstance(notes, str):
        raise ValueError("measurement_notes 必须为字符串")

    samples = data.get("samples")
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("samples 至少须有三组独立测量")
    values: dict[str, list[float]] = {axis: [] for axis in _AXES}
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{index}] 须为 xyz 测量映射")
        for axis in _AXES:
            values[axis].append(_finite_number(sample.get(axis), f"samples[{index}].{axis}"))

    # 样本标准差描述重复性，不能据此判定准确度或修正朝向。
    means = {axis: statistics.mean(axis_values) for axis, axis_values in values.items()}
    try:
        deviations = {axis: statistics.stdev(axis_values) for axis, axis_values in values.items()}
    except OverflowError as exc:
        raise ValueError("样本标准差超出有限数值范围") from exc
    ranges = {axis: max(axis_values) - min(axis_values) for axis, axis_values in values.items()}
    if not all(math.isfinite(value) for summary in (means, deviations, ranges) for value in summary.values()):
        raise ValueError("样本统计超出有限数值范围")
    x, y, z = (means[axis] / 1000.0 for axis in _AXES)
    return {
        **{key: contract[key] for key in ("schema_version", "frame", "tcp_axes", "tag_ids")},
        "axis_definitions": _AXIS_DEFINITIONS.copy(),
        "marker_size_mm": marker_size,
        "measured_tag_distance_mm": tag_distance,
        "gripper_openness": openness,
        "measurement_notes": notes,
        "sample_statistics_mm": {
            "count": len(samples),
            "mean": means,
            "sample_sd": deviations,
            "min": {axis: min(axis_values) for axis, axis_values in values.items()},
            "max": {axis: max(axis_values) for axis, axis_values in values.items()},
            "range": ranges,
        },
        # 齐次矩阵平移统一用米，旋转沿用训练约定的单位矩阵。
        "T_d_tag_tcp": [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]],
        "transform_translation_units": "m",
        "verified": False,
        "source": {"path": str(path.resolve()), "sha256": sha256(content).hexdigest()},
    }
