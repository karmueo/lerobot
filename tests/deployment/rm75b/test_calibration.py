"""验证手测模板、坐标契约及候选矩阵."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import yaml

from lerobot.deployment.rm75b.calibration import measurement_template, solve_measurements


@pytest.fixture
def measured() -> dict[str, Any]:
    """构造测试专用测值，包含有符号的三个坐标."""
    data = measurement_template()
    data.update(
        marker_size_mm=16.25,
        measured_tag_distance_mm=134.0,
        gripper_openness=0.7,
        measurement_notes="测试夹具",
        samples=[
            {"x_mm": -3.0, "y_mm": 119.0, "z_mm": -21.0},
            {"x_mm": -2.0, "y_mm": 120.0, "z_mm": -19.0},
            {"x_mm": -1.0, "y_mm": 121.0, "z_mm": -20.0},
        ],
    )
    return data


@pytest.mark.base
def test_template_contains_no_invented_measurements(tmp_path: Path) -> None:
    """空模板保留三个独立填写槽，提交时拒绝未测值."""
    data = measurement_template()
    assert data["marker_size_mm"] is None
    assert data["gripper_openness"] is None
    assert data["samples"] == [dict.fromkeys(("x_mm", "y_mm", "z_mm"))] * 3
    assert data["samples"][0] is not data["samples"][1]
    path = tmp_path / "blank.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="marker_size_mm"):
        solve_measurements(path)


@pytest.mark.base
def test_solve_records_signed_translation_statistics_and_provenance(
    tmp_path: Path, measured: dict[str, Any]
) -> None:
    """毫米量测只确定平移，矩阵和来源摘要应可复算."""
    path = tmp_path / "measured.yaml"
    path.write_text(yaml.safe_dump(measured))
    result = solve_measurements(path)
    assert result["T_d_tag_tcp"] == [
        [1.0, 0.0, 0.0, -0.002],
        [0.0, 1.0, 0.0, 0.120],
        [0.0, 0.0, 1.0, -0.020],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert result["transform_translation_units"] == "m"
    assert result["verified"] is False
    assert result["gripper_openness"] == 0.7
    assert result["measured_tag_distance_mm"] == 134.0
    assert result["sample_statistics_mm"] == {
        "count": 3,
        "mean": {"x_mm": -2.0, "y_mm": 120.0, "z_mm": -20.0},
        "sample_sd": {"x_mm": 1.0, "y_mm": 1.0, "z_mm": 1.0},
        "min": {"x_mm": -3.0, "y_mm": 119.0, "z_mm": -21.0},
        "max": {"x_mm": -1.0, "y_mm": 121.0, "z_mm": -19.0},
        "range": {"x_mm": 2.0, "y_mm": 2.0, "z_mm": 2.0},
    }
    assert result["source"] == {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


@pytest.mark.base
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("units", "m"),
        ("frame", "base"),
        ("tcp_axes", "measured_rotation"),
        ("tag_ids", [1, 0]),
        ("tag_ids", [False, True]),
        ("marker_size_mm", 0),
        ("marker_size_mm", "16"),
        ("measured_tag_distance_mm", -1),
        ("gripper_openness", None),
        ("gripper_openness", 1.01),
        ("gripper_openness", True),
        ("measurement_notes", 3),
        ("samples", []),
    ],
)
def test_rejects_invalid_contract(
    tmp_path: Path, measured: dict[str, Any], field: str, value: object
) -> None:
    """拒绝错单位、错轴、缺少实际测值或无效开度."""
    measured[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(measured))
    with pytest.raises(ValueError, match=field):
        solve_measurements(path)


@pytest.mark.base
@pytest.mark.parametrize("value", [None, True, ".nan", ".inf", "-.inf"])
def test_rejects_invalid_sample_coordinates(tmp_path: Path, measured: dict[str, Any], value: object) -> None:
    """空值、布尔值和 YAML 非有限坐标均不能参与统计."""
    path = tmp_path / "invalid.yaml"
    measured["samples"][1]["z_mm"] = (
        {".nan": float("nan"), ".inf": float("inf"), "-.inf": float("-inf")}[value]
        if isinstance(value, str)
        else value
    )
    path.write_text(yaml.safe_dump(measured))
    with pytest.raises(ValueError, match=r"samples\[2\]\.z_mm"):
        solve_measurements(path)
