# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""验证右手 TCP 转换、原样视频复制以及跨分片回合的统计量。"""

import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_stats
from lerobot.processor.relative_pose_processor import (
    to_absolute_pose,
    to_relative_pose,
)
from scripts.convert_handumi_to_lerobot import (
    GRIPPER_KEY,
    INDEX_KEYS,
    SOURCE_IMAGE_KEY,
    TCP_KEY,
    VALID_KEYS,
    convert_dataset,
    file_hash,
    main,
    read_json,
    write_json,
)

pytestmark = pytest.mark.dataset


@pytest.fixture
def source_dataset(tmp_path: Path) -> Path:
    """构造两个回合、三个数据分片，共享一个无需解码的原始视频文件。"""
    root = tmp_path / "raw"
    half = math.sqrt(0.5)
    tcp = [
        [1, 2, 3, 0, 0, half, half],
        [1, 2.01, 3, 0, 0, -half, -half],
        [1, 2.02, 3, 0, 0, half, half],
        [-1, 0, 2, 1, 0, 0, 1e-7],
        [-0.99, 0, 2, -1, 0, 0, -1e-7],
    ]
    grippers = [1, 0.9, 0.8, 0.7, 0.6]
    validity_names = [
        "left_device_tracked",
        "left_pose_valid",
        "right_device_tracked",
        "right_pose_valid",
        "hmd_tracked",
        "clock_synced",
        "connected",
        "streaming",
    ]
    features = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": None},
        "action": {"dtype": "float32", "shape": [16], "names": None},
        TCP_KEY: {"dtype": "float32", "shape": [7], "names": None},
        GRIPPER_KEY: {"dtype": "float32", "shape": [1], "names": None},
        "observation.valid": {"dtype": "int64", "shape": [8], "names": validity_names},
        **{key: {"dtype": "int64", "shape": [1], "names": None} for key in VALID_KEYS},
        **{key: {"dtype": "int64", "shape": [1], "names": None} for key in INDEX_KEYS},
        SOURCE_IMAGE_KEY: {
            "dtype": "video",
            "shape": [1080, 1920, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.height": 1080, "video.width": 1920, "video.fps": 30, "video.codec": "h264"},
        },
    }
    features["timestamp"]["dtype"] = "float32"
    data = {
        "observation.state": [[0.0] * 16 for _ in range(5)],
        "action": [[0.0] * 16 for _ in range(5)],
        TCP_KEY: tcp,
        GRIPPER_KEY: grippers,
        "observation.valid": [[0, 0, 1, 1, 0, 1, 1, 1]] * 5,
        **{key: [1] * 5 for key in VALID_KEYS},
        "timestamp": [0, 1 / 30, 2 / 30, 0, 1 / 30],
        "frame_index": [0, 1, 2, 0, 1],
        "episode_index": [0, 0, 0, 1, 1],
        "index": list(range(5)),
        "task_index": [0] * 5,
    }
    table = pa.table(data)
    for file, (start, stop) in enumerate([(0, 1), (1, 4), (4, 5)]):
        path = root / f"data/chunk-000/file-{file:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(start, stop - start), path)
    records, stats = [], []
    image_stats = {
        "min": np.zeros((3, 1, 1)),
        "max": np.ones((3, 1, 1)),
        "mean": np.full((3, 1, 1), 0.5),
        "std": np.full((3, 1, 1), 0.2),
    }
    for e, (start, stop) in enumerate([(0, 3), (3, 5)]):
        ep_stats = compute_episode_stats(
            {key: np.asarray(data[key][start:stop]) for key in ("observation.state", "action", *INDEX_KEYS)},
            features,
        )
        ep_stats[SOURCE_IMAGE_KEY] = {**image_stats, "count": np.array([stop - start])}
        stats.append(ep_stats)
        record = {
            "episode_index": e,
            "length": stop - start,
            "tasks": ["Pick and place"],
            "dataset_from_index": start,
            "dataset_to_index": stop,
            "data/chunk_index": 0,
            "data/file_index": 0 if e == 0 else 1,
            f"videos/{SOURCE_IMAGE_KEY}/chunk_index": 0,
            f"videos/{SOURCE_IMAGE_KEY}/file_index": 0,
            f"videos/{SOURCE_IMAGE_KEY}/from_timestamp": start / 30,
            f"videos/{SOURCE_IMAGE_KEY}/to_timestamp": stop / 30,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": e,
            "stats/observation.tracking.right_tracked/min": [1],
        }
        for key, values in ep_stats.items():
            record.update({f"stats/{key}/{name}": value.tolist() for name, value in values.items()})
        records.append(record)
    for e, record in enumerate(records):
        path = root / f"meta/episodes/chunk-000/file-{e:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([record]), path)
    write_stats(aggregate_stats(stats), root)
    pq.write_table(pa.table({"task_index": [0], "task": ["Pick and place"]}), root / "meta/tasks.parquet")
    write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": "handumi_raw",
            "fps": 30,
            "features": features,
            "total_frames": 5,
            "total_episodes": 2,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "splits": {"train": "0:2"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "handumi": {
                "tcp_pose": {
                    "feature": TCP_KEY,
                    "frame": "table",
                    "sample_time_feature": "observation.tracking.aligned_time_ns",
                }
            },
        },
    )
    video = root / f"videos/{SOURCE_IMAGE_KEY}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original encoded video sentinel")
    return root


@pytest.mark.parametrize("reference", ["episode_start_tcp", "current_tcp"])
def test_conversion_preserves_shards_videos_and_recomputes_stats(
    source_dataset: Path, reference: str
) -> None:
    """非单位首帧在跨分片处仍使用同一个锚点，图像及其统计量保持原样。"""
    output = source_dataset.parent / reference
    image_key = "observation.images.front"
    result = convert_dataset(source_dataset, output, pose_reference=reference)
    assert result["total_frames"] == 5 and result["total_episodes"] == 2 and result["video_files"] == 1
    assert result["repo_id"] == f"handumi/{reference}"
    table = pq.read_table(sorted((output / "data").rglob("*.parquet")))
    state = np.asarray(table["observation.state"].to_pylist())
    np.testing.assert_array_equal(state, table["action"].to_pylist())
    if reference == "episode_start_tcp":
        np.testing.assert_allclose(state[[0, 3], :6], 0, atol=1e-7)
        np.testing.assert_allclose(state[:3, :3], [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], atol=1e-7)
        np.testing.assert_allclose(state[4, :6], [0.01, 0, 0, 0, 0, 0], atol=1e-7)
    else:
        np.testing.assert_allclose(state[0, :3], [1, 2, 3])
        np.testing.assert_allclose(state[0, 3:6], [0, 0, math.pi / 2], atol=1e-7)
        np.testing.assert_allclose(state[3, 3:6], [math.pi - 2e-7, 0, 0], atol=2e-7)
        # 正负四元数应还原出相同旋转。
        np.testing.assert_allclose(state[3, 3:6], state[4, 3:6], atol=1e-7)
    np.testing.assert_allclose(state[:, 6], [1, 0.9, 0.8, 0.7, 0.6], atol=1e-7)
    src_video = next((source_dataset / "videos").rglob("*.mp4"))
    dst_video = next((output / "videos").rglob("*.mp4"))
    assert file_hash(src_video) == file_hash(dst_video)
    assert image_key in str(dst_video)
    assert [pq.read_metadata(p).num_rows for p in sorted((output / "data").rglob("*.parquet"))] == [1, 3, 1]
    info = read_json(output / "meta/info.json")
    assert info["features"][image_key]["shape"] == [1080, 1920, 3]
    assert set(info["features"]) == {"action", "observation.state", image_key, *INDEX_KEYS}
    meta = read_json(output / "meta/fastumi.json")
    assert meta["pose_frame"] == ("table" if reference == "current_tcp" else reference)
    assert meta["training_pose_frame"] == reference
    assert meta["source_handumi"] == read_json(source_dataset / "meta/info.json")["handumi"]
    stats = read_json(output / "meta/stats.json")
    np.testing.assert_allclose(stats["action"]["mean"], state.mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(stats["action"]["std"], state.std(axis=0), atol=1e-7)
    assert stats["action"]["count"] == [5]
    assert stats[image_key] == read_json(source_dataset / "meta/stats.json")[SOURCE_IMAGE_KEY]
    records = pq.read_table(sorted((output / "meta/episodes").rglob("*.parquet"))).to_pylist()
    assert not any("tracking" in key for key in records[0])
    assert len(records[0]["stats/action/mean"]) == 7
    assert records[0]["stats/action/count"] == [3]
    assert records[1][f"videos/{image_key}/from_timestamp"] == 3 / 30
    loaded = LeRobotDatasetMetadata(result["repo_id"], root=output)
    assert loaded.total_frames == 5 and loaded.total_episodes == 2


def test_current_reference_roundtrip_matches_episode_reference(source_dataset: Path) -> None:
    """两种文件参考系经过当前帧转换后得到相同十维窗口，动作可逆恢复。"""
    states = []
    for reference in ("episode_start_tcp", "current_tcp"):
        output = source_dataset.parent / reference
        convert_dataset(source_dataset, output, pose_reference=reference)
        table = pq.read_table(sorted((output / "data").rglob("*.parquet")))
        states.append(torch.tensor(table["observation.state"].to_pylist(), dtype=torch.float64))
    relative = [to_relative_pose(s[:3].unsqueeze(0), s[1].unsqueeze(0), [0], [0]) for s in states]
    torch.testing.assert_close(relative[0], relative[1], atol=2e-7, rtol=0)
    recovered = to_absolute_pose(relative[1], states[1][1].unsqueeze(0), [0], [0])
    torch.testing.assert_close(recovered[0], states[1][:3])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("action", [1.0] * 16, "action differs"),
        (TCP_KEY, [0, 0, 0, 0, 0, 0, 0], "Zero or invalid TCP quaternion"),
        (TCP_KEY, [float("nan"), 0, 0, 0, 0, 0, 1], "Non-finite"),
        (GRIPPER_KEY, 1.1, "Gripper outside"),
        ("observation.gripper.right.valid", 0, "Invalid"),
        ("observation.valid", [0] * 8, "Invalid"),
    ],
)
def test_invalid_inputs_report_location_and_leave_no_output(
    source_dataset: Path, column: str, value: object, message: str
) -> None:
    """无效标签及姿态不能生成半成品数据集，错误应给出回合与帧号。"""
    path = source_dataset / "data/chunk-000/file-001.parquet"
    data = pq.read_table(path).to_pydict()
    data[column][0] = value
    pq.write_table(pa.table(data), path)
    output = source_dataset.parent / "converted"
    with pytest.raises(ValueError, match=rf"{message}.*episode=0, frame=1"):
        convert_dataset(source_dataset, output)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}-*"))


def test_missing_video_cleans_staging_and_existing_output_is_preserved(source_dataset: Path) -> None:
    """复制中途失败仅清理本次临时目录，已有目标数据集始终拒绝覆盖。"""
    next((source_dataset / "videos").rglob("*.mp4")).unlink()
    output = source_dataset.parent / "converted"
    with pytest.raises(FileNotFoundError):
        convert_dataset(source_dataset, output)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}-*"))
    output.mkdir()
    marker = output / "existing"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        convert_dataset(source_dataset, output)
    assert marker.read_text() == "keep"


def test_invalid_paths_and_image_name_are_rejected(source_dataset: Path) -> None:
    """阻止数据集嵌套和会造成视频目录穿越的图像字段。"""
    with pytest.raises(ValueError, match="overlap"):
        convert_dataset(source_dataset, source_dataset / "nested")
    with pytest.raises(ValueError, match="image key"):
        convert_dataset(
            source_dataset, source_dataset.parent / "converted", image_key="observation.images../bad"
        )


def test_cli_prints_matching_training_configuration(
    source_dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI 自定义图像字段及标识后，动态参考系的训练提示与输出元数据一致。"""
    output = source_dataset.parent / "cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_handumi_to_lerobot.py",
            "--source",
            str(source_dataset),
            "--output",
            str(output),
            "--repo-id",
            "local/custom",
            "--pose-reference",
            "current_tcp",
            "--image-key",
            "observation.images.wrist",
        ],
    )
    main()
    printed = capsys.readouterr().out
    assert 'repo_id: "local/custom"' in printed
    assert "use_relative_pose: true" in printed
    assert "observation.images.wrist" in read_json(output / "meta/info.json")["features"]
    assert read_json(output / "meta/fastumi.json")["repo_id"] == "local/custom"
