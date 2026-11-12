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
"""将原始 HandUMI v3 数据转换为右手 TCP 七维示教数据，原样复制视频。

用法示例：
    uv run python scripts/convert_handumi_to_lerobot.py \
        --output data/handumi_data/umi-demo-tcp-native \
        --pose-reference episode_start_tcp

current_tcp 模式保存 table 坐标下的七维位姿。训练时须启用打印出的
policy.use_relative_pose 配置，由现有处理器按最新观测转换整个窗口。
"""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_stats
from lerobot.processor.relative_pose_processor import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_axis_angle,
)

PoseReference = Literal["episode_start_tcp", "current_tcp"]
SOURCE_IMAGE_KEY = "observation.images.right_wrist"
TCP_KEY = "observation.tracking.right_tcp_pose"
GRIPPER_KEY = "observation.gripper.right.normalized"
INDEX_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
VALID_KEYS = (
    "observation.tracking.right_tracked",
    "observation.gripper.right.valid",
    "observation.camera.right_wrist.healthy",
)
POSE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def read_json(path: Path) -> dict[str, Any]:
    """读取本地元数据；文件缺失直接报错，避免触发 Hub 下载。"""
    return json.loads(path.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    """以可读 JSON 写入转换元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def file_hash(path: Path) -> str:
    """分块计算视频摘要，用于验证复制过程没有改变编码内容。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_rows(table: pa.Table, invalid: NDArray[np.bool_], message: str) -> None:
    """将逐行检查的首个错误定位到源数据的回合和帧号。"""
    rows = np.flatnonzero(invalid)
    if rows.size:
        row = int(rows[0])
        episode = table["episode_index"][row].as_py()
        frame = table["frame_index"][row].as_py()
        raise ValueError(f"{message} (episode={episode}, frame={frame})")


def numeric_column(table: pa.Table, key: str, width: int) -> NDArray[np.float64]:
    """兼容源文件中标量与长度为一的列表，并核验数值列的宽度。"""
    values = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if values.shape not in ((len(table), width), (len(table),)) or (values.ndim == 1 and width != 1):
        raise ValueError(f"Invalid shape for {key}: {values.shape}, expected ({len(table)}, {width})")
    values = values.reshape(len(table), width)
    check_rows(table, ~np.isfinite(values).all(axis=1), f"Non-finite {key}")
    return values


def validate_frames(table: pa.Table, info: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    """校验示教标签、右手有效性和连续索引，防止丢帧或错用控制器动作。"""
    if len(table) != info["total_frames"] or len(episodes) != info["total_episodes"]:
        raise ValueError("Frame/episode counts disagree with meta/info.json")
    state = numeric_column(table, "observation.state", 16)
    action = numeric_column(table, "action", 16)
    check_rows(table, (state != action).any(axis=1), "Source action differs from state")
    for key in VALID_KEYS:
        check_rows(table, numeric_column(table, key, 1)[:, 0] != 1, f"Invalid {key}")
    validity = numeric_column(table, "observation.valid", len(info["features"]["observation.valid"]["names"]))
    names = info["features"]["observation.valid"]["names"]
    for name in ("right_device_tracked", "right_pose_valid", "clock_synced", "connected", "streaming"):
        if name not in names:
            raise ValueError(f"Missing validity flag {name}")
        check_rows(table, validity[:, names.index(name)] != 1, f"Invalid {name}")
    for key in INDEX_KEYS:
        numeric_column(table, key, 1)
    if table["index"].to_pylist() != list(range(len(table))):
        raise ValueError("Source index must be consecutive in shard order")
    offset = 0
    for expected_id, episode in enumerate(episodes):
        length = episode["length"]
        if (
            episode["episode_index"] != expected_id
            or length <= 0
            or episode["dataset_from_index"] != offset
            or episode["dataset_to_index"] != offset + length
        ):
            raise ValueError(f"Inconsistent episode metadata for episode={expected_id}")
        rows = table.slice(offset, length)
        if rows["episode_index"].to_pylist() != [expected_id] * length or rows[
            "frame_index"
        ].to_pylist() != list(range(length)):
            raise ValueError(f"Inconsistent frame order for episode={expected_id}")
        offset += length
    if offset != len(table):
        raise ValueError("Episode ranges do not cover all frames")


def convert_states(
    table: pa.Table, episodes: list[dict[str, Any]], pose_reference: PoseReference
) -> NDArray[np.float32]:
    """使用已标定 TCP 构造七维数据；回合首帧锚点跨 Parquet 分片共享。"""
    pose = numeric_column(table, TCP_KEY, 7)
    gripper = numeric_column(table, GRIPPER_KEY, 1)
    check_rows(table, ((gripper < 0) | (gripper > 1)).any(axis=1), "Gripper outside [0, 1]")
    norms = np.linalg.norm(pose[:, 3:], axis=1)
    check_rows(table, ~np.isfinite(norms) | (norms < 1e-12), "Zero or invalid TCP quaternion")
    # 源四元数为 xyzw；共享旋转工具使用 wxyz，并选择最短轴角表示。
    quaternion = torch.from_numpy(pose[:, [6, 3, 4, 5]] / norms[:, None])
    quaternion = torch.where(quaternion[:, :1] < 0, -quaternion, quaternion)
    rotations = axis_angle_to_matrix(quaternion_to_axis_angle(quaternion))
    positions = torch.from_numpy(pose[:, :3].copy())
    if pose_reference == "episode_start_tcp":
        for episode in episodes:
            start, end = episode["dataset_from_index"], episode["dataset_to_index"]
            reference_rotation = rotations[start].clone().T
            reference_position = positions[start].clone()
            positions[start:end] = (positions[start:end] - reference_position) @ reference_rotation.T
            rotations[start:end] = reference_rotation @ rotations[start:end]
            positions[start] = 0
            rotations[start] = torch.eye(3, dtype=rotations.dtype)
    return np.concatenate(
        (positions.numpy(), matrix_to_axis_angle(rotations).numpy(), gripper), axis=1
    ).astype(np.float32)


def write_converted_data(
    source: Path,
    output: Path,
    info: dict[str, Any],
    table: pa.Table,
    data_files: list[Path],
    episode_files: list[Path],
    episodes: list[dict[str, Any]],
    states: NDArray[np.float32],
    image_key: str,
) -> None:
    """保持源分片和视频时间引用，更新七维数据及逐回合和全局统计量。"""
    converted = pa.table(
        {
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 7)),
            "action": pa.array(states.tolist(), type=pa.list_(pa.float32(), 7)),
            **{key: table[key] for key in INDEX_KEYS},
        }
    )
    offset = 0
    for path in data_files:
        count = pq.read_metadata(path).num_rows
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(converted.slice(offset, count), target)
        offset += count

    all_stats = []
    records = []
    numeric_features = {key: info["features"][key] for key in ("observation.state", "action")}
    retained = set(info["features"])
    for episode in episodes:
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        stats = compute_episode_stats(
            {"observation.state": states[start:end], "action": states[start:end]}, numeric_features
        )
        all_stats.append(stats)
        record = {}
        for key, value in episode.items():
            renamed = key.replace(SOURCE_IMAGE_KEY, image_key)
            # 清除被删除的跟踪特征统计量；标准索引与原始图像统计量可继续沿用。
            if key.startswith("stats/") and renamed.rsplit("/", 1)[0][6:] not in retained:
                continue
            record[renamed] = value
        for key, values in stats.items():
            for stat, value in values.items():
                record[f"stats/{key}/{stat}"] = value.tolist()
        records.append(record)
    episode_table = pa.Table.from_pylist(records)
    offset = 0
    for path in episode_files:
        count = pq.read_metadata(path).num_rows
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(episode_table.slice(offset, count), target)
        offset += count
    stats = {
        key.replace(SOURCE_IMAGE_KEY, image_key): value
        for key, value in read_json(source / "meta/stats.json").items()
        if key.replace(SOURCE_IMAGE_KEY, image_key) in retained
    }
    stats.update(aggregate_stats(all_stats))
    write_stats(stats, output)


def copy_videos(
    source: Path, output: Path, info: dict[str, Any], episodes: list[dict[str, Any]], image_key: str
) -> int:
    """按 episode 引用复制所有视频分片，并逐文件核验 SHA-256。"""
    prefix = f"videos/{SOURCE_IMAGE_KEY}/"
    paths = {
        info["video_path"].format(
            video_key=SOURCE_IMAGE_KEY,
            chunk_index=episode[prefix + "chunk_index"],
            file_index=episode[prefix + "file_index"],
        )
        for episode in episodes
    }
    for relative in sorted(paths):
        path = source / relative
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Video path escapes source: {relative}")
        target = output / relative.replace(SOURCE_IMAGE_KEY, image_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_hash(path) != file_hash(target):
            raise OSError(f"Video checksum mismatch: {relative}")
    return len(paths)


def convert_dataset(
    source: Path,
    output: Path,
    repo_id: str | None = None,
    pose_reference: PoseReference = "episode_start_tcp",
    image_key: str = "observation.images.front",
) -> dict[str, Any]:
    """原子生成新数据集；目标已存在或输入不满足原始 HandUMI 契约时退出。"""
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output directories must not overlap")
    if pose_reference not in ("episode_start_tcp", "current_tcp"):
        raise ValueError(f"Unsupported pose reference: {pose_reference}")
    if re.fullmatch(r"observation\.images\.[A-Za-z0-9_.-]+", image_key) is None:
        raise ValueError(f"Invalid image key: {image_key}")
    repo_id = repo_id or f"handumi/{output.name}"
    original_info = read_json(source / "meta/info.json")
    info = deepcopy(original_info)
    handumi = info.pop("handumi")
    if info["codebase_version"] != "v3.0" or handumi["tcp_pose"] != {
        "feature": TCP_KEY,
        "frame": "table",
        "sample_time_feature": "observation.tracking.aligned_time_ns",
    }:
        raise ValueError("Expected HandUMI v3 with a calibrated table-frame right TCP")
    features = info["features"]
    if features[SOURCE_IMAGE_KEY]["dtype"] != "video":
        raise ValueError("Expected right-wrist video feature")
    info["features"] = {
        "observation.state": {"dtype": "float32", "shape": [7], "names": POSE_NAMES},
        "action": {"dtype": "float32", "shape": [7], "names": POSE_NAMES},
        image_key: features[SOURCE_IMAGE_KEY],
        **{key: features[key] for key in INDEX_KEYS},
    }
    info["robot_type"] = "umi"
    data_files = sorted((source / "data").glob("chunk-*/*.parquet"))
    episode_files = sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))
    if not data_files or not episode_files:
        raise FileNotFoundError("Missing data or episode parquet shards")
    required = [
        *INDEX_KEYS,
        "observation.state",
        "action",
        TCP_KEY,
        GRIPPER_KEY,
        *VALID_KEYS,
        "observation.valid",
    ]
    table = pq.read_table(data_files, columns=required)
    episodes = pq.read_table(episode_files).to_pylist()
    validate_frames(table, original_info, episodes)
    states = convert_states(table, episodes, pose_reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        write_converted_data(
            source, temporary, info, table, data_files, episode_files, episodes, states, image_key
        )
        video_count = copy_videos(source, temporary, original_info, episodes, image_key)
        shutil.copyfile(source / "meta/tasks.parquet", temporary / "meta/tasks.parquet")
        write_json(temporary / "meta/info.json", info)
        write_json(
            temporary / "meta/fastumi.json",
            {
                "pose_frame": "episode_start_tcp" if pose_reference == "episode_start_tcp" else "table",
                "training_pose_frame": pose_reference,
                "end_frame": "tcp",
                "tcp_definition": "gripper_grasp_center",
                "state_layout": "xyz_rotvec_gripper",
                "action_layout": "xyz_rotvec_gripper",
                "action_source": "demonstration_trajectory",
                "state_time_reference": "image_timestamp",
                "action_time_reference": "image_timestamp",
                "position_unit": "m",
                "rotation_unit": "rad",
                "gripper_mode": "normalized",
                "gripper_representation": "normalized_0_1",
                "fps": info["fps"],
                "image_size": features[SOURCE_IMAGE_KEY]["shape"][:2],
                "image_preprocessing": "none",
                "source_dataset": str(source),
                "repo_id": repo_id,
                "source_handumi": handumi,
                "episodes": [
                    {"episode_index": ep["episode_index"], "source_episode_index": ep["episode_index"]}
                    for ep in episodes
                ],
            },
        )
        # 发布前通过框架读取元数据，并确认每个回合的数据和视频引用均可解析。
        metadata = LeRobotDatasetMetadata(repo_id, root=temporary)
        for episode in episodes:
            episode_id = episode["episode_index"]
            for relative in (
                metadata.get_data_file_path(episode_id),
                metadata.get_video_file_path(episode_id, image_key),
            ):
                if not (temporary / relative).is_file():
                    raise FileNotFoundError(f"Missing output reference for episode={episode_id}: {relative}")
        # 目标存在时拒绝覆盖；临时目录在最终同级目录中，发布使用同文件系统重命名。
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "output": str(output),
        "repo_id": repo_id,
        "pose_reference": pose_reference,
        "total_episodes": len(episodes),
        "total_frames": len(table),
        "video_files": video_count,
        "image_key": image_key,
        "image_shape": info["features"][image_key]["shape"],
    }


def main() -> None:
    """解析转换参数并打印现有训练 YAML 需要配套修改的字段。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source", type=Path, default=Path("data/handumi_data/umi-demo"), help="原始 HandUMI v3 数据集目录"
    )
    parser.add_argument("--output", type=Path, required=True, help="新数据集目录，须尚未存在")
    parser.add_argument("--repo-id", help="训练使用的数据集标识，省略时为 handumi/<输出目录名>")
    parser.add_argument(
        "--pose-reference",
        choices=("episode_start_tcp", "current_tcp"),
        default="episode_start_tcp",
        help="训练参考系；current_tcp 须配合 policy.use_relative_pose=true",
    )
    parser.add_argument("--image-key", default="observation.images.front", help="输出图像特征名称")
    args = parser.parse_args()
    result = convert_dataset(args.source, args.output, args.repo_id, args.pose_reference, args.image_key)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n训练 YAML 配置片段")
    print(f"dataset:\n  root: {json.dumps(result['output'])}\n  repo_id: {json.dumps(result['repo_id'])}")
    print(f"policy:\n  use_relative_pose: {str(args.pose_reference == 'current_tcp').lower()}")


if __name__ == "__main__":
    main()
