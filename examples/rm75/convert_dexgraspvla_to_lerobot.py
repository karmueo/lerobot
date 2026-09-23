#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Convert the DexGraspVLA gripper Zarr demos of the RM75 toy-robot task to a LeRobot v3 dataset.

The source tree (``.../DexGraspVLA/data/gripper_zarr`` plus its image cache) is treated as
strictly read-only: every zarr store is opened with ``mode="r"`` and nothing is ever written
there. All derived data lands under ``--output-root``.

Expected source layout (see examples/rm75/README.md for details)::

    gripper_zarr/                 # one read-only Zarr group
        data/state               # [N, 8]
        data/action              # [N, 8]
        data/gripper_image_paths # one relative path per control step
        meta/episode_ends
        meta/episode_folders     # source_000/Target4/episode_0, etc.

Image paths resolve into the sibling ``gripper_frames`` tree.

The converter writes ``observation.images.gripper`` (plain RGB video, no masks),
``observation.state`` and ``action`` (float32, kept at their original [-1, 1] semantics —
no renormalization), and a single fixed ``task`` instruction.

Run (zarr is a one-shot conversion dependency, not a LeRobot core dependency)::

    uv run --with "zarr>=2.18,<3" examples/rm75/convert_dexgraspvla_to_lerobot.py \\
        --source-zarr /home/scl/work/UMI/DexGraspVLA/data/gripper_zarr \\
        --output-root data/rm75_toy_robot_lerobot
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# ── Task constants ────────────────────────────────────────────────────────────────

TARGET_CATEGORIES = ("Target", "Target2", "Target3", "Target4", "Target5")
TASK_INSTRUCTION = "Pick up the toy robot and place it in the pink bowl."
FPS = 50
ROBOT_TYPE = "rm75"
STATE_AXES = [f"joint_{i}" for i in range(7)] + ["gripper"]
MANIFEST_FILENAME = "manifest.json"

# Full-count precheck for the real dataset. Pass 0 to disable a check (tests use small
# synthetic sources).
EXPECTED_EPISODES = 600
EXPECTED_FRAMES = 267_847
EXPECTED_STATE_DIM = 8

# Deterministic per-category 90/10 episode split: within each category's episode list
# in source metadata order, an episode is held out for eval iff its position modulo
# SPLIT_MODULUS equals SPLIT_REMAINDER. Counts 200/100/100/100/100 yield 540 train
# and 60 eval episodes; output is interleaved by category within each split.
SPLIT_MODULUS = 10
SPLIT_REMAINDER = 9

# ── Source-layout discovery ───────────────────────────────────────────────────────

STATE_KEY_CANDIDATES = (
    "data/state",
    "state",
    "data/robot0_qpos",
    "observations/state",
    "data/joint_state",
)
ACTION_KEY_CANDIDATES = (
    "data/action",
    "action",
    "actions",
    "data/robot0_action",
    "observations/action",
)
EPISODE_ENDS_CANDIDATES = ("meta/episode_ends", "episode_ends", "data/episode_ends")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
IMAGE_CACHE_DIR_CANDIDATES = ("image_cache", "img_cache", "cache", "frames")


class ConversionError(RuntimeError):
    """Raised when the source layout, the precheck or the resume contract fails.

    The error is raised before anything is written, so a failed conversion leaves the
    output directory untouched.
    """


@dataclass(frozen=True)
class EpisodePlan:
    """One source episode and its destination in the converted dataset."""

    output_index: int
    category: str
    source_key: str  # unique source id, e.g. "Target/ep_0000" or "Target#0003"
    num_frames: int
    split: str  # "train" | "eval"
    zarr_path: str  # source zarr group holding this episode (read-only)
    state_key: str  # array path of the state inside the zarr group
    action_key: str  # array path of the action inside the zarr group
    episode_slice: tuple[int, int] | None  # (start, stop) when the zarr holds several episodes
    image_dir: str  # directory of the episode's cached frames

    def as_manifest_entry(self) -> dict:
        return {
            "output_index": self.output_index,
            "category": self.category,
            "source_key": self.source_key,
            "num_frames": self.num_frames,
            "split": self.split,
        }


def natural_key(name: str) -> list:
    """Sort key so that ep_2 comes before ep_10."""
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", name)]


def is_zarr_group(path: Path) -> bool:
    return (
        path.name.endswith(".zarr")
        or (path / ".zgroup").exists()
        or (path / ".zattrs").exists()
        or (path / ".zmetadata").exists()
        or (path / "zarr.json").exists()
    )


def discover_image_cache(source_root: Path, explicit: Path | None) -> Path:
    """Locate the image cache directory (read-only lookup)."""
    if explicit is not None:
        if not explicit.is_dir():
            raise ConversionError(f"--image-cache does not exist or is not a directory: {explicit}")
        return explicit
    candidates = [source_root / name for name in IMAGE_CACHE_DIR_CANDIDATES]
    candidates += [source_root.parent / name for name in IMAGE_CACHE_DIR_CANDIDATES]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ConversionError(
        "Could not auto-discover the image cache. Pass --image-cache explicitly. "
        f"Looked at: {', '.join(str(c) for c in candidates)}"
    )


def episode_image_dir(image_cache: Path, plan: dict, template: str) -> Path:
    """Resolve the cached-frame directory of one episode from --image-dir-template."""
    rel = template.format(
        category=plan["category"],
        episode=plan["episode"],
        episode_index=plan["episode_index"],
        source_rel=plan["source_rel"],
    )
    return image_cache / rel


def list_cached_frames(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise ConversionError(f"Image directory for an episode is missing: {image_dir}")
    frames = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not frames:
        raise ConversionError(f"No image files ({', '.join(IMAGE_EXTENSIONS)}) in {image_dir}")
    return sorted(frames, key=lambda p: natural_key(p.name))


def resolve_stored_image(source_root: Path, stored_path: str) -> Path:
    path = Path(stored_path)
    path = (source_root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(source_root.parent.resolve()):
        raise ConversionError(f"Image path escapes the source data tree: {stored_path}")
    if not path.is_file():
        raise ConversionError(f"Image referenced by Zarr is missing: {stored_path} -> {path}")
    return path


def read_image_rgb(path: Path) -> np.ndarray:
    """Decode one cached frame as plain RGB / HWC / uint8 (no masks, no color swap)."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"))
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ConversionError(f"Unexpected image layout in {path}: dtype={arr.dtype} shape={arr.shape}")
    return arr


def resolve_array(group, candidates: tuple[str, ...], kind: str, zarr_path: Path) -> str:
    for key in candidates:
        try:
            arr = group[key]
        except Exception:  # noqa: BLE001 - zarr raises different types for missing keys
            continue
        if arr is not None and hasattr(arr, "shape") and arr.ndim >= 2:
            return key
    raise ConversionError(
        f"Could not find the {kind} array in {zarr_path}. Tried: {', '.join(candidates)}. "
        "Pass --state-key / --action-key to point at the right array."
    )


def open_zarr_readonly(path: Path):
    try:
        import zarr
    except ImportError as err:  # pragma: no cover - depends on env
        raise ConversionError(
            "zarr is required to read the source data but is not installed. "
            'Run the converter with: uv run --with "zarr>=2.18,<3" examples/rm75/convert_dexgraspvla_to_lerobot.py'
        ) from err
    return zarr.open_group(str(path), mode="r")


def discover_category_episodes(
    category_dir: Path,
    category: str,
    image_cache: Path,
    image_template: str,
    state_key: str | None,
    action_key: str | None,
) -> list[dict]:
    """List the raw episode descriptors of one category (before split/output assignment)."""
    episode_zarrs = sorted(
        (d for d in category_dir.iterdir() if d.is_dir() and is_zarr_group(d)),
        key=lambda d: natural_key(d.name),
    )
    episodes = []
    if episode_zarrs:
        for zarr_dir in episode_zarrs:
            group = open_zarr_readonly(zarr_dir)
            s_key = state_key or resolve_array(group, STATE_KEY_CANDIDATES, "state", zarr_dir)
            a_key = action_key or resolve_array(group, ACTION_KEY_CANDIDATES, "action", zarr_dir)
            num_frames = int(group[s_key].shape[0])
            if int(group[a_key].shape[0]) != num_frames:
                raise ConversionError(
                    f"State/action frame count mismatch in {zarr_dir}: "
                    f"state={group[s_key].shape[0]} action={group[a_key].shape[0]}"
                )
            episodes.append(
                {
                    "category": category,
                    "episode": zarr_dir.name,
                    "episode_index": len(episodes),
                    "source_rel": str(zarr_dir.relative_to(category_dir.parent)),
                    "num_frames": num_frames,
                    "zarr_path": str(zarr_dir),
                    "state_key": s_key,
                    "action_key": a_key,
                    "episode_slice": None,
                }
            )
    else:
        # Single zarr for the whole category, segmented by episode_ends.
        group = open_zarr_readonly(category_dir)
        s_key = state_key or resolve_array(group, STATE_KEY_CANDIDATES, "state", category_dir)
        a_key = action_key or resolve_array(group, ACTION_KEY_CANDIDATES, "action", category_dir)
        ends_key = None
        for key in EPISODE_ENDS_CANDIDATES:
            try:
                group[key]
                ends_key = key
                break
            except Exception:  # noqa: BLE001
                continue
        if ends_key is None:
            raise ConversionError(
                f"{category_dir} contains neither per-episode zarr groups nor an episode_ends array "
                f"(tried: {', '.join(EPISODE_ENDS_CANDIDATES)})."
            )
        ends = np.asarray(group[ends_key])
        start = 0
        for ep_idx, end in enumerate(ends.astype(int)):
            episodes.append(
                {
                    "category": category,
                    "episode": f"{ep_idx:06d}",
                    "episode_index": ep_idx,
                    "source_rel": str(category_dir.relative_to(category_dir.parent)),
                    "num_frames": int(end - start),
                    "zarr_path": str(category_dir),
                    "state_key": s_key,
                    "action_key": a_key,
                    "episode_slice": (start, int(end)),
                }
            )
            start = int(end)

    for ep in episodes:
        ep["image_dir"] = str(episode_image_dir(image_cache, ep, image_template))
    return episodes


def assign_splits(episodes_per_category: dict[str, list[dict]]) -> list[EpisodePlan]:
    """Deterministic balanced 90/10 split + output ordering.

    Within a category, episodes sorted by natural key are eval iff their position modulo
    ``SPLIT_MODULUS`` equals ``SPLIT_REMAINDER``. Output order is: all train episodes
    (categories in canonical order), then all eval episodes (same order), so the held-out
    episodes sit at the end of the dataset — exactly where ``dataset.eval_split`` expects
    them (the last ceil(n * eval_split) episodes per task).
    """
    plan: list[EpisodePlan] = []
    for split in ("train", "eval"):
        selected = {
            category: [
                ep
                for position, ep in enumerate(episodes_per_category.get(category, []))
                if ((position % SPLIT_MODULUS == SPLIT_REMAINDER) == (split == "eval"))
            ]
            for category in TARGET_CATEGORIES
        }
        # Interleave categories so capped validation windows remain category-balanced.
        for position in range(max((len(items) for items in selected.values()), default=0)):
            for category in TARGET_CATEGORIES:
                if position >= len(selected[category]):
                    continue
                ep = selected[category][position]
                plan.append(
                    EpisodePlan(
                        output_index=len(plan),
                        category=category,
                        source_key=f"{ep['source_rel']}#{ep['episode']}",
                        num_frames=ep["num_frames"],
                        split=split,
                        zarr_path=ep["zarr_path"],
                        state_key=ep["state_key"],
                        action_key=ep["action_key"],
                        episode_slice=tuple(ep["episode_slice"]) if ep["episode_slice"] else None,
                        image_dir=ep["image_dir"],
                    )
                )
    return plan


def build_plan(
    source_root: Path,
    image_cache: Path,
    image_template: str,
    state_key: str | None,
    action_key: str | None,
) -> tuple[list[EpisodePlan], dict]:
    """Discover all episodes under the read-only source root and build the conversion plan."""
    if not source_root.is_dir():
        raise ConversionError(f"--source-zarr does not exist or is not a directory: {source_root}")

    # The RM75 source is one root Zarr group. Episode categories and per-control-step
    # RGB paths are metadata arrays, not category subdirectories.
    root_group = open_zarr_readonly(source_root)
    root_keys = (
        "data/state",
        "data/action",
        "data/gripper_image_paths",
        "meta/episode_ends",
        "meta/episode_folders",
    )
    if all(key in root_group for key in root_keys):
        state = root_group["data/state"]
        action = root_group["data/action"]
        image_paths = root_group["data/gripper_image_paths"]
        ends = np.asarray(root_group["meta/episode_ends"], dtype=np.int64)
        folders = np.asarray(root_group["meta/episode_folders"]).astype(str)
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != EXPECTED_STATE_DIM:
            raise ConversionError(
                f"Expected matching state/action [N,8], got {state.shape} and {action.shape}"
            )
        if image_paths.shape[0] != state.shape[0] or len(ends) != len(folders):
            raise ConversionError("State/action/image or episode metadata lengths differ")
        if len(ends) and (np.any(np.diff(ends) <= 0) or ends[-1] != state.shape[0]):
            raise ConversionError("episode_ends must increase and end at the total frame count")
        episodes_per_category = {category: [] for category in TARGET_CATEGORIES}
        start = 0
        for source_index, (stop, folder) in enumerate(zip(ends, folders, strict=True)):
            stop = int(stop)
            parts = Path(folder).parts
            category = parts[-2] if len(parts) >= 2 else None
            if category in TARGET_CATEGORIES:
                episodes_per_category[category].append(
                    {
                        "category": category,
                        "episode": str(source_index),
                        "episode_index": source_index,
                        "source_rel": folder,
                        "num_frames": stop - start,
                        "zarr_path": str(source_root),
                        "state_key": "data/state",
                        "action_key": "data/action",
                        "episode_slice": (start, stop),
                        "image_dir": "",
                    }
                )
            start = stop
        plan = assign_splits(episodes_per_category)
        for ep in plan:
            start, stop = ep.episode_slice
            for stored_path in image_paths[start:stop]:
                resolve_stored_image(source_root, str(stored_path))
        stats = {
            "episodes": len(plan),
            "frames": sum(ep.num_frames for ep in plan),
            "state_dim": EXPECTED_STATE_DIM,
            "num_tasks": 1,
            "train_episodes": sum(ep.split == "train" for ep in plan),
            "eval_episodes": sum(ep.split == "eval" for ep in plan),
            "per_category": {
                category: {
                    "episodes": len(items),
                    "frames": sum(x["num_frames"] for x in items),
                    "eval": sum(i % SPLIT_MODULUS == SPLIT_REMAINDER for i in range(len(items))),
                }
                for category, items in episodes_per_category.items()
            },
        }
        return plan, stats

    episodes_per_category: dict[str, list[dict]] = {}
    for category in TARGET_CATEGORIES:
        category_dir = source_root / category
        if not category_dir.is_dir():
            raise ConversionError(
                f"Category directory '{category}' is missing under {source_root}. "
                f"Expected exactly these category names: {', '.join(TARGET_CATEGORIES)} "
                "(strict name matching; near-misses like 'target' or 'Target10' are rejected)."
            )
        episodes_per_category[category] = discover_category_episodes(
            category_dir, category, image_cache, image_template, state_key, action_key
        )

    extra = sorted(
        d.name
        for d in source_root.iterdir()
        if d.is_dir() and d.name not in TARGET_CATEGORIES and not d.name.startswith(".")
    )
    if extra:
        print(f"[info] Ignoring non-target directories in the source root: {', '.join(extra)}")

    plan = assign_splits(episodes_per_category)

    # Consistency checks that do not depend on the expected full-scale counts.
    for ep in plan:
        if ep.num_frames <= 0:
            raise ConversionError(f"Episode {ep.source_key} has {ep.num_frames} frames")
        group = open_zarr_readonly(Path(ep.zarr_path))
        state_dim = int(group[ep.state_key].shape[1])
        action_dim = int(group[ep.action_key].shape[1])
        if state_dim != action_dim:
            raise ConversionError(
                f"Episode {ep.source_key}: state dim {state_dim} != action dim {action_dim}"
            )
        if state_dim != EXPECTED_STATE_DIM:
            raise ConversionError(
                f"Episode {ep.source_key}: expected {EXPECTED_STATE_DIM}-dim state/action "
                f"(7 joints + gripper), got {state_dim}-dim."
            )
        if ep.num_frames != len(list_cached_frames(Path(ep.image_dir))):
            raise ConversionError(
                f"Episode {ep.source_key}: {ep.num_frames} state/action frames but "
                f"{len(list_cached_frames(Path(ep.image_dir)))} cached images in {ep.image_dir}"
            )

    stats = {
        "episodes": len(plan),
        "frames": sum(ep.num_frames for ep in plan),
        "state_dim": EXPECTED_STATE_DIM,
        "num_tasks": 1,
        "train_episodes": sum(1 for ep in plan if ep.split == "train"),
        "eval_episodes": sum(1 for ep in plan if ep.split == "eval"),
        "per_category": {
            category: {
                "episodes": len(episodes_per_category[category]),
                "frames": sum(ep["num_frames"] for ep in episodes_per_category[category]),
                "eval": sum(
                    1
                    for pos, _ in enumerate(episodes_per_category[category])
                    if pos % SPLIT_MODULUS == SPLIT_REMAINDER
                ),
            }
            for category in TARGET_CATEGORIES
        },
    }
    return plan, stats


def run_precheck(plan: list[EpisodePlan], stats: dict, expected_episodes: int, expected_frames: int) -> None:
    """Full-count precheck against the real dataset (600 episodes / 267,847 frames)."""
    print(
        "[precheck] episodes={episodes} (train={train_episodes} / eval={eval_episodes}) "
        "frames={frames} state_dim={state_dim} tasks={num_tasks}".format(**stats)
    )
    for category, cat_stats in stats["per_category"].items():
        print(f"[precheck]   {category}: {cat_stats}")
    if expected_episodes and stats["episodes"] != expected_episodes:
        raise ConversionError(
            f"Precheck failed: found {stats['episodes']} episodes, expected {expected_episodes}. "
            "Pass --expected-episodes 0 to disable this check."
        )
    if expected_frames and stats["frames"] != expected_frames:
        raise ConversionError(
            f"Precheck failed: found {stats['frames']} frames, expected {expected_frames}. "
            "Pass --expected-frames 0 to disable this check."
        )
    if stats["num_tasks"] != 1:
        raise ConversionError("Precheck failed: the dataset must contain exactly one task.")


def build_features(image_shape: tuple[int, int, int], state_dim: int) -> dict:
    axes = STATE_AXES[:state_dim]
    return {
        "observation.images.gripper": {
            "dtype": "video",
            "shape": tuple(image_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"axes": axes},
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"axes": axes},
        },
    }


def load_episode_arrays(ep: EpisodePlan) -> tuple[np.ndarray, np.ndarray]:
    """Read one episode's state/action arrays from the read-only zarr source."""
    group = open_zarr_readonly(Path(ep.zarr_path))
    episode_slice = slice(*ep.episode_slice) if ep.episode_slice is not None else slice(None)
    state = np.asarray(group[ep.state_key][episode_slice], dtype=np.float32)
    action = np.asarray(group[ep.action_key][episode_slice], dtype=np.float32)
    return state, action


def write_manifest(output_root: Path, manifest: dict) -> None:
    tmp_path = output_root / (MANIFEST_FILENAME + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(output_root / MANIFEST_FILENAME)


def load_manifest(output_root: Path) -> dict:
    return json.loads((output_root / MANIFEST_FILENAME).read_text())


def manifest_matches(manifest: dict, plan: list[EpisodePlan], output_root: Path, args) -> str | None:
    """Return None when the existing manifest describes exactly the current plan."""
    checks = [
        ("task", manifest.get("task"), TASK_INSTRUCTION),
        ("fps", manifest.get("fps"), FPS),
        ("robot_type", manifest.get("robot_type"), ROBOT_TYPE),
        ("source_root", manifest.get("source_root"), str(args.source_zarr)),
        ("num_episodes", len(manifest.get("episodes", [])), len(plan)),
    ]
    for name, got, want in checks:
        if got != want:
            return f"{name}: manifest has {got!r}, current source resolves to {want!r}"
    for entry, ep in zip(manifest["episodes"], plan, strict=True):
        if entry != ep.as_manifest_entry():
            return f"episode {ep.output_index}: manifest has {entry}, current source resolves to {ep.as_manifest_entry()}"
    return None


def convert(args) -> None:
    args.source_zarr = args.source_zarr.resolve()
    args.output_root = args.output_root.resolve()
    output_root: Path = args.output_root

    # The production source stores image paths in Zarr; the legacy cache discovery is
    # retained only for small compatibility fixtures.
    source_group = open_zarr_readonly(args.source_zarr)
    is_root_layout = "data/gripper_image_paths" in source_group
    image_cache = (
        args.source_zarr.parent / "gripper_frames"
        if is_root_layout
        else discover_image_cache(args.source_zarr, args.image_cache)
    )
    print(f"[info] Image cache (read-only): {image_cache}")

    plan, stats = build_plan(
        args.source_zarr, image_cache, args.image_dir_template, args.state_key, args.action_key
    )
    run_precheck(plan, stats, args.expected_episodes, args.expected_frames)

    if args.dry_run:
        print("[dry-run] Precheck passed. Conversion plan (first 5 / last 5):")
        sample = plan[:5] + plan[-5:] if len(plan) > 10 else plan
        for ep in sample:
            print(f"[dry-run]   {ep.as_manifest_entry()}")
        return

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # ── Overwrite protection / resume contract ────────────────────────────────
    # NOTE: LeRobotDataset.create() insists on creating the root itself (exist_ok=False),
    # so the fresh path must hand it a non-existent root directory.
    has_dataset = (output_root / "meta").is_dir()
    has_manifest = (output_root / MANIFEST_FILENAME).is_file()
    completed: list[int] = []
    if args.resume:
        if not has_manifest:
            raise ConversionError(
                f"--resume requires an existing manifest at {output_root / MANIFEST_FILENAME}. "
                "Without it a partial conversion cannot be verified; delete the output and reconvert."
            )
        mismatch = manifest_matches(load_manifest(output_root), plan, output_root, args)
        if mismatch is not None:
            raise ConversionError(
                f"--resume: the manifest does not match the current source ({mismatch}). "
                "A resume is only allowed on an identical conversion plan; delete the output and reconvert."
            )
        completed = list(load_manifest(output_root)["completed"])
        if has_dataset:
            meta = LeRobotDatasetMetadata(args.repo_id, root=output_root)
            if meta.total_episodes != len(completed):
                raise ConversionError(
                    f"--resume: the dataset holds {meta.total_episodes} episodes but the manifest "
                    f"records only {len(completed)} completed ones. A crashed save_episode cannot be "
                    "repaired in place; delete the output and reconvert."
                )
            dataset = LeRobotDataset.resume(
                args.repo_id,
                root=output_root,
                image_writer_threads=args.image_writer_threads,
                image_writer_processes=args.image_writer_processes,
            )
        else:
            raise ConversionError(
                f"--resume: {output_root} has a manifest but no dataset metadata. A manifest is "
                "only written after the dataset is created, so the output is corrupt; delete the "
                "output and reconvert."
            )
    else:
        existing = sorted(p.name for p in output_root.iterdir()) if output_root.is_dir() else []
        if existing:
            raise ConversionError(
                f"Output root {output_root} is not empty ({len(existing)} entries). Refusing to "
                "overwrite. Pass --resume to continue a matching partial conversion, or delete "
                "the directory yourself."
            )
        if output_root.is_dir():
            # Empty leftover from a previous crashed run; create() needs the root gone.
            output_root.rmdir()
        dataset = create_dataset(args, plan, image_cache)

    # ── Per-episode conversion ────────────────────────────────────────────────
    if args.resume and has_manifest:
        manifest = load_manifest(output_root)
    else:
        manifest = new_manifest(args, plan, stats, image_cache)
        write_manifest(output_root, manifest)
    pending = [ep for ep in plan if ep.output_index not in set(completed)]
    print(f"[info] {len(completed)} episodes already converted, {len(pending)} to go.")

    start_time = time.time()
    frames_done = 0
    for _i, ep in enumerate(pending):
        state, action = load_episode_arrays(ep)
        if ep.image_dir:
            frame_paths = list_cached_frames(Path(ep.image_dir))
        else:
            start, stop = ep.episode_slice
            group = open_zarr_readonly(Path(ep.zarr_path))
            frame_paths = [
                resolve_stored_image(args.source_zarr, str(path))
                for path in group["data/gripper_image_paths"][start:stop]
            ]
        for t in range(ep.num_frames):
            dataset.add_frame(
                {
                    "observation.images.gripper": read_image_rgb(frame_paths[t]),
                    "observation.state": state[t],
                    "action": action[t],
                    "task": TASK_INSTRUCTION,
                }
            )
        dataset.save_episode()
        frames_done += ep.num_frames

        manifest["completed"].append(ep.output_index)
        write_manifest(output_root, manifest)

        elapsed = time.time() - start_time
        print(
            f"[convert] episode {ep.output_index + 1}/{len(plan)} ({ep.source_key}, "
            f"{ep.split}, {ep.num_frames} frames) | {frames_done} frames in {elapsed:.0f}s"
        )

    dataset.finalize()
    print(
        f"[done] Dataset written to {output_root} ({stats['frames']} frames, {stats['episodes']} episodes)."
    )


def create_dataset(args, plan: list[EpisodePlan], image_cache: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if plan[0].image_dir:
        first_path = list_cached_frames(Path(plan[0].image_dir))[0]
    else:
        group = open_zarr_readonly(Path(plan[0].zarr_path))
        first_path = resolve_stored_image(
            args.source_zarr, str(group["data/gripper_image_paths"][plan[0].episode_slice[0]])
        )
    first_image = read_image_rgb(first_path)
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        robot_type=ROBOT_TYPE,
        root=args.output_root,
        features=build_features(first_image.shape, EXPECTED_STATE_DIM),
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


def new_manifest(args, plan: list[EpisodePlan], stats: dict, image_cache: Path) -> dict:
    return {
        "format": "rm75-dexgraspvla-conversion/1.0",
        "task": TASK_INSTRUCTION,
        "fps": FPS,
        "robot_type": ROBOT_TYPE,
        "repo_id": args.repo_id,
        "source_root": str(args.source_zarr),
        "image_cache": str(image_cache),
        "image_dir_template": args.image_dir_template,
        "split": {"modulus": SPLIT_MODULUS, "remainder": SPLIT_REMAINDER},
        "stats": stats,
        "episodes": [ep.as_manifest_entry() for ep in plan],
        "completed": [],
    }


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-zarr",
        type=Path,
        required=True,
        help="Read-only DexGraspVLA source root containing Target/, Target2/ ... Target5/.",
    )
    parser.add_argument(
        "--image-cache",
        type=Path,
        default=None,
        help="Read-only image cache root (default: auto-discover next to the source root).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination dataset directory, e.g. data/rm75_toy_robot_lerobot.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="rm75_toy_robot",
        help="Dataset id recorded in the metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a partial conversion; the existing manifest must match the source exactly.",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=EXPECTED_EPISODES,
        help="Precheck: abort unless the source holds this many episodes (0 disables).",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=EXPECTED_FRAMES,
        help="Precheck: abort unless the source holds this many frames (0 disables).",
    )
    parser.add_argument(
        "--image-dir-template",
        type=str,
        default="{category}/{episode}",
        help="Cache-relative episode frame directory template (default: %(default)s).",
    )
    parser.add_argument("--state-key", type=str, default=None, help="Override the zarr state array path.")
    parser.add_argument("--action-key", type=str, default=None, help="Override the zarr action array path.")
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Threads used by the LeRobot video encoder (default: %(default)s).",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Processes used by the LeRobot video encoder (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery + precheck only; write nothing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_arg_parser().parse_args(argv)
    try:
        convert(args)
    except ConversionError as err:
        print(f"[error] {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
