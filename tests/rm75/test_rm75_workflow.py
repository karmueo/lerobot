"""Tests for the RM75 single-root DexGraspVLA conversion workflow."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

zarr = pytest.importorskip("zarr")
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/rm75/convert_dexgraspvla_to_lerobot.py"
CATEGORIES = ("Target", "Target2", "Target3", "Target4", "Target5")


def load_converter():
    spec = importlib.util.spec_from_file_location("rm75_converter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses inspect sys.modules while decorating.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def converter():
    return load_converter()


def make_source(root, counts=None, frames=2):
    counts = counts or dict.fromkeys(CATEGORIES, 10)
    source = root / "gripper_zarr"
    image_root = root / "gripper_frames"
    group = zarr.open_group(str(source), mode="w")
    states, actions, paths, ends, folders = [], [], [], [], []
    total = 0
    for category_index, category in enumerate(CATEGORIES):
        for episode in range(counts.get(category, 0)):
            rng = np.random.default_rng(category_index * 1000 + episode)
            state = rng.uniform(-1, 1, (frames, 8)).astype(np.float32)
            action = (state * np.float32(0.5)).astype(np.float32)
            states.append(state)
            actions.append(action)
            folder = f"source_000/{category}/episode_{episode}"
            folders.append(folder)
            for frame in range(frames):
                path = image_root / folder / "gripper" / f"frame_{frame:06d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                rgb = np.zeros((4, 5, 3), dtype=np.uint8)
                rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
                Image.fromarray(rgb).save(path)
                paths.append(str(Path("../gripper_frames") / folder / "gripper" / path.name))
            total += frames
            ends.append(total)
    group.create_dataset("data/state", data=np.concatenate(states))
    group.create_dataset("data/action", data=np.concatenate(actions))
    group.create_dataset("data/gripper_image_paths", data=np.asarray(paths, dtype="U256"))
    group.create_dataset("meta/episode_ends", data=np.asarray(ends, dtype=np.int64))
    group.create_dataset("meta/episode_folders", data=np.asarray(folders, dtype="U128"))
    return source


def test_single_root_discovery_and_balanced_split(converter, tmp_path):
    source = make_source(tmp_path, {"Target": 20, "Target2": 10, "Target3": 10, "Target4": 10, "Target5": 10})
    plan, stats = converter.build_plan(source, None, "unused", None, None)
    assert stats["episodes"] == 60
    assert stats["train_episodes"] == 54
    assert stats["eval_episodes"] == 6
    assert stats["frames"] == 120
    assert [episode.split for episode in plan] == ["train"] * 54 + ["eval"] * 6
    assert {episode.category for episode in plan[-6:]} == set(CATEGORIES)
    assert [episode.output_index for episode in plan] == list(range(60))


def test_exact_category_filter(converter, tmp_path):
    source = make_source(tmp_path)
    group = zarr.open_group(str(source), mode="r+")
    folders = np.asarray(group["meta/episode_folders"]).astype(str)
    folders[0] = "source_000/Target10/episode_0"
    del group["meta/episode_folders"]
    group.create_dataset("meta/episode_folders", data=folders.astype("U128"))
    plan, stats = converter.build_plan(source, None, "unused", None, None)
    assert stats["episodes"] == 49
    assert all("Target10" not in episode.source_key for episode in plan)


def test_rgb_and_relative_image_paths(converter, tmp_path):
    source = make_source(tmp_path)
    group = zarr.open_group(str(source), mode="r")
    stored = str(group["data/gripper_image_paths"][0])
    path = converter.resolve_stored_image(source, stored)
    image = converter.read_image_rgb(path)
    assert image.shape == (4, 5, 3)
    assert image.dtype == np.uint8
    assert tuple(image[0, 0]) == pytest.approx((10, 20, 30), abs=3)


def test_missing_referenced_image_rejected(converter, tmp_path):
    source = make_source(tmp_path)
    group = zarr.open_group(str(source), mode="r")
    converter.resolve_stored_image(source, str(group["data/gripper_image_paths"][0])).unlink()
    with pytest.raises(converter.ConversionError, match="missing"):
        converter.build_plan(source, None, "unused", None, None)


def test_dry_run_writes_nothing(converter, tmp_path):
    source = make_source(tmp_path)
    output = tmp_path / "output"
    result = converter.main(
        [
            "--source-zarr",
            str(source),
            "--output-root",
            str(output),
            "--expected-episodes",
            "0",
            "--expected-frames",
            "0",
            "--dry-run",
        ]
    )
    assert result == 0
    assert not output.exists()
