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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def training_capture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    # 替身 uv 执行真实的轻量启动器，仅替换硬件探测与最终 exec，避免启动训练。
    executable = tmp_path / "uv"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
        "args = sys.argv[1:]\n"
        "if 'which' in args:\n"
        "    print('/tmp/lerobot-train')\n"
        "    sys.exit(0)\n"
        "def capture(command, environment):\n"
        "    path = Path(os.environ['TRAINING_TEST_CAPTURE'])\n"
        "    path.write_text('\\n'.join(command) + '\\n')\n"
        "    path.with_suffix('.json').write_text(json.dumps(environment))\n"
        "if 'scripts/launch_handumi_training.py' in args:\n"
        "    from scripts import launch_handumi_training as launcher\n"
        "    launcher.check_gpu_availability = lambda ids, environment: None\n"
        "    launcher.os.execvpe = lambda executable, command, environment: capture(command, environment)\n"
        "    sys.argv = args[args.index('scripts/launch_handumi_training.py'):]\n"
        "    launcher.main()\n"
        "else:\n"
        "    capture(args, dict(os.environ))\n"
    )
    executable.chmod(0o755)
    capture = tmp_path / "args.txt"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "TRAINING_TEST_CAPTURE": str(capture)}
    env.pop("TRAINING_RECIPE_VARIANT", None)
    return capture, env


@pytest.mark.parametrize(
    ("recipe", "variant", "dims", "color", "ema"),
    [
        (None, None, "[256,512,1024]", True, True),
        ("baseline", None, "[512,1024,2048]", False, False),
        ("baseline", "small_only", "[256,512,1024]", False, False),
        ("baseline", "color_only", "[512,1024,2048]", True, False),
        ("regularized", None, "[256,512,1024]", True, True),
    ],
)
def test_train_recipe_arguments(
    tmp_path: Path,
    training_capture: tuple[Path, dict[str, str]],
    recipe: str | None,
    variant: str | None,
    dims: str,
    color: bool,
    ema: bool,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    capture, env = training_capture
    if variant is not None:
        env["TRAINING_RECIPE_VARIANT"] = variant
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_dp_handumi_tcp.sh"
    args = ["bash", str(script), str(dataset)]
    if recipe is not None:
        args.extend(["--recipe", recipe])
    subprocess.run(args, env=env, check=True)
    captured = capture.read_text().splitlines()

    assert f"--policy.down_dims={dims}" in captured
    assert "--nproc-per-node=2" in captured
    assert "--dataset.eval_split_strategy=random" in captured
    assert "--dataset.eval_split_seed=1000" in captured
    assert "--eval_seed=1000" in captured and "--eval_steps=500" in captured
    assert "--steps=19466" in captured
    assert ("--ema.type=warmup" in captured) is ema
    assert ("--eval_use_ema=true" in captured) is ema
    assert f"--dataset.image_transforms.enable={str(color).lower()}" in captured
    if color:
        transforms = next(
            arg.split("=", 1)[1] for arg in captured if arg.startswith("--dataset.image_transforms.tfs=")
        )
        assert json.loads(transforms) == {
            "color": {
                "weight": 1.0,
                "type": "ColorJitter",
                "kwargs": {"brightness": [0.8, 1.2], "contrast": [0.8, 1.2], "saturation": [0.9, 1.1]},
            }
        }
        assert "--dataset.image_transforms.backend=gpu" in captured
    else:
        assert not any(arg.startswith("--dataset.image_transforms.tfs=") for arg in captured)


@pytest.mark.parametrize("equals_form", [False, True])
def test_train_yaml_arguments(
    tmp_path: Path, training_capture: tuple[Path, dict[str, str]], equals_form: bool
) -> None:
    # 从仓库外传入带空格的相对路径，并确保旧配方的参数不会覆盖 YAML。
    repo = Path(__file__).resolve().parents[2]
    config = tmp_path / "training config.yaml"
    config.write_bytes((repo / "scripts" / "config" / "umi" / "dp_handumi_tcp_train.yaml").read_bytes())
    capture, env = training_capture
    script = repo / "scripts" / "train_dp_handumi_tcp.sh"
    options = [f"--config_path={config.name}"] if equals_form else ["--config_path", config.name]
    subprocess.run(["bash", str(script), *options], env=env, cwd=tmp_path, check=True)
    captured = capture.read_text().splitlines()

    entry = captured.index("/tmp/lerobot-train")
    assert "--nproc-per-node=2" in captured[:entry]
    config_arg, output_arg = captured[entry + 1 :]
    assert config_arg == f"--config_path={config}"
    assert output_arg.startswith("--output_dir=outputs/train/dp_handumi_tcp_")


@pytest.mark.parametrize("mode", ["episode", "current"])
def test_native_training_configs(mode: str, training_capture: tuple[Path, dict[str, str]]) -> None:
    """新配置通过实际解析器与原脚本入口加载，无需启动双卡训练。"""
    import draccus

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    repo = Path(__file__).resolve().parents[2]
    config = repo / "scripts" / "config" / "umi" / f"dp_handumi_tcp_native_{mode}_train.yaml"
    cfg = draccus.parse(TrainPipelineConfig, config_path=config, args=[])
    assert isinstance(cfg.policy, DiffusionConfig)
    assert cfg.dataset.root == f"data/handumi_data/umi-demo-tcp-native-{mode}"
    assert cfg.dataset.repo_id == f"handumi/umi-demo-tcp-native-{mode}"
    assert cfg.gpu_ids == ([0] if mode == "episode" else [1])
    assert cfg.policy.n_obs_steps == 2 and cfg.policy.serial_observation
    assert cfg.policy.resize_shape == (224, 224) and cfg.policy.resize_mode == "letterbox"
    assert cfg.policy.use_relative_pose is (mode == "current")
    assert cfg.policy.action_delta_indices == list(range(-1, 15))
    assert cfg.policy.n_action_steps == 8 and cfg.policy.crop_shape is None
    capture, env = training_capture
    subprocess.run(
        ["bash", str(repo / "scripts" / "train_dp_handumi_tcp.sh"), "--config_path", str(config)],
        env=env,
        check=True,
    )
    assert f"--config_path={config}" in capture.read_text().splitlines()
    assert "--nproc-per-node=1" in capture.read_text().splitlines()
    assert json.loads(capture.with_suffix(".json").read_text())["CUDA_VISIBLE_DEVICES"] == (
        "0" if mode == "episode" else "1"
    )


@pytest.mark.parametrize("gpu_ids", [[0], [1], [0, 1], [1, 0], None])
def test_yaml_gpu_selection(
    tmp_path: Path, training_capture: tuple[Path, dict[str, str]], gpu_ids: list[int] | None
) -> None:
    """从真实 shell 入口检查设备覆盖、rank 顺序、进程数和省略字段的兼容性。"""
    config = tmp_path / "gpu.yaml"
    config.write_text("job_name: test_gpu\n" + (f"gpu_ids: {gpu_ids}\n" if gpu_ids is not None else ""))
    capture, env = training_capture
    env["CUDA_VISIBLE_DEVICES"] = "1,0"
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_dp_handumi_tcp.sh"
    subprocess.run(["bash", str(script), "--config_path", str(config)], env=env, check=True)
    expected = ",".join(map(str, gpu_ids)) if gpu_ids is not None else "1,0"
    assert json.loads(capture.with_suffix(".json").read_text())["CUDA_VISIBLE_DEVICES"] == expected
    assert (
        f"--nproc-per-node={len(gpu_ids) if gpu_ids is not None else 2}" in capture.read_text().splitlines()
    )
