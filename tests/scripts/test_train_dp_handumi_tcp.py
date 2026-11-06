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
from pathlib import Path

import pytest


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
    tmp_path: Path, recipe: str | None, variant: str | None, dims: str, color: bool, ema: bool
) -> None:
    # 用替身 uv 捕获命令，确保测试不会启动实际训练或更改数据集。
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    executable = tmp_path / "uv"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == "which" ]]; then echo /tmp/lerobot-train; exit 0; fi\n'
        "done\n"
        'printf "%s\\n" "$@" > "$TRAINING_TEST_CAPTURE"\n'
    )
    executable.chmod(0o755)
    capture = tmp_path / "args.txt"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "TRAINING_TEST_CAPTURE": str(capture)}
    if variant is not None:
        env["TRAINING_RECIPE_VARIANT"] = variant
    else:
        env.pop("TRAINING_RECIPE_VARIANT", None)
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_dp_handumi_tcp.sh"
    args = ["bash", str(script), str(dataset)]
    if recipe is not None:
        args.extend(["--recipe", recipe])
    subprocess.run(args, env=env, check=True)
    captured = capture.read_text().splitlines()

    assert f"--policy.down_dims={dims}" in captured
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
