#!/usr/bin/env bash
# 仅检查或安装本地部署所需的硬件依赖，保留 Jetson 定制 Torch。
set -euo pipefail

mode="${1:---check}"
if [[ "$#" -gt 1 || ( "$mode" != --check && "$mode" != --install ) ]]; then
    printf '用法: bash %s [--check|--install]\n' "$0" >&2
    exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/scl/work/FastUMI_Data/ros2_ws/.deps/unitree_gripper/install/cyclonedds}"
export RM75B_SDK_ROOT="${RM75B_SDK_ROOT:-/home/scl/work/RM_API2/Python}"

if [[ "$mode" == --install ]]; then
    unitree_source="${RM75B_UNITREE_SDK_SOURCE:-/home/scl/work/unitree/unitree_sdk2_python-master}"
    uvc_package="${RM75B_UVC_PACKAGE:-pupil-labs-uvc==1.0.4}"
    if [[ ! -f "$CYCLONEDDS_HOME/lib/libddsc.so.0.10.2" || ! -d "$unitree_source/unitree_sdk2py" ]]; then
        printf '%s\n' '缺少 CycloneDDS 0.10.2 C 库或 Unitree Python 源码，请检查对应环境变量。' >&2
        exit 1
    fi
    project_python="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
    if [[ "$project_python" != */bin/python* || ! -f "${project_python%/bin/*}/pyvenv.cfg" ]]; then
        printf '部署依赖必须安装到项目虚拟环境，当前解释器为 %s\n' "$project_python" >&2
        exit 1
    fi
    torch_versions="$(uv run --no-sync python -c 'from importlib.metadata import version; print(version("torch"), version("torchvision"))')"
    staging="$(mktemp -d -t lerobot-rm75b-setup.XXXXXXXX)"
    trap 'rm -rf -- "$staging"' EXIT
    # 从临时目录构建 SDK，防止构建产物写入现有 Unitree 源码。
    cp -a -- "$unitree_source"/{unitree_sdk2py,README.md,setup.py,pyproject.toml} "$staging"
    # 跳过 SDK 安装元信息中的 opencv-python，项目已有 opencv-python-headless。
    uv pip install --python "$project_python" --no-deps --no-binary cyclonedds \
        'cyclonedds==0.10.2' "$staging" "$uvc_package"
    actual_versions="$(uv run --no-sync python -c 'from importlib.metadata import version; print(version("torch"), version("torchvision"))')"
    if [[ "$actual_versions" != "$torch_versions" ]]; then
        printf 'Torch 版本发生变化，预期 %s，实际 %s\n' "$torch_versions" "$actual_versions" >&2
        exit 1
    fi
fi

# 只导入模块及加载本地动态库。此检查不创建控制器或订阅设备。
uv run --no-sync python - <<'PY'
import importlib
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from lerobot.deployment.rm75b.arm import load_rm_sdk

failures = []
for package in ("torch", "torchvision", "numpy", "diffusers", "cyclonedds", "unitree-sdk2py", "pupil-labs-uvc"):
    try:
        print(f"{package} {version(package)}")
    except PackageNotFoundError:
        failures.append(f"缺少 {package}")
for module in ("cv2", "uvc", "cyclonedds", "unitree_sdk2py.core.channel"):
    try:
        importlib.import_module(module)
    except Exception as error:
        failures.append(f"{module} 导入失败 {error}")
try:
    load_rm_sdk(Path(os.environ["RM75B_SDK_ROOT"]))
except Exception as error:
    failures.append(f"RM SDK 加载失败 {error}")
if failures:
    print("\n".join(failures), file=sys.stderr)
    raise SystemExit(1)
print("硬件依赖导入检查通过。")
PY
