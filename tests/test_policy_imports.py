# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""验证配置注册不会提前导入其他策略的可选运行时依赖。"""

import subprocess
import sys


def test_policy_configs_import_without_unrelated_vlm_runtimes() -> None:
    """阻断 G0.5 和 Wall-OSS 实现，确保 Diffusion 配置仍可正常注册。"""
    code = """
import importlib.abc
import sys

blocked_modules = {
    "lerobot.policies.g05.modeling_g05",
    "lerobot.policies.g05.processor_g05",
    "lerobot.policies.wall_oss_05.modeling_wall_oss_05",
    "lerobot.policies.wall_oss_05.processor_wall_oss_05",
}

class BlockVLMRuntimes(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in blocked_modules:
            raise ModuleNotFoundError(f"blocked optional VLM runtime: {fullname}")
        return None

# 隔离当前测试进程的模块缓存，防止提前导入掩盖回归。
sys.meta_path.insert(0, BlockVLMRuntimes())
from lerobot.policies import DiffusionConfig, G05Config, WallOSS05Config, make_policy_config
from lerobot.policies.g05 import G05Config as PackageG05Config
from lerobot.policies.wall_oss_05 import WallOSS05Config as PackageWallOSS05Config

assert isinstance(make_policy_config("diffusion"), DiffusionConfig)
assert isinstance(make_policy_config("g05"), G05Config)
assert isinstance(make_policy_config("wall_oss_05"), WallOSS05Config)
assert PackageG05Config is G05Config
assert PackageWallOSS05Config is WallOSS05Config
assert blocked_modules.isdisjoint(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
