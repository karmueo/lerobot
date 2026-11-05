#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

"""Wall-OSS-0.5 策略入口：配置直接导出，模型和处理器按需加载可选依赖。"""

from typing import TYPE_CHECKING, Any

from .configuration_wall_oss_05 import WallOSS05Config

if TYPE_CHECKING:
    from .modeling_wall_oss_05 import WallOSS05Policy
    from .processor_wall_oss_05 import make_wall_oss_05_pre_post_processors

__all__ = ["WallOSS05Config", "WallOSS05Policy", "make_wall_oss_05_pre_post_processors"]


def __getattr__(name: str) -> Any:
    """仅在访问 Wall-OSS-0.5 运行时接口时加载实现，避免影响其他策略的配置注册。"""
    if name == "WallOSS05Policy":
        from .modeling_wall_oss_05 import WallOSS05Policy

        globals()[name] = WallOSS05Policy
        return WallOSS05Policy
    if name == "make_wall_oss_05_pre_post_processors":
        from .processor_wall_oss_05 import make_wall_oss_05_pre_post_processors

        globals()[name] = make_wall_oss_05_pre_post_processors
        return make_wall_oss_05_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
