# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

"""G0.5 策略入口：配置直接导出，模型和处理器按需加载可选依赖。"""

from typing import TYPE_CHECKING, Any

from .configuration_g05 import G05Config

if TYPE_CHECKING:
    from .modeling_g05 import G05Policy
    from .processor_g05 import make_g05_pre_post_processors

__all__ = ["G05Config", "G05Policy", "make_g05_pre_post_processors"]


def __getattr__(name: str) -> Any:
    """仅在访问 G0.5 运行时接口时加载实现，避免影响其他策略的配置注册。"""
    if name == "G05Policy":
        from .modeling_g05 import G05Policy

        globals()[name] = G05Policy
        return G05Policy
    if name == "make_g05_pre_post_processors":
        from .processor_g05 import make_g05_pre_post_processors

        globals()[name] = make_g05_pre_post_processors
        return make_g05_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
