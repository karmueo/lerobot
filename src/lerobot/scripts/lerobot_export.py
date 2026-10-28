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
"""将本地 Diffusion 检查点导出为可验证的双网络模型包."""

from lerobot.configs import parser
from lerobot.export.exporter import ExportConfig, export_policy
from lerobot.utils.utils import init_logging


@parser.wrap()
def export(cfg: ExportConfig):
    """解析配置并执行导出；异常保留原始上下文以便定位."""
    init_logging()
    export_policy(cfg)


def main():
    """命令行入口."""
    export()


if __name__ == "__main__":
    main()
