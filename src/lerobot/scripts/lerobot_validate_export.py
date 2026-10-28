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
"""验证已导出的 Diffusion 模型包，不重复构建 engine."""

from lerobot.configs import parser
from lerobot.export.validation import ValidationConfig, validate_export
from lerobot.utils.utils import init_logging


@parser.wrap()
def validate(cfg: ValidationConfig):
    """失败返回非零状态，完整结果仍保存于模型包."""
    init_logging()
    report = validate_export(cfg)
    if report["status"] != "passed":
        raise SystemExit(1)


def main():
    """独立验证 CLI 入口."""
    validate()


if __name__ == "__main__":
    main()
