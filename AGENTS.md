本文件为 AI 代理在本仓库中处理代码提供指导。

> **面向用户的帮助 → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)**（涵盖 SO-101 配置、数据录制、策略选择、训练时长和评估，并提供可直接复制执行的命令）。

## 项目概述

LeRobot 是一个基于 PyTorch 的真实世界机器人库，提供数据集、预训练策略，以及训练、评估、数据采集和机器人控制工具。它集成了 Hugging Face Hub，用于共享模型和数据集。

## 技术栈

Python 3.12+ · PyTorch · Hugging Face（datasets、Hub、accelerate）· draccus（配置与命令行）· Gymnasium（环境）· uv（包管理）

## 开发环境配置

```bash
uv sync --locked                            # 安装基础依赖
uv sync --locked --extra test --extra dev   # 安装测试和开发工具
uv sync --locked --extra all                # 安装全部依赖
git lfs install && git lfs pull            # 安装 Git LFS 并拉取测试资源
```

## 常用命令

```bash
uv run pytest tests -svv --maxfail=10                 # 运行全部测试
DEVICE=cuda make test-end-to-end                      # 运行全部端到端测试
pre-commit run --all-files                           # 检查与格式化（ruff、typos、bandit 等）
```

## 代码架构（`src/lerobot/`）

- **`scripts/`** — 命令行入口（`lerobot-train`、`lerobot-eval`、`lerobot-record` 等），在 `pyproject.toml` 的 `[project.scripts]` 中配置映射。
- **`configs/`** — 由 draccus 解析的数据类配置。`train.py` 包含顶层配置 `TrainPipelineConfig`，`policies.py` 包含基类 `PreTrainedConfig`。通过 `draccus.ChoiceRegistry` 和 `@register_subclass("name")` 装饰器实现多态。
- **`policies/`** — 每种策略位于独立子目录中，均继承 `pretrained.py` 中的 `PreTrainedPolicy`（`nn.Module` + `HubMixin`）。`factory.py` 中的工厂采用延迟导入。
- **`processor/`** — 数据转换流水线。基类 `ProcessorStep` 配有注册机制，`DataProcessorPipeline` / `PolicyProcessorPipeline` 将各处理步骤串联起来。
- **`datasets/`** — 包含 `LeRobotDataset`（支持按回合采样和视频解码）与 `LeRobotDatasetMetadata`。
- **`envs/`** — 基类 `EnvConfig` 位于 `configs.py`，工厂位于 `factory.py`。每个环境子类定义 `gym_kwargs` 和 `create_envs()`。
- **`robots/`、`motors/`、`cameras/`、`teleoperators/`** — 硬件抽象层。
- **`types.py` 和 `configs/types.py`** — 核心类型别名和特征类型定义。

## 仓库结构（`src/` 之外）

- **`tests/`** — 按模块组织的 Pytest 测试套件。测试夹具位于 `tests/fixtures/`，模拟对象位于 `tests/mocks/`。硬件测试使用 `tests/utils.py` 中的跳过装饰器。通过 `Makefile` 运行的端到端测试将结果写入 `tests/outputs/`。
- **`.github/workflows/`** — 持续集成工作流：`quality.yml`（pre-commit）、`fast_tests.yml`（基础依赖，每次 PR 运行）、`full_tests.yml`（全部可选依赖、端到端测试和 GPU 测试，审批后运行）、`latest_deps_tests.yml`（每日更新依赖锁定文件）、`security.yml`（TruffleHog）、`release.yml`（推送标签时发布到 PyPI）。
- **`docs/source/`** — Hugging Face 文档（`.mdx` 文件），包含各策略的 README、硬件指南和教程。通过 `docs-requirements.txt` 和持续集成工作流单独构建。
- **`examples/`** — 按使用场景组织的用户教程和脚本（数据集创建、训练、硬件配置）。
- **`docker/`** — 面向用户的 `Dockerfile.user` 和面向持续集成的 `Dockerfile.internal`。
- **`benchmarks/`** — 性能基准测试脚本。
- **根目录文件**：`pyproject.toml`（依赖、构建和工具配置的统一来源）、`Makefile`（端到端测试目标）、`uv.lock`，以及提供通用说明的 `CONTRIBUTING.md` 和 `README.md`。

## 注意事项

- **中文注释**：生成的代码必须添加中文注释，说明关键逻辑、函数或方法的用途，以及必要的参数含义和边界条件。注释应准确、简洁，避免仅重复代码字面含义；修改代码时同步更新相关注释。
- **Mypy 采用渐进式类型检查**：仅对 `lerobot.envs`、`lerobot.configs`、`lerobot.optim`、`lerobot.model`、`lerobot.cameras`、`lerobot.motors`、`lerobot.transport` 启用严格检查。修改这些模块时需添加类型注解。
- **导入规范**：优先在模块顶部导入；同一模块内的同级文件之间使用相对导入（`from .sibling import X`），跨模块使用绝对导入（`from lerobot.module import X`）。
- **可选依赖**：许多策略、环境和机器人依赖可选依赖组（例如 `lerobot[aloha]`，详见 `pyproject.toml`）。在模块顶部通过 `TYPE_CHECKING or _foo_available` 条件保护可选导入，并在使用时调用 `require_package(...)` 检查依赖。复用 `utils/import_utils.py` 中的 `_foo_available` 标志，不要调用 `is_package_available`。
- **视频解码**：数据集可将观测数据存储为视频文件。`LeRobotDataset` 负责提取视频帧，但运行测试需要安装 ffmpeg。
- **优先使用 `uv run` 执行 Python 命令**，不要直接使用 `python` 或 `pip`。
