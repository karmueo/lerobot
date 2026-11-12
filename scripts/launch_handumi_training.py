#!/usr/bin/env python
"""在导入训练程序前，依据 YAML 选择机器 GPU 并启动对应数量的训练进程。"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


def validate_gpu_ids(value: object) -> list[int]:
    """仅接受非空、不重复的非负整数列表，拒绝 bool、字符串和浮点编号。"""
    if not isinstance(value, list) or not value:
        raise ValueError("gpu_ids 必须是非空 GPU 编号列表，例如 [0]、[1] 或 [0, 1]")
    if any(type(gpu) is not int or gpu < 0 for gpu in value):
        raise ValueError("gpu_ids 的每个编号必须是非负整数")
    if len(set(value)) != len(value):
        raise ValueError("gpu_ids 不能包含重复编号")
    return value


def check_gpu_availability(gpu_ids: list[int], environment: dict[str, str]) -> None:
    """在全新进程中检测 CUDA，避免父进程缓存旧的可见设备或初始化 CUDA 上下文。

    环境已包含最终的可见设备列表；CUDA 对不存在或不可用的编号会返回不足的设备数。
    """
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise ValueError(f"无法检测 GPU {gpu_ids}: {result.stderr.strip()}")
    try:
        count = int(result.stdout.strip())
    except ValueError as error:
        raise ValueError(f"无法解析 GPU 检测结果: {result.stdout.strip()}") from error
    if count != len(gpu_ids):
        raise ValueError(f"gpu_ids={gpu_ids} 包含不存在或不可用的 GPU，仅检测到 {count} 张可用卡")


def prepare_launch(
    config_path: Path,
    training_entry: str,
    timestamp: str,
    environment: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """生成 torchrun 参数及独立环境，不修改调用方环境或原始配置文件。"""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("训练 YAML 顶层必须是配置映射")
    child_environment = dict(os.environ if environment is None else environment)
    gpu_ids = config.get("gpu_ids")
    processes = 2
    if gpu_ids is not None:
        gpu_ids = validate_gpu_ids(gpu_ids)
        policy = config.get("policy") or {}
        if policy.get("device", "cuda") == "cpu":
            raise ValueError("gpu_ids 指定 GPU 训练时，policy.device 不能为 cpu")
        # 显式列表覆盖继承的 mask，并按机器 PCI 顺序解释编号；保留列表指定的 rank 顺序。
        child_environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        child_environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        check_gpu_availability(gpu_ids, child_environment)
        processes = len(gpu_ids)
    # 任务名进入目录前移除路径分隔符；纳秒时间戳避免并发运行复用同一输出目录。
    job_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config.get("job_name") or "dp_handumi_tcp_yaml"))
    output = Path("outputs/train") / f"{job_name}_{timestamp}"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={processes}",
        training_entry,
        f"--config_path={config_path}",
        f"--output_dir={output}",
    ]
    return command, child_environment


def main() -> None:
    """轻量读取 YAML 后替换当前进程，训练程序只能看到最终选择的 GPU。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--training-entry", required=True)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    try:
        command, environment = prepare_launch(args.config_path, args.training_entry, args.timestamp)
    except (ValueError, OSError, yaml.YAMLError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    print(
        f"GPU 可见设备: {environment.get('CUDA_VISIBLE_DEVICES', '继承系统默认')}；"
        f"{command[4]}；{command[-1]}",
        flush=True,
    )
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
