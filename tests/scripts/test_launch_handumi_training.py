"""验证 YAML 选卡在 CUDA 探测前生效，且检查点保存完整的 GPU 配置。"""

import subprocess
from pathlib import Path

import draccus
import pytest
import yaml

from lerobot.configs.train import TrainPipelineConfig
from scripts import launch_handumi_training as launcher


@pytest.mark.parametrize("value", [[], [0, 0], [-1], [0.5], ["0"], [True], "0,1", 0])
def test_invalid_gpu_ids_rejected_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    """输入错误不能触发 CUDA 探测或训练启动。"""
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump({"gpu_ids": value}))

    def unexpected_probe(*args: object) -> None:
        pytest.fail("Invalid gpu_ids must fail before probing CUDA")

    monkeypatch.setattr(launcher, "check_gpu_availability", unexpected_probe)
    with pytest.raises(ValueError, match="gpu_ids"):
        launcher.prepare_launch(config, "/tmp/train", "timestamp")


@pytest.mark.parametrize("count", [0, 1])
def test_unavailable_gpu_detected_in_selected_environment(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    """探测子进程必须已收到最终 mask，不得沿用父进程 CUDA 缓存。"""
    selected = {"CUDA_VISIBLE_DEVICES": "0,999", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    def probe(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert command[1:3] == ["-c", "import torch; print(torch.cuda.device_count())"]
        assert kwargs["env"] == selected
        return subprocess.CompletedProcess(command, 0, stdout=f"{count}\n", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", probe)
    with pytest.raises(ValueError, match="不存在或不可用"):
        launcher.check_gpu_availability([0, 999], selected)


def test_gpu_config_roundtrip(tmp_path: Path) -> None:
    """gpu_ids 是正式训练配置字段，YAML 与 train_config.json 均可往返。"""
    config = tmp_path / "train.yaml"
    config.write_text("dataset:\n  repo_id: handumi/test\ngpu_ids: [1, 0]\n")
    loaded = draccus.parse(TrainPipelineConfig, config_path=config, args=[])
    assert loaded.gpu_ids == [1, 0]
    loaded.save_pretrained(tmp_path)
    restored = TrainPipelineConfig.from_pretrained(tmp_path, local_files_only=True)
    assert restored.gpu_ids == [1, 0]


def test_independent_output_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """两份配置在同一时间启动仍由任务名隔离，重复启动再由纳秒时间戳隔离。"""
    monkeypatch.setattr(launcher, "check_gpu_availability", lambda ids, environment: None)
    outputs = []
    for name, timestamp in (
        ("episode", "time_000000001"),
        ("current", "time_000000001"),
        ("episode", "time_000000002"),
    ):
        config = tmp_path / f"{name}.yaml"
        config.write_text(yaml.safe_dump({"gpu_ids": [0], "job_name": name}))
        command, environment = launcher.prepare_launch(config, "/tmp/train", timestamp, {"OTHER": "kept"})
        assert environment["OTHER"] == "kept"
        assert "--standalone" in command
        outputs.append(command[-1])
    assert len(set(outputs)) == 3


def test_legacy_config_keeps_mask_and_two_processes(tmp_path: Path) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text("job_name: legacy\n")
    inherited = {"CUDA_VISIBLE_DEVICES": "1,0", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    command, environment = launcher.prepare_launch(config, "/tmp/train", "now", inherited)
    assert environment == inherited
    assert "--nproc-per-node=2" in command
    assert inherited == environment


def test_explicit_gpu_rejects_cpu_policy(tmp_path: Path) -> None:
    config = tmp_path / "cpu.yaml"
    config.write_text("gpu_ids: [0]\npolicy:\n  device: cpu\n")
    with pytest.raises(ValueError, match="cpu"):
        launcher.prepare_launch(config, "/tmp/train", "now")
