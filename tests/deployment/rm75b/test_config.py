"""覆盖 YAML 误配置及检查点归一化资产拒绝条件."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from lerobot.deployment.rm75b.config import (
    IMAGE_KEY,
    INPUT_FEATURES,
    NORMALIZATION_MAPPING,
    OUTPUT_FEATURES,
    REPO_ROOT,
    STATE_KEY,
    DeploymentConfig,
    _processor_files,
    config_to_dict,
    load_config,
    validate_config,
)


def test_load_config_defaults_are_root_relative(tmp_path: Path) -> None:
    """配置文件所在目录不影响模型路径解释."""
    path = tmp_path / "deploy.yaml"
    path.write_text("policy:\n  checkpoint: outputs/test/model\n  device: cpu\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.policy.checkpoint == REPO_ROOT / "outputs/test/model"
    assert cfg.execution.control_hz == 50
    assert cfg.execution.n_obs_steps == 3
    assert cfg.gripper.mapping == ((0, 0), (1, 1))
    assert json.loads(json.dumps(config_to_dict(cfg)))["policy"]["checkpoint"] == str(cfg.policy.checkpoint)


@pytest.mark.parametrize(
    "yaml_text, expected",
    [
        ("unsupported: {}", "未知配置节"),
        ("policy:\n  fp16: true", "未知配置字段"),
        ("policy:\n  num_inference_steps: true", "类型或值无效"),
        ("policy:\n  num_inference_steps: 101", "1～100"),
        ("execution:\n  mode: pipelined", "只支持"),
        ("execution:\n  history_tolerance_s: 0.02", "半个观测周期"),
        ("execution:\n  n_obs_steps: 2", "3 帧输入"),
        ("gripper:\n  mapping: [[0,0],[0.5,0.2],[0.7,0.1],[1,1]]", "严格递增"),
        ("safety:\n  max_translation_m: .nan", "有限数值"),
        ("safety:\n  workspace_min: [0.91, 0.0, 0.05]", "工作空间"),
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, yaml_text: str, expected: str) -> None:
    """模型契约、数值类型和安全范围无效时停止配置加载."""
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        load_config(path)


def test_python_override_is_also_validated() -> None:
    """替换数据类绕过 YAML 时仍拒绝无限大的安全阈值."""
    cfg = DeploymentConfig()
    invalid = replace(cfg, safety=replace(cfg.safety, drift_rotation_rad=float("inf")))
    with pytest.raises(ValueError, match="有限数值"):
        validate_config(invalid)


@pytest.fixture
def processors(tmp_path: Path) -> Path:
    """创建有实际统计量的最小模型处理器资产."""
    pre = {
        "steps": [
            {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
            {"registry_name": "to_batch_processor", "config": {}},
            {"registry_name": "device_processor", "config": {"device": "cuda", "float_dtype": None}},
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "features": INPUT_FEATURES | OUTPUT_FEATURES,
                    "norm_map": NORMALIZATION_MAPPING,
                    "eps": 1e-8,
                },
                "state_file": "pre.safetensors",
            },
        ],
    }
    post = {
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {"features": OUTPUT_FEATURES, "norm_map": NORMALIZATION_MAPPING, "eps": 1e-8},
                "state_file": "post.safetensors",
            },
            {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": None}},
        ],
    }
    for name, pipeline in (("pre", pre), ("post", post)):
        (tmp_path / f"policy_{name}processor.json").write_text(json.dumps(pipeline), encoding="utf-8")
    save_file(
        {
            f"{IMAGE_KEY}.mean": np.zeros((3, 1, 1), dtype=np.float32),
            f"{IMAGE_KEY}.std": np.ones((3, 1, 1), dtype=np.float32),
            f"{STATE_KEY}.min": np.zeros(7, dtype=np.float32),
            f"{STATE_KEY}.max": np.ones(7, dtype=np.float32),
        },
        tmp_path / "pre.safetensors",
    )
    save_file(
        {"action.min": np.zeros(7, dtype=np.float32), "action.max": np.ones(7, dtype=np.float32)},
        tmp_path / "post.safetensors",
    )
    return tmp_path


def test_reject_processor_mode_mismatch(processors: Path) -> None:
    """读取保存的归一化模式，避免静默接受错误统计量应用."""
    assert len(_processor_files(processors)) == 4
    path = processors / "policy_postprocessor.json"
    pipeline = json.loads(path.read_text(encoding="utf-8"))
    pipeline["steps"][0]["config"]["norm_map"]["ACTION"] = "IDENTITY"
    path.write_text(json.dumps(pipeline), encoding="utf-8")
    with pytest.raises(ValueError, match="norm_map"):
        _processor_files(processors)


def test_reject_swapped_statistics(processors: Path) -> None:
    """反归一化范围颠倒时失败."""
    save_file(
        {"action.min": np.ones(7, dtype=np.float32), "action.max": np.zeros(7, dtype=np.float32)},
        processors / "post.safetensors",
    )
    with pytest.raises(ValueError, match="最小值大于最大值"):
        _processor_files(processors)
