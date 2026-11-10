"""读取串行部署参数和本地模型、数据及标定资产."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from safetensors import safe_open

from .geometry import validate_transform
from .types import FloatArray

REPO_ROOT = Path(__file__).resolve().parents[4]
FASTUMI_ROOT = Path("/home/scl/work/FastUMI_Data")
CHECKPOINT = (
    Path("outputs/train/dp_handumi_tcp_yaml_2026-11-09_18-24-13") / "checkpoints/best/pretrained_model_ema"
)
DATASET = Path("data/handumi_data/umi-demo-tcp-dp-240_134")
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
IMAGE_SHAPE = (3, 134, 240)
ARM_URDF = FASTUMI_ROOT / "assets/rm_75.urdf"
INPUT_FEATURES = {
    IMAGE_KEY: {"type": "VISUAL", "shape": list(IMAGE_SHAPE)},
    STATE_KEY: {"type": "STATE", "shape": [7]},
}
OUTPUT_FEATURES = {"action": {"type": "ACTION", "shape": [7]}}
NORMALIZATION_MAPPING = {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}


@dataclass(frozen=True)
class PolicyConfig:
    """指定本地检查点、验证集和采样参数."""

    checkpoint: Path = CHECKPOINT
    dataset_root: Path = DATASET
    device: str = "cuda:0"
    num_inference_steps: int | None = None
    startup_timeout_s: float = 120.0
    seed: int = 0


@dataclass(frozen=True)
class ArmConfig:
    """机械臂 TCP/UDP、SDK 和外部运动学资产."""

    ip: str = "192.168.1.18"
    port: int = 8080
    udp_ip: str = "192.168.1.100"
    udp_port: int = 8089
    udp_cycle_ms: int = 5
    sdk_root: Path = Path("/home/scl/work/RM_API2/Python")
    urdf: Path = ARM_URDF
    initial_pose: Path = FASTUMI_ROOT / "ros2_ws/src/fastumi_bringup/config/initial_pose.yaml"


@dataclass(frozen=True)
class GripperConfig:
    """夹爪电机参数及训练开度到机器人开度的标定节点."""

    network_interface: str = "wlP1p1s0"
    domain_id: int = 0
    sides: tuple[str, ...] = ("left", "right")
    discovery_s: float = 2.0
    pos_close_rad: float = 0.02
    pos_open_rad: float = 5.0
    kp: float = 5.0
    kd: float = 0.05
    max_speed_rad_s: float = 6.0
    max_hold_error_rad: float = 0.30
    feedback_timeout_s: float = 0.25
    mapping: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
    server_mode: str = "auto"
    server_vendor_dir: Path = FASTUMI_ROOT / "ros2_ws/install/unitree_gripper/share/unitree_gripper/vendor"
    server_start_timeout_s: float = 12.0


@dataclass(frozen=True)
class CameraConfig:
    """按物理端口选择相机并限制采集模式及队列容量."""

    video_device: str = "/dev/v4l/by-path/platform-3610000.usb-usb-0:4.2:1.0-video-index0"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    frame_timeout_s: float = 0.2
    history_size: int = 16


@dataclass(frozen=True)
class ExecutionConfig:
    """观测频率、关节执行节拍及回合和动作段的稳定条件."""

    control_hz: float = 50.0
    fps: float = 30.0
    n_obs_steps: int = 3
    n_action_steps: int = 8
    velocity_scale: float = 0.25
    sync_slop_s: float = 0.05
    history_tolerance_s: float = 0.015
    input_timeout_s: float = 0.2
    inference_timeout_s: float = 30.0
    settle_s: float = 0.3
    settle_timeout_s: float = 5.0
    start_settle_s: float = 0.5
    start_joint_tolerance_rad: float = 0.035
    start_gripper_min: float = 0.95
    stationary_velocity_rad_s: float = 0.02
    home_timeout_s: float = 120.0
    mode: str = "sequential"


@dataclass(frozen=True)
class SafetyConfig:
    """以米、弧度和秒表示的工作空间、步长和反馈保护."""

    workspace_min: tuple[float, float, float] = (0.05, -0.70, 0.05)
    workspace_max: tuple[float, float, float] = (0.90, 0.70, 1.0)
    max_translation_m: float = 0.02
    max_rotation_rad: float = math.radians(10)
    max_joint_step_rad: float = math.radians(10)
    feedback_timeout_s: float = 0.25
    max_send_gap_s: float = 0.1
    drift_translation_m: float = 0.002
    drift_rotation_rad: float = math.radians(1)


@dataclass(frozen=True)
class LoggingConfig:
    """本机日志目录和执行器进程的独占锁路径."""

    output_dir: Path = Path("outputs/deploy/rm75b")
    lock_path: Path = Path("/tmp/lerobot-rm75b.lock")


@dataclass(frozen=True)
class DeploymentConfig:
    """串行 RM75B 任务的完整运行参数."""

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass(frozen=True)
class DeploymentAssets:
    """已校验的坐标、初始位和用于记录来源的模型、数据及文件摘要."""

    tcp_offset: FloatArray
    initial_joints: FloatArray
    model_config: dict[str, Any]
    metadata: dict[str, Any]
    hashes: dict[str, str]


def _number(value: Any, label: str) -> float:
    """接收有限实数，拒绝 YAML 布尔值和字符串."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} 必须是有限数值")
    return float(value)


def _coerce(value: Any, default: Any, label: str) -> Any:
    """根据字段默认类型解析配置，避免数值和字符串的隐式转换."""
    if isinstance(default, Path):
        if not isinstance(value, (Path, str)) or not str(value).strip():
            raise ValueError(f"{label} 必须是非空路径")
        return (REPO_ROOT / Path(value).expanduser()).resolve()
    if isinstance(default, float):
        return _number(value, label)
    if isinstance(default, tuple):
        if not isinstance(value, (tuple, list)):
            raise ValueError(f"{label} 必须是列表")
        if label == "gripper.mapping":
            return tuple(_number_tuple(pair, 2, label) for pair in value)
        if label == "gripper.sides":
            if any(type(side) is not str for side in value):
                raise ValueError(f"{label} 必须由字符串组成")
            return tuple(value)
        return _number_tuple(value, len(default), label)
    if default is None:
        if value is None or type(value) is int:
            return value
    elif type(value) is type(default) and (not isinstance(value, str) or value.strip()):
        return value
    raise ValueError(f"{label} 类型或值无效: {value!r}")


def _number_tuple(value: Any, length: int, label: str) -> tuple[float, ...]:
    """将固定长度的有限数值序列转为不可变元组."""
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{label} 需要 {length} 个数值")
    return tuple(_number(item, label) for item in value)


def _load_section[Section](defaults: Section, raw: Any, name: str) -> Section:
    """从 YAML 严格构造一节配置，同时解析默认相对路径."""
    if not isinstance(raw, dict):
        raise ValueError(f"{name} 必须是映射")
    values = {key: getattr(defaults, key) for key in vars(defaults)}
    for key in raw.keys() - values.keys():
        raise ValueError(f"未知配置字段: {name}.{key}")
    return type(defaults)(
        **{key: _coerce(raw.get(key, default), default, f"{name}.{key}") for key, default in values.items()}
    )


def validate_config(cfg: DeploymentConfig) -> None:
    """拒绝无效范围、时序及与本模型不兼容的配置."""
    # dataclasses.replace 等直接构造路径也需要相同类型与有限性检查。
    for section_name, defaults in vars(DeploymentConfig()).items():
        values = getattr(cfg, section_name)
        if type(values) is not type(defaults):
            raise ValueError(f"配置节类型无效: {section_name}")
        for key, default in vars(defaults).items():
            _coerce(getattr(values, key), default, f"{section_name}.{key}")
    if re.fullmatch(r"cpu|cuda(?::[0-9]+)?", cfg.policy.device) is None:
        raise ValueError("policy.device 必须是 cpu、cuda 或 cuda:<索引>")
    if cfg.policy.num_inference_steps is not None and not 1 <= cfg.policy.num_inference_steps <= 100:
        raise ValueError("policy.num_inference_steps 必须在 1～100 之间")
    if not 0 <= cfg.policy.seed < 2**32:
        raise ValueError("policy.seed 必须在 0～2**32-1 之间")
    for address in (cfg.arm.ip, cfg.arm.udp_ip):
        ipaddress.IPv4Address(address)
    if not all(1 <= port <= 65535 for port in (cfg.arm.port, cfg.arm.udp_port)):
        raise ValueError("机械臂端口必须在 1～65535 之间")
    if not 5 <= cfg.arm.udp_cycle_ms <= 100 or cfg.arm.udp_cycle_ms % 5:
        raise ValueError("arm.udp_cycle_ms 必须为 5～100 毫秒之间的 5 毫秒倍数")
    gripper = cfg.gripper
    if gripper.server_mode not in ("auto", "external") or not 0 < gripper.server_start_timeout_s <= 60:
        raise ValueError("gripper.server_mode 须为 auto 或 external，服务启动超时须在 (0, 60] 秒内")
    if (
        not 0 <= gripper.domain_id <= 232
        or not gripper.sides
        or len(set(gripper.sides)) != len(gripper.sides)
        or set(gripper.sides) - {"left", "right"}
        or gripper.pos_close_rad >= gripper.pos_open_rad
        or min(gripper.kp, gripper.kd) < 0
    ):
        raise ValueError("夹爪 DDS 域、侧别、电机端点或增益无效")
    mapping = gripper.mapping
    if (
        len(mapping) < 2
        or mapping[0] != (0.0, 0.0)
        or mapping[-1] != (1.0, 1.0)
        or any(a >= c or b >= d for (a, b), (c, d) in zip(mapping[:-1], mapping[1:], strict=True))
    ):
        raise ValueError("gripper.mapping 须从 (0,0) 到 (1,1)，两轴严格递增")
    camera = cfg.camera
    if not Path(camera.video_device).is_absolute() or min(camera.width, camera.height) < 1:
        raise ValueError("相机设备路径和采集尺寸无效")
    execution = cfg.execution
    if execution.mode != "sequential":
        raise ValueError("当前只支持 execution.mode=sequential")
    if (execution.control_hz, execution.fps, execution.n_obs_steps, execution.n_action_steps) != (
        50,
        30,
        3,
        8,
    ):
        raise ValueError("当前契约要求 50 Hz 执行、30 Hz 观测、3 帧输入和 8 步输出")
    if camera.fps != execution.fps or camera.history_size < execution.n_obs_steps:
        raise ValueError("相机帧率或历史容量与观测契约不一致")
    if not 0 < execution.velocity_scale <= 1 or not 0 <= execution.start_gripper_min <= 1:
        raise ValueError("执行限速比例或起始夹爪开度无效")
    if not 0 < execution.history_tolerance_s < 0.5 / execution.fps:
        raise ValueError("历史帧匹配容差须小于半个观测周期")
    if execution.settle_timeout_s <= execution.settle_s or execution.input_timeout_s <= 2 / execution.fps:
        raise ValueError("稳定超时或输入新鲜度上限不足")
    safety = cfg.safety
    if any(a >= b for a, b in zip(safety.workspace_min, safety.workspace_max, strict=True)):
        raise ValueError("工作空间最小坐标须小于最大坐标")
    if safety.max_send_gap_s <= 1 / execution.control_hz:
        raise ValueError("允许的发送间隔必须大于控制周期")
    for section in (cfg.policy, cfg.gripper, cfg.camera, execution, safety):
        for name, value in vars(section).items():
            if (name.endswith(("_s", "_rad_s")) or name.startswith(("max_", "drift_"))) and value <= 0:
                raise ValueError(f"{name} 必须为正数")
    if min(execution.start_joint_tolerance_rad, execution.stationary_velocity_rad_s) <= 0:
        raise ValueError("起点关节容差和静止速度阈值必须为正数")


def load_config(path: str | Path) -> DeploymentConfig:
    """读取 YAML，所有相对路径以 LeRobot 仓库根目录为基准."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = DeploymentConfig()
    if not isinstance(raw, dict):
        raise ValueError("部署配置必须是 YAML 映射")
    known = {item.name for item in fields(defaults)}
    if unknown := raw.keys() - known:
        raise ValueError(f"未知配置节: {unknown}")
    cfg = DeploymentConfig(
        **{name: _load_section(getattr(defaults, name), raw.get(name, {}), name) for name in known}
    )
    validate_config(cfg)
    return cfg


def config_to_dict(cfg: DeploymentConfig) -> dict[str, Any]:
    """将冻结配置转换为可直接写入运行清单的 JSON 内容."""
    return {
        section: {name: str(value) if isinstance(value, Path) else value for name, value in values.items()}
        for section, values in asdict(cfg).items()
    }


def _read_json(path: Path) -> dict[str, Any]:
    """读取文件，要求顶层为 JSON 对象."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层须是 JSON 对象")
    return data


def _matches(data: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    """比较固定模型或数据字段，指出首个不一致处."""
    for key, value in expected.items():
        actual = data.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"{label}.{key}: 期望 {value!r}，得到 {actual!r}")


def _processor_files(root: Path) -> list[Path]:
    """核对固定处理步骤和实际存在的有限统计量，使用 NumPy 读取."""
    specs: dict[str, tuple[tuple[str, ...], dict[str, tuple[int, ...]]]] = {
        "policy_preprocessor": (
            (
                "rename_observations_processor",
                "to_batch_processor",
                "device_processor",
                "normalizer_processor",
            ),
            {
                f"{IMAGE_KEY}.mean": (3, 1, 1),
                f"{IMAGE_KEY}.std": (3, 1, 1),
                f"{STATE_KEY}.min": (7,),
                f"{STATE_KEY}.max": (7,),
            },
        ),
        "policy_postprocessor": (
            ("unnormalizer_processor", "device_processor"),
            {"action.min": (7,), "action.max": (7,)},
        ),
    }
    files: list[Path] = []
    for name, (expected_steps, stats_shapes) in specs.items():
        path = root / f"{name}.json"
        steps = _read_json(path)["steps"]
        if tuple(step["registry_name"] for step in steps) != expected_steps:
            raise ValueError(f"不支持的处理器序列: {path}")
        normalizer = next(step for step in steps if "normalizer_processor" in step["registry_name"])
        normalizer_config = normalizer["config"]
        features = INPUT_FEATURES | OUTPUT_FEATURES if name == "policy_preprocessor" else OUTPUT_FEATURES
        _matches(
            normalizer_config,
            {"features": features, "norm_map": NORMALIZATION_MAPPING},
            f"{name}.normalizer.config",
        )
        if _number(normalizer_config.get("eps"), f"{name}.normalizer.eps") <= 0:
            raise ValueError(f"归一化 eps 必须为正数: {path}")
        if name == "policy_preprocessor":
            _matches(steps[0]["config"], {"rename_map": {}}, f"{name}.rename.config")
        device_step = next(step for step in steps if step["registry_name"] == "device_processor")
        if device_step["config"].get("float_dtype") is not None:
            raise ValueError(f"设备处理器不得改变 FP32 类型: {path}")
        state = normalizer.get("state_file")
        if not isinstance(state, str) or Path(state).name != state or not state.endswith(".safetensors"):
            raise ValueError(f"处理器统计量须为同目录 safetensors 文件: {path}")
        stats_path = root / state
        if not stats_path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"处理器统计量越出检查点目录: {stats_path}")
        with safe_open(stats_path, framework="numpy") as reader:
            stats = {}
            keys = set(reader.keys())
            for key, shape in stats_shapes.items():
                if key not in keys:
                    raise ValueError(f"缺少处理器统计量: {key}")
                tensor = reader.get_tensor(key)
                if tensor.shape != shape or not np.isfinite(tensor).all():
                    raise ValueError(f"处理器统计量形状或值无效: {key}")
                stats[key] = tensor
            if name == "policy_preprocessor" and np.any(stats[f"{IMAGE_KEY}.std"] <= 0):
                raise ValueError(f"图像标准差须为正数: {stats_path}")
            feature = STATE_KEY if name == "policy_preprocessor" else "action"
            if np.any(stats[f"{feature}.min"] > stats[f"{feature}.max"]):
                raise ValueError(f"{feature} 统计最小值大于最大值: {stats_path}")
        files.extend((path, stats_path))
    return files


def _hash(path: Path) -> str:
    """以流式方式生成摘要，避免复制完整模型权重到内存."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_assets(cfg: DeploymentConfig) -> DeploymentAssets:
    """核对固定 EMA 模型、数据坐标契约、URDF 及初始位."""
    validate_config(cfg)
    root = cfg.policy.checkpoint
    model_path = root / "config.json"
    weights = root / "model.safetensors"
    model = _read_json(model_path)
    _matches(
        model,
        {
            "type": "diffusion",
            "n_obs_steps": 3,
            "n_action_steps": 8,
            "horizon": 16,
            "noise_scheduler_type": "DDPM",
            "num_train_timesteps": 100,
            "use_relative_pose": False,
            "use_amp": False,
            "use_peft": False,
            "input_features": INPUT_FEATURES,
            "output_features": OUTPUT_FEATURES,
            "normalization_mapping": NORMALIZATION_MAPPING,
        },
        "config.json",
    )
    if model.get("num_inference_steps") not in (None, 100):
        raise ValueError("模型检查点的默认采样步数必须为 100")
    # safe_open 先验证权重索引及大小，不将约 1 GB 权重读入数组。
    with safe_open(weights, framework="numpy") as reader:
        if not reader.keys():
            raise ValueError("检查点权重为空")
    processors = _processor_files(root)

    metadata_path = cfg.policy.dataset_root / "meta/fastumi.json"
    info_path = cfg.policy.dataset_root / "meta/info.json"
    metadata = _read_json(metadata_path)
    _matches(
        metadata,
        {
            "fps": 30,
            "gripper_mode": "normalized",
            "deployment_robot": "rm75b",
            "deployment_ee_link": "Link7",
            "deployment_tool_offset_convention": "p_ee = deployment_tool_offset @ p_tcp",
            "pose_frame": "episode_start_tcp",
            "end_frame": "tcp",
            "tcp_definition": "gripper_grasp_center",
            "state_layout": "xyz_rotvec_gripper",
            "action_layout": "xyz_rotvec_gripper",
            "action_source": "demonstration_trajectory",
            "state_time_reference": "image_timestamp",
            "action_time_reference": "image_timestamp",
            "position_unit": "m",
            "rotation_unit": "rad",
            "gripper_representation": "normalized_0_1",
            "image_size": [134, 240],
            "image_preprocessing": "center_crop_inter_area",
        },
        "fastumi.json",
    )
    info = _read_json(info_path)
    if info.get("fps") != cfg.execution.fps:
        raise ValueError("info.json.fps 与模型契约不一致")
    features = info["features"]
    for key, shape in ((IMAGE_KEY, [134, 240, 3]), (STATE_KEY, [7]), ("action", [7])):
        if features.get(key, {}).get("shape") != shape:
            raise ValueError(f"数据集 {key} 形状与模型契约不一致")
    offset = validate_transform(metadata["deployment_tool_offset"])
    calibration = metadata.get("robot_tcp_calibration", {}).get("arms", {}).get("right", {})
    if calibration.get("ee_link") != "Link7" or not np.allclose(
        offset, validate_transform(calibration["ee_to_tcp"]), rtol=0, atol=1e-8
    ):
        raise ValueError("数据集 TCP 矩阵与右侧 Link7 标定不一致")
    urdf_hash = _hash(cfg.arm.urdf)
    if metadata.get("urdf_sha256") != urdf_hash:
        raise ValueError("URDF SHA-256 与训练数据元信息不一致")

    initial = yaml.safe_load(cfg.arm.initial_pose.read_text(encoding="utf-8"))
    initial_joints = np.deg2rad(_number_tuple(initial["initial_joint_positions_deg"], 7, "初始关节位"))
    extra = [root / name for name in ("split_manifest.json", "train_config.json", "evaluation_protocol.json")]
    paths = [model_path, weights, *processors, metadata_path, info_path, cfg.arm.initial_pose]
    hashes = {str(path): _hash(path) for path in [*paths, *(file for file in extra if file.is_file())]}
    hashes[str(cfg.arm.urdf)] = urdf_hash
    return DeploymentAssets(offset, initial_joints, model, metadata, hashes)
