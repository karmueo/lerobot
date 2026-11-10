"""运行清单、JSONL 事件与回合摘要。磁盘写入不占用 50 Hz 控制循环."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import queue
import shutil
import subprocess
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np

from .config import config_to_dict

if TYPE_CHECKING:
    from .config import DeploymentAssets, DeploymentConfig
    from .types import ActionChunk, ObservationWindow

logger = logging.getLogger(__name__)


def json_value(value: Any) -> Any:
    """复制数组和反馈对象，避免日志线程读取之后变化的数据."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def _dependency_versions() -> dict[str, str | None]:
    """读取已安装包元信息，检查模式无需导入 Torch 或设备 SDK."""
    packages = (
        "lerobot",
        "torch",
        "torchvision",
        "numpy",
        "diffusers",
        "safetensors",
        "opencv-python-headless",
        "opencv-python",
        "pupil-labs-uvc",
        "cyclonedds",
        "unitree-sdk2py",
        "Robotic-Arm",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _source_revision() -> dict[str, Any]:
    """记录 Git 基线与部署模块摘要，含未提交的实际运行代码."""
    module = Path(__file__).resolve().parent
    revision: str | None = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=module,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        logger.debug("无法读取运行代码的 Git 版本", exc_info=True)
    return {
        "git_head": revision,
        "module_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(module.glob("*.py"))
        },
    }


def build_manifest(cfg: DeploymentConfig, assets: DeploymentAssets, mode: str) -> dict[str, Any]:
    """构造可复核的运行清单，记录去噪步数的实际生效值."""
    model = assets.model_config
    return {
        "mode": mode,
        "created_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": config_to_dict(cfg),
        "asset_sha256": assets.hashes,
        "source": _source_revision(),
        "dependencies": _dependency_versions(),
        "policy": {
            "dtype": "float32",
            "scheduler": model["noise_scheduler_type"],
            "num_inference_steps": (
                cfg.policy.num_inference_steps
                or model.get("num_inference_steps")
                or model["num_train_timesteps"]
            ),
            "n_obs_steps": model["n_obs_steps"],
            "n_action_steps": model["n_action_steps"],
        },
        "clock": "host_monotonic_receive_seconds",
    }


@dataclass(frozen=True)
class _InferenceRecord:
    """将已快照的元数据和可选原始帧交给日志线程."""

    name: str
    metadata: dict[str, Any]
    jpegs: tuple[bytes, ...] = ()


class RunLog:
    """有界日志队列记录运行证据，写入失败后通知控制层停止."""

    def __init__(self, output_dir: Path, manifest: dict[str, Any], *, save_inference: bool = False) -> None:
        """为本次运行创建目录并先持久化配置与文件指纹."""
        stamp = datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y_%m%d__%H%M_%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        for sequence in count(1):
            # 独占创建目录，避免同一秒内的并发会话覆盖彼此的记录。
            suffix = "" if sequence == 1 else f"_{sequence:02d}"
            self.directory = output_dir / f"{stamp}{suffix}-{manifest['mode']}"
            try:
                self.directory.mkdir()
            except FileExistsError:
                continue
            break
        self._save_inference = save_inference
        if save_inference:
            (self.directory / "inference_inputs").mkdir()
            (self.directory / "inference_outputs").mkdir()
        manifest = {**manifest, "save_inference": save_inference}
        (self.directory / "manifest.json").write_text(
            json.dumps(json_value(manifest), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._queue: queue.Queue[dict[str, Any] | _InferenceRecord | None] = queue.Queue(maxsize=2048)
        # 除总队列上限外，单独限制等待写盘的原始图像窗口。
        self._image_slots = threading.BoundedSemaphore(8)
        self._error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._consume, name="rm75b-log", daemon=True)
        self._thread.start()

    def _consume(self) -> None:
        """按控制线程提交顺序持久化事件并汇总回合."""
        counts: Counter[str] = Counter()
        episodes: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        faults: list[str] = []
        try:
            with (self.directory / "events.jsonl").open("w", encoding="utf-8") as stream:
                while (record := self._queue.get()) is not None:
                    if isinstance(record, _InferenceRecord):
                        self._write_inference_record(record)
                        continue
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    event = record["event"]
                    counts[event] += 1
                    if event == "episode_summary":
                        episodes.append(record)
                    elif event == "fault":
                        faults.append(record["reason"])
                    elif event == "closed":
                        final = record
            summary = {
                "event_counts": dict(counts),
                "episodes": episodes,
                "faults": faults,
                "shutdown": final,
            }
            (self.directory / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            self._error = exc

    def write(self, event: str, **values: Any) -> None:
        """提交快照，队列满或磁盘故障时显式失败."""
        record = {"event": event, "monotonic": time.monotonic(), **values}
        self._enqueue(json_value(record))

    def _enqueue(self, record: dict[str, Any] | _InferenceRecord) -> None:
        """非阻塞入队，磁盘失败或积压时通知控制层."""
        if self._error is not None:
            raise RuntimeError("写入运行日志失败") from self._error
        if self._closed:
            raise RuntimeError("运行日志已经关闭")
        try:
            self._queue.put_nowait(record)
        except queue.Full as exc:
            raise RuntimeError("运行日志队列已满") from exc

    def write_inference_input(self, window: ObservationWindow) -> None:
        """保存已提交窗口的原始帧和同序的观测元数据."""
        archive = window.archive
        if not self._save_inference or archive is None or not archive.jpegs:
            raise RuntimeError("推理输入存档未启用或缺少原始观测")
        if not self._image_slots.acquire(blocking=False):
            raise RuntimeError("推理原始图像存档队列已满")
        try:
            name = f"episode_{window.episode_id:06d}_request_{window.request_id:06d}"
            metadata = {
                "episode_id": window.episode_id,
                "request_id": window.request_id,
                "frame_ids": window.frame_ids,
                "timestamps": window.timestamps,
                "images": [f"{index:02d}.jpg" for index in range(len(archive.jpegs))],
                "joints_rad": archive.joints,
                "states": window.states,
                "anchor": window.anchor,
            }
            self._enqueue(_InferenceRecord(name, json_value(metadata), archive.jpegs))
        except BaseException:
            self._image_slots.release()
            raise

    def write_inference_output(self, result: ActionChunk, discarded: bool) -> None:
        """保存取回的模型输出，标记已作废的结果."""
        if not self._save_inference:
            raise RuntimeError("推理输出存档未启用")
        name = f"episode_{result.episode_id:06d}_request_{result.request_id:06d}"
        self._enqueue(_InferenceRecord(name, {**json_value(result), "discarded": discarded}))

    def _write_inference_record(self, record: _InferenceRecord) -> None:
        """只在日志线程写盘，完整记录写好后才发布到正式路径."""
        if record.jpegs:
            target = self.directory / "inference_inputs" / record.name
            staging = target.with_name(f".{target.name}.tmp")
            try:
                staging.mkdir()
                for filename, jpeg in zip(record.metadata["images"], record.jpegs, strict=True):
                    (staging / filename).write_bytes(jpeg)
                (staging / "observation.json").write_text(
                    json.dumps(record.metadata, ensure_ascii=False, allow_nan=False), encoding="utf-8"
                )
                staging.rename(target)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
                self._image_slots.release()
        else:
            target = self.directory / "inference_outputs" / f"{record.name}.json"
            staging = target.with_name(f".{target.name}.tmp")
            try:
                staging.write_text(
                    json.dumps(record.metadata, ensure_ascii=False, allow_nan=False), encoding="utf-8"
                )
                staging.rename(target)
            finally:
                staging.unlink(missing_ok=True)

    def close(self) -> None:
        """控制资源释放后排空日志，失败时返回明确错误."""
        if self._closed:
            return
        self._closed = True
        if self._error is None:
            try:
                self._queue.put(None, timeout=2.0)
            except queue.Full as exc:
                raise RuntimeError("关闭运行日志时队列无法排空") from exc
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("运行日志写入线程关闭超时")
        if self._error is not None:
            raise RuntimeError("写入运行日志失败") from self._error

    def __enter__(self) -> RunLog:
        """返回当前会话日志."""
        return self

    def __exit__(self, *exception: Any) -> None:
        """保留日志写入错误，不把不完整的记录报告为通过."""
        self.close()
