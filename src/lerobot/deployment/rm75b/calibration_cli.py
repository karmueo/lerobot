"""准备手测候选外参与图像复核证据，不应用外参或控制执行器."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import ImageArray

_FASTUMI_ROOT = Path("/home/scl/work/FastUMI_Data")
_CAPTURE_TIMEOUT_S = 3.0


def _write_report(directory: Path, report: dict[str, Any]) -> None:
    """保存可供人工核对的 JSON 记录，拒绝非有限值."""
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    """保留模板字段顺序及待填的 null 值."""
    import yaml

    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_image(path: Path, bgr: ImageArray) -> None:
    """OpenCV 编码输入采用 BGR，写入失败时保留拒绝原因."""
    import cv2

    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"无法保存图像 {path}")


def _image(args: argparse.Namespace, directory: Path) -> int:
    """生成图像分析报告和叠加图，分析拒绝时仍保存证据."""
    from .calibration_vision import CalibrationImageError, analyze_image

    try:
        report, overlay = analyze_image(
            args.image,
            args.camera_calibration,
            fastumi_root=args.fastumi_root,
            marker_size_mm=args.marker_size_mm,
            candidate_path=args.candidate,
            measured_tag_distance_mm=args.measured_tag_distance_mm,
            observed_tcp_px=None if args.tcp_pixel is None else (args.tcp_pixel[0], args.tcp_pixel[1]),
        )
    except CalibrationImageError as exc:
        _write_report(directory, {**exc.report, "status": "rejected", "reason": str(exc)})
        print(str(exc), file=sys.stderr)
        return 1
    _write_image(directory / "overlay.png", overlay)
    _write_report(directory, report)
    return 0


def _capture(config_path: Path, directory: Path) -> None:
    """只占用腕部相机，保存同一帧原图及模型输入像素."""
    import numpy as np

    from .camera import CameraSource
    from .cli import conflicting_processes, deployment_lock
    from .config import load_config
    from .observations import preprocess_rgb

    cfg = load_config(config_path)
    with deployment_lock(cfg.logging.lock_path):
        conflicts = conflicting_processes()
        if conflicts:
            raise RuntimeError(f"发现可能占用腕部相机的进程，请先退出 {conflicts}")
        camera = CameraSource(cfg.camera)
        try:
            deadline = time.monotonic() + _CAPTURE_TIMEOUT_S
            camera.connect()
            while time.monotonic() < deadline:
                frames = camera.read_frames()
                if frames:
                    frame = max(frames, key=lambda item: item.timestamp)
                    age = time.monotonic() - frame.timestamp
                    if 0.0 <= age <= cfg.camera.frame_timeout_s:
                        break
                time.sleep(0.01)
            else:
                raise TimeoutError(f"未在 {_CAPTURE_TIMEOUT_S:g} 秒内取得有效相机帧")
        finally:
            camera.close()

    model_rgb = np.rint(np.moveaxis(preprocess_rgb(frame.rgb), 0, -1) * 255).astype(np.uint8)
    raw_name, model_name = "raw_rgb.png", "model_rgb.png"
    # 图像以 RGB 保存，传给 OpenCV 编码器前交换红蓝通道。
    _write_image(directory / raw_name, frame.rgb[..., ::-1])
    _write_image(directory / model_name, model_rgb[..., ::-1])
    _write_report(
        directory,
        {
            "status": "captured",
            "config": str(config_path.resolve()),
            "device": cfg.camera.video_device,
            "frame_id": frame.frame_id,
            "host_receive_monotonic_s": frame.timestamp,
            "raw_rgb": {"path": raw_name, "shape_hwc": list(frame.rgb.shape)},
            "model_rgb": {"path": model_name, "shape_hwc": list(model_rgb.shape)},
        },
    )


def _parser() -> argparse.ArgumentParser:
    """定义模板、候选求解、图像复核和单帧采集四个独立入口."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    template = modes.add_parser("template", help="生成待填的毫米制手测模板")
    solve = modes.add_parser("solve", help="统计手测值并生成未验收候选")
    solve.add_argument("--measurements", type=Path, required=True)
    image = modes.add_parser("image", help="用双 Tag 图像复核测量和候选")
    image.add_argument("--image", type=Path, required=True)
    image.add_argument("--camera-calibration", type=Path, required=True)
    image.add_argument("--marker-size-mm", type=float, required=True)
    image.add_argument("--candidate", type=Path)
    image.add_argument("--measured-tag-distance-mm", type=float)
    image.add_argument("--tcp-pixel", type=float, nargs=2, metavar=("U", "V"))
    image.add_argument("--fastumi-root", type=Path, default=_FASTUMI_ROOT)
    capture = modes.add_parser("capture", help="只从腕部相机保存一帧")
    capture.add_argument("--config", type=Path, required=True)
    for mode in (template, solve, image, capture):
        mode.add_argument("--output-dir", type=Path, required=True, help="不存在的输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    """为每次运行创建独立证据目录，失败时返回非零退出码."""
    args = _parser().parse_args(argv)
    directory: Path = args.output_dir
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"无法创建新输出目录 {directory}：{exc}", file=sys.stderr)
        return 1
    try:
        if args.mode == "template":
            from .calibration import measurement_template

            _write_yaml(directory / "measurements.yaml", measurement_template())
        elif args.mode == "solve":
            from .calibration import solve_measurements

            report = solve_measurements(args.measurements)
            _write_yaml(directory / "candidate.yaml", report)
            _write_report(directory, report)
        elif args.mode == "image":
            return _image(args, directory)
        else:
            _capture(args.config, directory)
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        reason = "用户中断" if interrupted else str(exc)
        print(reason, file=sys.stderr)
        try:
            _write_report(directory, {"status": "rejected", "mode": args.mode, "reason": reason})
        except (OSError, ValueError) as report_error:
            print(f"无法保存拒绝报告：{report_error}", file=sys.stderr)
        return 130 if interrupted else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
