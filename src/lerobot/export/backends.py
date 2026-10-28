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
"""ONNX CPU 和 TensorRT CUDA 执行器，统一以 PyTorch 张量交接调度器."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lerobot.utils.import_utils import _onnxruntime_available, _tensorrt_available, require_package

if TYPE_CHECKING or _onnxruntime_available:
    import onnxruntime as ort
if TYPE_CHECKING or _tensorrt_available:
    import tensorrt as trt


def gpu_identity(device: str) -> dict:
    """记录实际 CUDA 设备及计算能力，用于拒绝不兼容 engine."""
    props = torch.cuda.get_device_properties(torch.device(device))
    return {"name": props.name, "compute_capability": [props.major, props.minor]}


def require_tensorrt() -> None:
    """此实现使用 10.13.3 的弱类型精度 API，不冒充兼容 TRT 11."""
    require_package("tensorrt-cu12", extra="export-tensorrt", import_name="tensorrt")
    if not trt.__version__.startswith("10.13.3."):
        raise RuntimeError(f"Expected TensorRT 10.13.3.*, found {trt.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT requires a CUDA GPU")


def build_engine(onnx_path: Path, output: Path, precision: str, device: str) -> dict:
    """用 Python API 构建固定形状 engine，保留解析错误且不写入失败产物."""
    require_tensorrt()
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")
    with torch.cuda.device(device):
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT ONNX parser failed:\n{errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
        config.clear_flag(trt.BuilderFlag.TF32)
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT build failed: {onnx_path.name} ({precision})")
        output.write_bytes(bytes(serialized))
    return {"tensorrt": trt.__version__, "gpu": gpu_identity(device), "precision": precision}


def check_inputs(inputs: dict, spec: dict) -> None:
    """在运行时拒绝 batch 或 dtype 不符的输入，避免固定图意外广播."""
    if set(inputs) != set(spec):
        raise ValueError(f"Input names mismatch: expected {list(spec)}, got {list(inputs)}")
    for name, tensor in inputs.items():
        expected = spec[name]
        if list(tensor.shape) != expected["shape"] or str(tensor.dtype) != expected["dtype"]:
            raise ValueError(f"Input shape/dtype mismatch for {name}: {tensor.shape}, {tensor.dtype}")


class TorchSession:
    """执行导出包装的 eager 模型，验证包装是否改变了原生行为."""

    def __init__(self, modules: dict):
        """接收策略适配器创建的网络与命名接口."""
        self.modules = modules

    def run(self, stage: str, inputs: dict) -> torch.Tensor:
        """参数顺序与导出图的命名输入一致."""
        module, _, names, _ = self.modules[stage]
        return module(*(inputs[name] for name in names))


class OnnxSession:
    """ORT 仅运行 CPU 图；调度器和随机数仍在参考设备上执行."""

    def __init__(self, root: Path, manifest: dict, device: str):
        """加载模型包中的两个 CPU 会话."""
        require_package("onnxruntime", extra="export")
        self.specs = manifest["stages"]
        self.device = device
        options = ort.SessionOptions()
        # 小 batch 网络避免使用整机数百线程导致调度开销失控。
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.sessions = {
            name: ort.InferenceSession(
                str(root / spec["onnx"]), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for name, spec in self.specs.items()
        }

    def run(self, stage: str, inputs: dict) -> torch.Tensor:
        """CPU/GPU 搬运显式计入调用开销，不将 CPU ORT 当作 GPU 基准."""
        check_inputs(inputs, self.specs[stage]["inputs"])
        result = self.sessions[stage].run(None, {k: v.detach().cpu().numpy() for k, v in inputs.items()})[0]
        return torch.from_numpy(result).to(self.device)


class TensorRTSession:
    """保留 runtime/context 生命周期，使用 torch 管理的设备内存和 CUDA stream."""

    def __init__(self, root: Path, manifest: dict, device: str, precision: str):
        """在指定 GPU 上创建与构建信息一致的执行上下文."""
        require_tensorrt()
        self.device = torch.device(device)
        self.specs = manifest["stages"]
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engines = {}
        self.contexts = {}
        with torch.cuda.device(self.device):
            # TensorRT 在默认流上会额外同步；独立流通过事件与 torch 调度器交接。
            self.stream = torch.cuda.Stream(device=self.device)
            for name, spec in self.specs.items():
                artifact = spec["engines"][precision]
                if artifact["gpu"] != gpu_identity(device) or artifact["tensorrt"] != trt.__version__:
                    raise RuntimeError("TensorRT engine device/version mismatch; rebuild on this device")
                engine = self.runtime.deserialize_cuda_engine((root / artifact["path"]).read_bytes())
                if engine is None:
                    raise RuntimeError(f"Failed to deserialize TensorRT engine: {name}")
                context = engine.create_execution_context()
                if context is None:
                    raise RuntimeError(f"Failed to create TensorRT context: {name}")
                self.engines[name] = engine
                self.contexts[name] = context

    def run(self, stage: str, inputs: dict) -> torch.Tensor:
        """一次异步推理；输入引用与输出 tensor 均覆盖本次 CUDA 使用期."""
        spec = self.specs[stage]
        check_inputs(inputs, spec["inputs"])
        engine, context = self.engines[stage], self.contexts[stage]
        with torch.cuda.device(self.device):
            caller = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                buffers = {name: value.to(self.device).contiguous() for name, value in inputs.items()}
                output_name = spec["output"]["name"]
                output = torch.empty(
                    tuple(engine.get_tensor_shape(output_name)), device=self.device, dtype=torch.float32
                )
                buffers[output_name] = output
                for name, value in buffers.items():
                    expected = trt.int64 if value.dtype == torch.int64 else trt.float32
                    if engine.get_tensor_dtype(name) != expected:
                        raise RuntimeError(f"Unexpected TensorRT I/O dtype for {name}")
                    if not context.set_tensor_address(name, value.data_ptr()):
                        raise RuntimeError(f"Could not bind TensorRT input/output: {name}")
                    value.record_stream(self.stream)
                if not context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError(f"TensorRT execution failed: {stage}")
            caller.wait_stream(self.stream)
            output.record_stream(caller)
        return output
