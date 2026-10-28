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
"""双网络模型包的完整动作块推理，保留原始 scheduler 与位姿处理."""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import _diffusers_available, require_package

from .package import load_config, load_manifest, load_processors, read_json

if TYPE_CHECKING or _diffusers_available:
    from diffusers import DDIMScheduler, DDPMScheduler


def synchronize(device: str) -> None:
    """完整调用计时前后同步 GPU，避免只测到异步提交耗时."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def encoder_inputs(batch: dict, cfg) -> dict:
    """保持与原生离线 predict_action_chunk 相同的相机堆叠顺序."""
    inputs = {OBS_STATE: batch[OBS_STATE]}
    if cfg.image_features:
        inputs[OBS_IMAGES] = torch.stack([batch[key] for key in cfg.image_features], dim=-4)
    if cfg.env_state_feature:
        inputs[OBS_ENV_STATE] = batch[OBS_ENV_STATE]
    return inputs


class ExportedDiffusion:
    """模型包运行器；session 仅提供网络，调度和随机数在统一设备执行."""

    def __init__(self, root: Path, session, device: str, manifest: dict | None = None):
        """加载模型包中的处理器与调度器，不需要原始 PyTorch 权重."""
        require_package("diffusers", extra="diffusion")
        self.manifest = manifest if manifest is not None else load_manifest(root)
        self.config = load_config(root / "assets", device)
        self.pre, self.post = load_processors(root / "assets", self.config, device)
        scheduler_cls = {"DDPM": DDPMScheduler, "DDIM": DDIMScheduler}[self.config.noise_scheduler_type]
        self.scheduler = scheduler_cls.from_config(read_json(root / "assets" / "scheduler.json"))
        self.session, self.device = session, device

    def reset(self) -> None:
        """切换 episode 或独立窗口时清除处理器历史及共享 anchor."""
        self.pre.reset()
        self.post.reset()

    @torch.inference_mode()
    def sample(
        self, inputs: dict, seed: int, timings: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """每次重新初始化局部 RNG，使初始噪声及每步方差噪声可复现."""
        started = time.perf_counter()
        cond = self.session.run("encoder", inputs)
        if timings is not None:
            synchronize(self.device)
            timings["encoder_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        sample = torch.randn(
            (1, self.config.horizon, self.config.model_action_dim),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        self.scheduler.set_timesteps(self.manifest["num_inference_steps"])
        for timestep in self.scheduler.timesteps:
            prediction = self.session.run(
                "denoiser",
                {
                    "sample": sample,
                    "timestep": torch.full((1,), timestep, device=self.device, dtype=torch.long),
                    "global_cond": cond,
                },
            )
            sample = self.scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
        start, end = self.manifest["action_slice"]
        if timings is not None:
            synchronize(self.device)
            timings["sampling_ms"] = (time.perf_counter() - started) * 1000
        return sample[:, start:end], cond

    @torch.inference_mode()
    def predict(self, raw: dict, seed: int = 0, profile: bool = False):
        """从已解码原始观测产生绝对动作；可单独采集阶段耗时."""
        timings = {}

        def mark(name: str, started: float) -> float:
            if profile:
                synchronize(self.device)
                timings[name] = (time.perf_counter() - started) * 1000
            return time.perf_counter()

        start = time.perf_counter()
        batch = self.pre(dict(raw))
        inputs = encoder_inputs(batch, self.config)
        start = mark("preprocess_ms", start)
        actions, cond = self.sample(inputs, seed, timings if profile else None)
        start = mark("encoder_and_sampling_ms", start)
        result = self.post(actions.clone())
        mark("postprocess_ms", start)
        return result, actions, cond, timings
