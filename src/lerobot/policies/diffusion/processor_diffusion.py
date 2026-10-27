#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from lerobot.processor import (
    AbsolutePoseProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RelativePoseProcessorStep,
    make_default_policy_processor_steps,
    make_default_pre_post_processors,
    make_policy_processor_pipelines,
)

from .configuration_diffusion import DiffusionConfig


def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Normalizing the input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving the data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the diffusion policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    When ``config.use_relative_pose`` is enabled, an SE(3) relative-pose step is inserted before
    normalization and its inverse after unnormalization (UMI / FastUMI style):
    ``raw -> relative pose (rot6d) -> normalize -> model -> unnormalize -> absolute pose``.
    The provided ``dataset_stats`` must then describe the relative space (see
    ``compute_relative_pose_stats``).

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    if not config.use_relative_pose:
        return make_default_pre_post_processors(config, dataset_stats)

    steps = make_default_policy_processor_steps(config, dataset_stats)
    relative_step = RelativePoseProcessorStep(
        enabled=True,
        state_pose_starts=list(config.relative_pose_state_starts),
        action_pose_starts=config.relative_pose_action_starts,
        n_obs_steps=config.n_obs_steps,
    )
    absolute_step = AbsolutePoseProcessorStep(
        enabled=True,
        state_pose_starts=list(config.relative_pose_state_starts),
        action_pose_starts=config.relative_pose_action_starts,
        relative_step=relative_step,
    )
    # 相对转换必须在设备迁移之后（与归一化同设备）、归一化之前；绝对还原在反归一化之后
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        relative_step,
        steps.normalize,
    ]
    output_steps: list[ProcessorStep] = [steps.unnormalize, absolute_step, steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
