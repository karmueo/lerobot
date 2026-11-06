#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + trackio extras)

Launch with torchrun for distributed runs; every parallelism/acceleration knob lives on the
config (`--parallelism.*`, `--accelerator.*`) so a run is reproducible from its
train_config.json alone:

```bash
torchrun --nproc-per-node=8 $(which lerobot-train) \
    --dataset.repo_id=... --policy.type=act \
    --parallelism.dp_shard=8 --accelerator.mixed_precision=bf16
```
"""

import dataclasses
import logging
import math
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from lerobot.common.eval_protocol import (
    EVAL_PROTOCOL,
    SPLIT_MANIFEST,
    build_eval_protocol,
    validate_eval_protocol,
)
from lerobot.common.trackio_utils import TrackioLogger
from lerobot.common.train_utils import (
    commit_best_and_last_checkpoint,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_best_checkpoint_state,
    load_training_metadata,
    prepare_best_and_last_checkpoint,
    publish_trained_model,
    push_checkpoint_to_hub,
    restore_best_checkpoint_link,
    resume_after_prepare,
    resume_before_prepare,
    save_checkpoint,
    should_save_checkpoint,
    update_last_checkpoint,
)
from lerobot.configs import EvalConfig, JobConfig, parser
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import CheckpointFormat, TrainPipelineConfig
from lerobot.datasets import (
    EpisodeAwareSampler,
    LeRobotDataset,
    StreamingLeRobotDataset,
    compute_sampler_state,
)
from lerobot.datasets.compute_stats import compute_relative_pose_stats
from lerobot.datasets.factory import build_split_manifest, make_train_eval_datasets
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.distributed import (
    ParallelDims,
    finalize_sharded_policy,
    is_main_process,
    make_accelerator,
    set_fsdp_wrap_modules,
)
from lerobot.envs import EnvConfig, close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.ema import EMA
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.processor import ImageAugmentationProcessorStep
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import (
    ACTION,
    BEST_CHECKPOINT_LINK,
    CHECKPOINTS_DIR,
    OBS_STATE,
    PRETRAINED_MODEL_DIR,
    PRETRAINED_MODEL_EMA_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.import_utils import (
    _accelerate_available,
    _peft_available,
    register_third_party_plugins,
    require_package,
)
from lerobot.utils.io_utils import write_json
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import seeded_context, set_seed
from lerobot.utils.sample_weighting import SampleWeighter
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

if TYPE_CHECKING or _peft_available:
    from peft import PeftModel
else:
    PeftModel = None

if TYPE_CHECKING or _accelerate_available:
    from accelerate.utils import send_to_device

from .lerobot_eval import eval_policy_all


@contextmanager
def _make_eval_envs(env_cfg: EnvConfig, eval_cfg: EvalConfig) -> Iterator[dict[str, dict[int, Any]]]:
    """Create evaluation environments for one run and always dispose of them."""
    envs = make_env(
        env_cfg,
        n_envs=eval_cfg.batch_size,
        use_async_envs=eval_cfg.use_async_envs,
    )
    try:
        yield envs
    finally:
        close_envs(envs)


def _make_image_augmentation(
    cfg: TrainPipelineConfig,
    dataset: LeRobotDataset | MultiLeRobotDataset,
    device: torch.device,
    process_index: int,
) -> ImageAugmentationProcessorStep | None:
    """Build the step that augments training batches on the device, or `None` on the dataloader backend.

    Applied to training batches only, before the policy preprocessor; held-out evaluation batches never
    pass through it and the policy's own preprocessor is not modified. The step is called on the batch
    dict directly (`step.observation(batch)`) rather than through a `PolicyProcessorPipeline`: the
    pipeline's batch-to-transition conversion keeps `observation.*` keys and a fixed list of others only,
    which would drop un-prefixed camera keys (such as `image`) and custom columns before `rename_map`
    gets to see them.
    """
    image_transforms = cfg.dataset.image_transforms
    if not (image_transforms.enable and image_transforms.backend == "gpu"):
        return None
    if dataset.image_transforms is not None:
        raise ValueError(
            "image_transforms.backend='gpu' augments on the device, but the dataset already carries "
            "image_transforms that run in the DataLoader workers. Use one or the other."
        )
    image_keys = [key for key in dataset.meta.camera_keys if key not in dataset.meta.depth_keys]
    return ImageAugmentationProcessorStep(
        config=image_transforms,
        image_keys=image_keys,
        device=str(device),
        chunk_size=image_transforms.gpu_chunk_size,
        compile_model=image_transforms.gpu_compile,
        seed=None if cfg.seed is None else cfg.seed + process_index,
    )


def _preprocess_dataset_batch(
    batch: dict[str, Any],
    camera_keys: list[str],
    rename_map: dict[str, str],
    preprocessor: Any,
) -> Any:
    """Prepare a raw dataset batch identically for training and held-out evaluation."""
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    batch = rename_batch_keys(batch, rename_map)
    return preprocessor(batch)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler: LRScheduler | None = None,
    lock: AbstractContextManager[Any] | None = None,
    sample_weighter: SampleWeighter | None = None,
    ema: EMA | None = None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically, and — under
    gradient accumulation — suppresses gradient sync on non-final micro-batches and rescales the loss.

    Under `mixed_precision=fp16` accelerate owns a `GradScaler`: it scales the loss in `backward()`,
    unscales before clipping, and skips the optimizer step whenever a gradient overflowed. State that
    tracks applied updates (the policy's `update()` and the EMA) is gated on the
    step having actually landed; the scheduler is not, because it advances per micro-batch by design.

    Args:
        train_metrics (MetricsTracker): A MetricsTracker instance to record training statistics.
            callers that do not want one simply omit its meter.
        policy (PreTrainedPolicy): The policy model to be trained (as returned by `accelerator.prepare`).
        batch (Any): A batch of training data.
        optimizer (Optimizer): The optimizer used to update the policy's parameters.
        grad_clip_norm (float): The maximum norm for gradient clipping (no clipping when <= 0).
        accelerator (Accelerator): The Accelerator instance for distributed training and mixed precision.
        lr_scheduler (LRScheduler | None, optional): An optional learning rate scheduler, stepped once
            per micro-batch. Defaults to None.
        lock (AbstractContextManager | None, optional): An optional lock (entered around the optimizer
            step) for thread-safe optimizer updates.
            Defaults to None.
        sample_weighter (SampleWeighter | None, optional): Optional SampleWeighter instance for
            per-sample loss weighting. Defaults to None.
        ema (EMA | None, optional): The EMA of the policy weights, stepped after every applied
            optimizer update. Defaults to None.

    Returns:
        tuple[MetricsTracker, dict | None]: The updated MetricsTracker with new statistics for this
        step, and the dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats: dict[str, Any] = {}
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Under gradient accumulation this context suppresses gradient sync (FSDP2:
    # set_requires_gradient_sync) on non-final micro-batches and divides the loss;
    # with gradient_accumulation_steps == 1 it is a transparent no-op.
    with accelerator.accumulate(policy):
        # Let accelerator handle mixed precision
        with accelerator.autocast():
            # `policy(...)`, never `policy.forward(...)`: FSDP2 all-gathers parameters through
            # nn.Module forward hooks, which only run via __call__.
            if sample_weights is not None:
                # Use per-sample loss for weighted training
                # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
                per_sample_loss, output_dict = policy(batch, reduction="none")

                # Weighted loss: each sample's contribution is scaled by its weight.
                # We divide by weight sum (not batch size) so that if some weights are zero,
                # the remaining samples contribute proportionally more, preserving gradient scale.
                # Weights are pre-normalized to sum to batch_size for stable training dynamics.
                epsilon = 1e-6
                loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

                # Log weighting statistics
                if output_dict is None:
                    output_dict = {}
                for key, value in weight_stats.items():
                    output_dict[f"sample_weight_{key}"] = value
            else:
                loss, output_dict = policy(batch)

            # TODO(rcadene): policy.unnormalize_outputs(out_dict)

        # Use accelerator's backward method
        accelerator.backward(loss)

        # Gradients are complete only on sync micro-batches; clipping partial gradients would
        # be meaningless. Always pass the full parameter list: accelerate's FSDP2 path requires
        # an exact match with the prepared model's parameters for a globally correct norm.
        grad_norm = None
        if accelerator.sync_gradients and grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        # Optimizer step (a no-op on non-final micro-batches under gradient accumulation).
        # Under fp16 the scaler runs it and skips it whenever a gradient overflowed.
        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()

        # `optimizer_step_was_skipped` is only refreshed on sync micro-batches, so elsewhere it
        # is a stale value from the previous update and must be read together with
        # `sync_gradients`. It is always False without a scaler ("no"/bf16 runs).
        update_was_skipped = accelerator.sync_gradients and accelerator.optimizer_step_was_skipped

        # Step through pytorch scheduler at every batch instead of epoch. Deliberately not
        # gated on the scaler: the schedule is a function of micro-batches consumed, and under
        # gradient accumulation it already advances on micro-batches that apply no update.
        if lr_scheduler is not None:
            lr_scheduler.step()

    # Update internal buffers if policy has update method, and the EMA. These track optimizer updates
    # (target networks, averages), not micro-batches: gate on the sync step under accumulation, and
    # on the update having actually landed — a skipped fp16 step left the weights untouched.
    if accelerator.sync_gradients and not update_was_skipped:
        unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
        if has_method(unwrapped_policy, "update"):
            unwrapped_policy.update()
        if ema is not None:
            ema.step(unwrapped_policy)

    train_metrics.loss = loss.item()
    # A skipped step's norm is inf/nan by construction (that is what the scaler detected);
    # recording it would poison the whole logging window's average.
    if grad_norm is not None and not update_was_skipped:
        train_metrics.grad_norm = grad_norm.item()
    # Only when the caller declared the meter: `MetricsTracker` raises on an undeclared name,
    # and the loss scale is diagnostic, not something a caller must opt into to train in fp16.
    if accelerator.scaler is not None and "grad_scale" in train_metrics.metrics:
        train_metrics.grad_scale = accelerator.scaler.get_scale()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available() and "gpu_mem_gb" in train_metrics.metrics:
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    # Aggregate the policy's scalar outputs for logging and rank-reduction across the log window.
    if output_dict:
        train_metrics.update_metrics(output_dict)
    return train_metrics, output_dict


def compute_offline_eval_loss(
    policy: torch.nn.Module,
    eval_dataloader: Any,
    preprocess_batch: Callable[[Any], Any],
    accelerator: "Accelerator",
    *,
    seed: int | None,
    ema: EMA | None = None,
) -> tuple[float, int]:
    """以指定权重计算验证损失，保证固定噪声不改变训练随机流。

    返回各进程的批次均值及总批次数；保留现有分布式归约口径。
    """
    # 某些策略会单独冻结子模块的训练模式，不能用统一 train() 覆盖它们。
    module_modes = [(module, module.training) for module in policy.modules()]
    rng_context = seeded_context(seed) if seed is not None else nullcontext()
    weights_context = (
        ema.average_parameters(accelerator.unwrap_model(policy)) if ema is not None else nullcontext()
    )
    try:
        policy.eval()
        eval_loss_sum = 0.0
        n_eval_batches = 0
        with rng_context, weights_context, torch.no_grad(), accelerator.autocast():
            for eval_batch in eval_dataloader:
                loss, _ = policy(preprocess_batch(eval_batch))
                eval_loss_sum += loss.item()
                n_eval_batches += 1
        loss_tensor = torch.tensor(eval_loss_sum / max(n_eval_batches, 1), device=accelerator.device)
        loss_value = accelerator.reduce(loss_tensor, reduction="mean").item()
        count_tensor = torch.tensor(n_eval_batches, device=accelerator.device)
        total_batches = int(accelerator.reduce(count_tensor, reduction="sum").item())
        return float(loss_value), total_batches
    finally:
        # 根模块先恢复，再修正与根模块状态不同的子模块。
        for module, was_training in module_modes:
            if module.training != was_training:
                module.train(was_training)


def make_dataloaders(
    cfg: TrainPipelineConfig,
    dataset: LeRobotDataset | StreamingLeRobotDataset,
    eval_dataset: LeRobotDataset | None,
    step: int,
    parallel_dims: ParallelDims,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Build the train (and optional eval) dataloader, including the sampler resume offset.

    The sampler offset is *derived* from `step` (`resume_before_prepare` loads step + RNG only):
    each loop step consumes `batch_size` samples on each of the `dp_world_size` distinct
    data-parallel workers — no grad-accumulation factor, since `step` counts micro-batches.

    Args:
        cfg (TrainPipelineConfig): The training config (batch size, workers, streaming, resume, seed).
        dataset (LeRobotDataset | StreamingLeRobotDataset): The training dataset.
        eval_dataset (LeRobotDataset | None): Optional held-out split; when provided, an eval
            dataloader is built (subsampled per task when `cfg.max_eval_samples > 0`).
        step (int): The loop step to resume the sampler from (0 for a fresh run).
        parallel_dims (ParallelDims): The resolved parallelism topology; provides the device type
            and the fallback dp world size for the resume offset.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]: The train
        dataloader and the eval dataloader (None when no eval split exists).
    """
    active_cfg = cfg.trainable_config
    if not cfg.dataset.streaming:
        if isinstance(dataset, StreamingLeRobotDataset):
            raise TypeError(
                "EpisodeAwareSampler requires a map-style dataset, got a StreamingLeRobotDataset."
            )
        # All non-streaming (map-style) datasets use EpisodeAwareSampler.
        # The order is a pure function of (seed, epoch), so every rank independently produces the
        # same permutation. accelerate then shards it disjointly across data-parallel ranks via
        # BatchSamplerShard without needing a `generator` attribute to synchronize an RNG, and
        # resume is sample-exact.
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_first_frames=getattr(active_cfg, "drop_n_first_frames", 0),
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if cfg.resume and step > 0:
            if cfg.checkpoint_path is None:
                raise ValueError(
                    "Resuming requires `checkpoint_path`; TrainPipelineConfig resolves it on --resume."
                )
            # The resume offset depends on the (dp_world_size, batch_size) that produced `step`,
            # so use the values recorded in the checkpoint (falling back to the current ones for
            # older checkpoints that did not store them).
            metadata = load_training_metadata(cfg.checkpoint_path / TRAINING_STATE_DIR)
            saved_dp_world = metadata["dp_world_size"]
            saved_batch_size = metadata["batch_size"]
            ckpt_dp_world = saved_dp_world or parallel_dims.dp_world_size
            ckpt_batch_size = saved_batch_size or cfg.batch_size
            if is_main_process() and saved_dp_world not in (None, parallel_dims.dp_world_size):
                logging.warning(
                    f"Resuming with dp_world_size={parallel_dims.dp_world_size} but the "
                    f"checkpoint was written with dp_world_size={saved_dp_world}. The data order "
                    "resumes at the right epoch/offset, but per-rank sample-exactness requires "
                    "the same data-parallel world size."
                )
            if is_main_process() and saved_batch_size not in (None, cfg.batch_size):
                logging.warning(
                    f"Resuming with batch_size={cfg.batch_size} but the checkpoint was written "
                    f"with batch_size={saved_batch_size}. The data order resumes at the right "
                    "epoch/offset, but per-rank sample-exactness requires the same batch size."
                )
            sampler_state = compute_sampler_state(step, len(sampler), ckpt_batch_size, ckpt_dp_world)
            sampler.load_state_dict(sampler_state)
            if is_main_process():
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
    else:
        shuffle = True
        sampler = None
        # Each DataLoader worker owns a frame-balanced part of the rank's episodes, with its own
        # pool and GIL. A worker needs at least one episode, so the rank's episode count caps them.
        rank_episodes = dataset.num_episodes_for_rank(parallel_dims.dp_rank, parallel_dims.dp_world_size)
        train_num_workers = min(cfg.num_workers, max(1, rank_episodes))
        if train_num_workers < cfg.num_workers and is_main_process():
            logging.info(
                "Using %d streaming DataLoader workers per rank: a rank owns %d episodes.",
                train_num_workers,
                rank_episodes,
            )
        dataset.set_data_parallel_shard(parallel_dims.dp_rank, parallel_dims.dp_world_size)  # not RANK env
        if (
            dataset.num_frames_for_rank(parallel_dims.dp_rank, parallel_dims.dp_world_size, train_num_workers)
            == 0
        ):
            raise ValueError("This rank owns no streaming episodes. Reduce the data-parallel world size.")
        if cfg.resume and step > 0:
            if cfg.checkpoint_path is None:
                raise ValueError(
                    "Resuming requires `checkpoint_path`; TrainPipelineConfig resolves it on --resume."
                )
            metadata = load_training_metadata(cfg.checkpoint_path / TRAINING_STATE_DIR)
            saved_dp_world = metadata["dp_world_size"]
            saved_batch_size = metadata["batch_size"]
            if saved_dp_world not in (None, parallel_dims.dp_world_size):
                raise ValueError(
                    "Sample-exact streaming resume requires the checkpoint data-parallel world size "
                    f"({saved_dp_world}) to match the current size ({parallel_dims.dp_world_size})."
                )
            if saved_batch_size not in (None, cfg.batch_size):
                raise ValueError(
                    "Sample-exact streaming resume requires the checkpoint batch size "
                    f"({saved_batch_size}) to match the current batch size ({cfg.batch_size})."
                )
            stream_offset = step * (saved_batch_size or cfg.batch_size)
            dataset.load_state_dict({"epoch": 0, "offset": stream_offset, "batch_size": cfg.batch_size})
            if is_main_process():
                logging.info("Resuming streaming data order at local sample %d", stream_offset)

    device_type = parallel_dims.device_type
    # Only swap in the language-aware collate when the dataset actually
    # declares language columns; otherwise stay on PyTorch's default
    # collate so non-language training runs are unaffected.
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_num_workers if cfg.dataset.streaming else cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        # Worker/iterator seeds must not consume the policy's diffusion-noise RNG
        # when an iterator is recreated after checkpoint resume.
        generator=torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None,
        pin_memory=device_type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=(
            cfg.prefetch_factor
            if (train_num_workers if cfg.dataset.streaming else cfg.num_workers) > 0
            else None
        ),
        persistent_workers=(
            cfg.persistent_workers and (train_num_workers if cfg.dataset.streaming else cfg.num_workers) > 0
        ),
        multiprocessing_context=(
            cfg.dataloader_multiprocessing_context
            if (train_num_workers if cfg.dataset.streaming else cfg.num_workers) > 0
            else None
        ),
    )

    # Build eval dataloader if a held-out split exists
    eval_dataloader = None
    if eval_dataset is not None:
        eval_ds = eval_dataset
        valid_frames = None
        drop_n_first_frames = getattr(active_cfg, "drop_n_first_frames", 0)
        drop_n_last_frames = getattr(active_cfg, "drop_n_last_frames", 0)
        if not cfg.dataset.streaming and (drop_n_first_frames or drop_n_last_frames):
            valid_frames = list(
                EpisodeAwareSampler(
                    eval_dataset.meta.episodes["dataset_from_index"],
                    eval_dataset.meta.episodes["dataset_to_index"],
                    episode_indices_to_use=eval_dataset.episodes,
                    drop_n_first_frames=getattr(active_cfg, "drop_n_first_frames", 0),
                    drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
                    absolute_to_relative_idx=eval_dataset.absolute_to_relative_idx,
                )
            )
            eval_ds = torch.utils.data.Subset(eval_dataset, valid_frames)
        if cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            if valid_frames is not None:
                task_arr = task_arr[valid_frames]
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for t in unique_tasks:
                frames = (task_arr == t).nonzero()[0][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_ds, selected)

        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            generator=torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None,
            pin_memory=device_type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            multiprocessing_context=cfg.dataloader_multiprocessing_context if cfg.num_workers > 0 else None,
        )
    return dataloader, eval_dataloader


def _with_relative_pose_stats(dataset: Any, policy_cfg: Any) -> dict[str, dict[str, Any]]:
    """返回数据集统计量的副本，其中 state/action 换成 SE(3) 相对位姿空间中的统计量。

    仅支持能提供 ``hf_dataset`` 的数据集（非流式、非 LanceDB 后端）。
    """
    try:
        hf_dataset = dataset.hf_dataset
    except AttributeError as err:
        raise ValueError(
            "`use_relative_pose` needs the full action/state columns to compute relative statistics, "
            f"but {type(dataset).__name__} does not expose `hf_dataset`."
        ) from err

    logging.info("Computing observation.state/action statistics in SE(3) relative pose space")
    relative_stats = compute_relative_pose_stats(
        hf_dataset,
        observation_delta_indices=list(policy_cfg.observation_delta_indices),
        action_delta_indices=list(policy_cfg.action_delta_indices),
        state_pose_starts=list(policy_cfg.relative_pose_state_starts),
        action_pose_starts=policy_cfg.relative_pose_action_starts,
        sample_stride=policy_cfg.relative_pose_stats_stride,
    )
    stats = dict(dataset.meta.stats)
    stats[OBS_STATE] = relative_stats[OBS_STATE]
    stats[ACTION] = relative_stats[ACTION]
    return stats


@parser.wrap()
def train(cfg: TrainPipelineConfig) -> None:
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and the distributed engine.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint (two-phase, around `accelerator.prepare`).
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Publishing the trained model to the Hugging Face Hub if configured.

    Args:
        cfg (TrainPipelineConfig): A `TrainPipelineConfig` object containing all training
            configurations, parsed from the CLI by `parser.wrap()`. On `--resume`, it is the config
            recorded in the checkpoint's `train_config.json`; when `cfg.job.is_remote`, the run is
            dispatched to HF Jobs instead of executing locally.
    """
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    require_package("accelerate", extra="training")

    cfg.validate()  # all fail-fasts fire here, before any distributed init

    # --- engine & topology --------------------------------------------------------------------
    # The factory is the ONLY accelerate configuration site: it guards against env-var
    # interference, resolves the declared parallelism degrees against the launched world, and
    # builds the Accelerator from the config mirrors.
    accelerator = make_accelerator(cfg)
    parallel_dims = ParallelDims.from_config(
        cfg.parallelism, accelerator.num_processes, accelerator.device.type
    )
    init_logging(accelerator=accelerator)

    if is_main_process():
        logging.info(pformat(cfg.to_dict()))

    if cfg.trackio.enable and cfg.trackio.project and is_main_process():
        trackio_logger = TrackioLogger(cfg)
    else:
        trackio_logger = None
        if is_main_process():
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    if cfg.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32

    # --- data (the main process downloads once; peers read the populated cache) ----------------
    if is_main_process():
        logging.info("Creating dataset")
        dataset, eval_dataset = make_train_eval_datasets(cfg)
    accelerator.wait_for_everyone()
    if not is_main_process():
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    split_manifest = None
    eval_protocol = None
    if eval_dataset is not None:
        if cfg.output_dir is None:
            raise ValueError("`output_dir` must be resolved before training starts.")
        split_manifest = build_split_manifest(cfg, dataset, eval_dataset)
        eval_protocol = build_eval_protocol(cfg, split_manifest, parallel_dims.dp_world_size)
        if cfg.resume:
            if cfg.checkpoint_path is None:
                raise ValueError("Resuming requires a checkpoint path.")
            # 历史检查点未记录协议，只有原始的尾部分割和 live 随机验证可延续其 best。
            allow_legacy = (
                cfg.dataset.eval_split_strategy == "tail" and cfg.eval_seed is None and not cfg.eval_use_ema
            )
            validate_eval_protocol(cfg.checkpoint_path, eval_protocol, allow_legacy=allow_legacy)
            best_path = cfg.output_dir / CHECKPOINTS_DIR / BEST_CHECKPOINT_LINK
            if cfg.keep_best_and_last and best_path.is_symlink():
                validate_eval_protocol(best_path, eval_protocol, allow_legacy=allow_legacy)
        if is_main_process():
            # 训练输出记录实际索引；数据集目录与其元数据保持只读。
            write_json(split_manifest, cfg.output_dir / SPLIT_MANIFEST)
            write_json(eval_protocol, cfg.output_dir / EVAL_PROTOCOL)
        accelerator.wait_for_everyone()

    # --- policy (weight source decided by the resume rule) -------------------------------------
    # On resume, cfg was parsed FROM the checkpoint's train_config.json, so cfg.checkpoint_format
    # IS the recorded value: DCP-bearing formats skip the safetensors load here and stream the
    # sharded weights in after prepare (resume_after_prepare).
    defer_weight_load = cfg.resume and cfg.checkpoint_format in (
        CheckpointFormat.DCP,
        CheckpointFormat.SAFETENSORS_DCP,
    )
    # On resume the weights and processors come from the checkpoint, while `pretrained_path`
    # keeps naming the model the run started from (the published card's `base_model`).
    resume_pretrained_dir = (
        cfg.checkpoint_path / PRETRAINED_MODEL_DIR if cfg.resume and cfg.checkpoint_path is not None else None
    )
    # validate() guarantees exactly one of `policy` / `reward_model` is set.
    active_cfg = cfg.trainable_config
    if isinstance(active_cfg, RewardModelConfig):
        if is_main_process():
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=active_cfg,
            pretrained_path=resume_pretrained_dir,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process():
            logging.info("Creating policy")
        policy = make_policy(
            cfg=active_cfg,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
            defer_weight_load=defer_weight_load,
            pretrained_path=resume_pretrained_dir,
        )

    peft_model = None
    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        require_package("peft", extra="peft")

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)
        peft_model = policy

    accelerator.wait_for_everyone()

    # --- processors (overrides built once, as one typed mapping) -------------------------------
    processor_pretrained_path = resume_pretrained_dir or active_cfg.pretrained_path
    if not cfg.resume and getattr(active_cfg, "recipe", None) is not None:
        if processor_pretrained_path is not None and is_main_process():
            logging.warning(
                "Language recipe fine-tuning rebuilds processors from the active configuration; "
                "saved processors from %s will not be loaded.",
                processor_pretrained_path,
            )
        # Language fine-tuning must use the active recipe, not the saved processor recipe.
        processor_pretrained_path = None

    processor_kwargs = ProcessorConfigKwargs()
    dataset_stats_for_processors = dataset.meta.stats
    if getattr(active_cfg, "use_relative_pose", False) and not cfg.resume:
        # 相对位姿模式下模型看到的是相对当前帧的 rot6d 位姿，归一化统计量必须在相对空间重新计算
        dataset_stats_for_processors = _with_relative_pose_stats(dataset, active_cfg)
    processor_dataset_stats = rename_stats(dataset_stats_for_processors, cfg.rename_map)
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = processor_dataset_stats
    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta
    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        # On resume, the checkpoint's saved processor stats are authoritative: they may have
        # been adapted by the policy (e.g. EVO1 pads state/action stats to max_state_dim),
        # and force-feeding raw dataset stats over them crashes normalization (#4006).
        # This mirrors the `dataset_stats` kwarg above, which is also skipped on resume.
        if not cfg.resume:
            preprocessor_overrides["normalizer_processor"]["stats"] = processor_dataset_stats
            postprocessor_overrides["unnormalizer_processor"]["stats"] = processor_dataset_stats
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if isinstance(active_cfg, RewardModelConfig):
        preprocessor, postprocessor = make_reward_pre_post_processors(
            active_cfg,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=active_cfg,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=active_cfg.pretrained_revision,
            **processor_kwargs,
        )

    image_augmentation = _make_image_augmentation(cfg, dataset, device, accelerator.process_index)
    if image_augmentation is not None and is_main_process():
        logging.info(f"Image transforms run on {device} for {image_augmentation.image_keys}")

    # Created BEFORE prepare on the unsharded parameters — accelerate's FSDP2 path requires the
    # model and optimizer in one prepare() call and rebinds the param groups itself.
    if is_main_process():
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    if cfg.output_dir is None or cfg.optimizer is None:
        # validate() resolves `output_dir` and make_optimizer_and_scheduler() rejects a missing optimizer.
        raise ValueError("`output_dir` and `optimizer` must be resolved before training starts.")

    # --- resume phase 1 + dataloaders ----------------------------------------------------------
    step = 0  # number of loop steps (= micro-batches consumed per data-parallel worker)
    if cfg.resume:
        step = resume_before_prepare(cfg)  # step + RNG only; sharded state loads after prepare

    dataloader, eval_dataloader = make_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims)

    # --- prepare & resume phase 2 ---------------------------------------------------------------
    # The FSDP wrap-unit class names resolve right before prepare: user override, else the
    # policy's _fsdp_wrap_modules declaration — root-only wrapping is never silently accepted.
    set_fsdp_wrap_modules(accelerator, policy)
    accelerator.wait_for_everyone()
    if cfg.dataset.streaming and eval_dataloader is not None:
        policy, optimizer, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, lr_scheduler, eval_dataloader
        )
    elif cfg.dataset.streaming:
        policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)
    elif eval_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, eval_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    finalize_sharded_policy(policy, parallel_dims)
    ema = EMA(accelerator.unwrap_model(policy), cfg.ema) if cfg.ema is not None else None
    if cfg.resume:
        resume_after_prepare(cfg, accelerator, policy, optimizer, lr_scheduler, ema)

    best_loss = math.inf
    best_step: int | None = None
    best_dir: str | None = None
    if cfg.keep_best_and_last and cfg.resume:
        if cfg.checkpoint_path is None:
            raise ValueError("Resuming requires a checkpoint path.")
        best_loss, best_step, best_dir = load_best_checkpoint_state(cfg.output_dir, cfg.checkpoint_path)
        if is_main_process():
            restore_best_checkpoint_link(cfg.output_dir, best_dir)
        accelerator.wait_for_everyone()

    # --- auxiliaries (after the core assembly, per the construction-order contract) -------------
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process():
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    # --- banner (main process only; numel() reads metadata — on DTensors it is the GLOBAL shape,
    # so the totals are correct even after sharding) ---------------------------------------------
    # One loop step consumes one micro-batch on every dp worker; the optimizer sees
    # `samples_per_step x gradient_accumulation_steps` samples per update.
    samples_per_step = cfg.batch_size * parallel_dims.dp_world_size
    effective_batch_size = samples_per_step * cfg.accelerator.gradient_accumulation.steps
    if is_main_process():
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {parallel_dims.dp_world_size} dp workers "
            f"x {cfg.accelerator.gradient_accumulation.steps} grad accum = {effective_batch_size} "
            f"(topology: dp_replicate={parallel_dims.dp_replicate}, dp_shard={parallel_dims.dp_shard})"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss the
        # data-parallel group is actually optimizing. grad_norm and lr are already identical on
        # every rank (post gradient sync / deterministic scheduler) so reducing them would be a
        # no-op collective.
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # Report the slowest rank for bottleneck-style timings so multi-GPU runs surface the
        # true straggler instead of rank 0's view.
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        "preprocessing_s": AverageMeter("prep_s", ":.3f", reduction="max"),
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "step_s": AverageMeter("step_s", ":.3f", reduction="max"),
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if accelerator.scaler is not None:
        # fp16 only. Identical on every rank (the overflow flag is reduced before the scaler
        # updates), so it needs no reduction — same as grad_norm and lr above. A falling scale
        # is the visible signature of skipped updates.
        train_metrics["grad_scale"] = AverageMeter("scale", ":.0f")
    if torch.cuda.is_available():
        # max() because headroom is gated by the worst-case rank.
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        dp_world_size=parallel_dims.dp_world_size,
    )

    if is_main_process():
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        step_start = time.perf_counter()
        batch = next(dl_iter)
        if cfg.dataset.streaming:
            batch = send_to_device(batch, device, non_blocking=device.type == "cuda")
        preprocessing_start = time.perf_counter()
        train_tracker.dataloading_s = preprocessing_start - step_start
        if image_augmentation is not None:
            batch = image_augmentation.observation(batch)
        batch = _preprocess_dataset_batch(batch, dataset.meta.camera_keys, cfg.rename_map, preprocessor)
        train_tracker.preprocessing_s = time.perf_counter() - preprocessing_start

        train_tracker, _ = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
            ema=ema,
        )
        train_tracker.step_s = time.perf_counter() - step_start

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process():
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = should_save_checkpoint(step, cfg.save_freq, cfg.steps)
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0
        is_eval_step = cfg.eval_steps > 0 and step % cfg.eval_steps == 0
        is_best_step = False

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            train_tracker.reduce_across_ranks()
            if is_main_process():
                if train_tracker.step_s.avg > 0:
                    train_tracker.samples_per_s = samples_per_step / train_tracker.step_s.avg
                logging.info(train_tracker)
                if trackio_logger:
                    # Policy sub-losses (latent_loss, action_loss, ...) are aggregated into the
                    # tracker by update_policy, so to_dict() already carries their windowed,
                    # rank-reduced averages — no per-step output_dict passthrough needed.
                    metrics_to_log = train_tracker.to_dict()
                    # Log sample weighting statistics if enabled
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        metrics_to_log.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    if ema is not None and ema.decay is not None:
                        metrics_to_log["ema/decay"] = ema.decay
                        metrics_to_log["ema/step"] = ema.num_updates
                    trackio_logger.log_dict(metrics_to_log, step)
            train_tracker.reset_averages()

        if is_eval_step and eval_dataloader is not None:
            eval_seed = (cfg.eval_seed + parallel_dims.dp_rank) % 2**32 if cfg.eval_seed is not None else None

            def preprocess_eval_batch(batch: Any) -> Any:
                return _preprocess_dataset_batch(
                    batch, dataset.meta.camera_keys, cfg.rename_map, preprocessor
                )

            live_eval_loss, eval_batches = compute_offline_eval_loss(
                policy, eval_dataloader, preprocess_eval_batch, accelerator, seed=eval_seed
            )
            eval_loss = live_eval_loss
            ema_eval_loss = None
            if cfg.eval_use_ema:
                if ema is None:
                    raise ValueError("eval_use_ema=true requires an EMA configuration.")
                # 固定种子时两套权重使用同一噪声序列；权重交换在保存前结束。
                ema_eval_loss, eval_batches = compute_offline_eval_loss(
                    policy, eval_dataloader, preprocess_eval_batch, accelerator, seed=eval_seed, ema=ema
                )
                eval_loss = ema_eval_loss

            if cfg.keep_best_and_last:
                # 所有进程根据同一个归约结果决定保存，空验证集不能产生最佳模型。
                is_best_step = eval_batches > 0 and math.isfinite(eval_loss) and eval_loss < best_loss
                if is_best_step:
                    best_loss, best_step = eval_loss, step

            if is_main_process():
                logging.info(f"step {step}: eval_loss={eval_loss:.4f} live_eval_loss={live_eval_loss:.4f}")
                if trackio_logger:
                    offline_eval_metrics = {"eval_loss": eval_loss, "live_eval_loss": live_eval_loss}
                    if ema_eval_loss is not None:
                        offline_eval_metrics["ema_eval_loss"] = ema_eval_loss
                    trackio_logger.log_dict(offline_eval_metrics, step=step, mode="eval")

        if cfg.save_checkpoint and (is_saving_step or is_best_step):
            # Collective: every rank participates (gathers / DCP shard writes); rank-0-only file
            # writes are gated inside save_checkpoint — no rank branches at the call site.
            if is_main_process():
                logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            write_dir = checkpoint_dir
            if cfg.keep_best_and_last:
                # 暂存目录名在所有进程上相同，主进程准备后再共同写入。
                write_dir = prepare_best_and_last_checkpoint(checkpoint_dir, is_main=is_main_process())
                accelerator.wait_for_everyone()
            save_checkpoint(
                checkpoint_dir=write_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                accelerator=accelerator,
                ema=ema,
            )
            if is_main_process():
                if split_manifest is not None and eval_protocol is not None:
                    # 模型子目录可独立复制；在提交 best/last 前写入完整评价口径。
                    artifact_dirs = [write_dir, write_dir / PRETRAINED_MODEL_DIR]
                    if ema is not None:
                        artifact_dirs.append(write_dir / PRETRAINED_MODEL_EMA_DIR)
                    for artifact_dir in artifact_dirs:
                        write_json(split_manifest, artifact_dir / SPLIT_MANIFEST)
                        write_json(eval_protocol, artifact_dir / EVAL_PROTOCOL)
                if cfg.keep_best_and_last:
                    checkpoint_dir = commit_best_and_last_checkpoint(
                        write_dir,
                        checkpoint_dir,
                        best_loss,
                        best_step,
                        best_dir,
                        update_best=is_best_step,
                        update_last=is_saving_step,
                    )
                    if is_best_step:
                        best_dir = checkpoint_dir.name
                else:
                    update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    if cfg.policy is None or not cfg.policy.repo_id:
                        # Already rejected by cfg.validate().
                        raise ValueError("save_checkpoint_to_hub requires --policy.repo_id.")
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        cfg.policy.repo_id,
                        private=cfg.policy.private,
                    )
                if trackio_logger:
                    trackio_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process():
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                eval_policy_model = accelerator.unwrap_model(policy)
                # With EMA, evaluate the EMA weights: the swap happens only on the main process (the
                # other ranks wait at the barrier below) and is exactly undone afterwards, so the live
                # weights stay in sync across ranks.
                with (
                    ema.average_parameters(eval_policy_model) if ema is not None else nullcontext(),
                    _make_eval_envs(cfg.env, cfg.eval) as eval_env,
                    torch.no_grad(),
                    accelerator.autocast(),
                ):
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=eval_policy_model,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    dp_world_size=parallel_dims.dp_world_size,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if trackio_logger:
                    metrics_to_log = {**eval_tracker.to_dict(), **eval_info}
                    trackio_logger.log_dict(metrics_to_log, step, mode="eval")
                    trackio_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process():
        progbar.close()
        logging.info("End of training")

    # --- publish (collective-safe: all ranks; the model commit gathers sharded weights) ---------
    if getattr(active_cfg, "push_to_hub", False):
        unwrapped = accelerator.unwrap_model(policy)
        model_to_publish = unwrapped.get_base_model() if peft_model is not None else unwrapped
        publish_trained_model(
            cfg,
            model_to_publish,
            preprocessor,
            postprocessor,
            dataset.meta,
            peft_model=unwrapped if peft_model is not None else None,
            ema=ema,
        )

    if trackio_logger:
        trackio_logger.finish()

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _remote_target_in_argv() -> bool:
    """Detect a remote HF Jobs run request on the raw CLI, before draccus parsing.

    Returns:
        bool: True when the CLI requests a remote HF Jobs run (`--job.target=<non-local>`).
    """
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        # The policy device is resolved on the remote pod, not here, so silence the
        # client-side "Device '...' is not available" warning PreTrainedConfig emits
        # while parsing the config (it fires before train() can dispatch remotely).
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
