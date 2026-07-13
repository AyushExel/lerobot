#!/usr/bin/env python

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

"""DDPM/DDIM training and sampling primitives shared across policies.

Canonical versions of the noise-scheduler factory, the epsilon/sample/velocity training-loss
computation, and the iterative denoising loop that the diffusion and multi_task_dit
policies historically each carried a copy of. All functions are stateless; adopting
them does not affect checkpoints.

MultiTaskDiT intentionally retains its policy-specific ``DiffusionObjective`` wrapper;
this module shares the scheduler, target, loss, and sampling mechanics rather than a
single policy-level objective class.

``diffusers`` is conditionally imported so this module stays importable with base
dependencies only while keeping imports visible at module scope.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.import_utils import _diffusers_available, require_package

if TYPE_CHECKING or _diffusers_available:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
else:
    DDIMScheduler = None
    DDPMScheduler = None


def make_noise_scheduler(name: str, **kwargs) -> "DDPMScheduler | DDIMScheduler":
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if DDIMScheduler is None or DDPMScheduler is None:
        require_package("diffusers", extra="diffusers-dep")

    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


def make_inference_scheduler(
    noise_scheduler: "DDPMScheduler | DDIMScheduler", name: str | None = None
) -> "DDPMScheduler | DDIMScheduler":
    """Build an optional inference-only solver from a training scheduler's config.

    Returning the original instance when no override is requested preserves the
    historical default exactly. Switching solver delegates config translation to
    Diffusers' ``SchedulerMixin.from_config`` implementation.
    """
    if name is None:
        return noise_scheduler
    if DDIMScheduler is None or DDPMScheduler is None:
        require_package("diffusers", extra="diffusers-dep")

    scheduler_types = {"DDPM": DDPMScheduler, "DDIM": DDIMScheduler}
    try:
        scheduler_type = scheduler_types[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported inference noise scheduler type {name}") from exc

    if type(noise_scheduler) is scheduler_type:
        return noise_scheduler
    return scheduler_type.from_config(noise_scheduler.config)


def compute_diffusion_loss(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    noise_scheduler: "DDPMScheduler | DDIMScheduler",
    trajectory: Tensor,
    prediction_type: str,
    action_is_pad: Tensor | None = None,
    noise_like_trajectory: bool = True,
) -> Tensor:
    """DDPM/DDIM training loss on a clean trajectory.

    Samples noise and per-item timesteps from the global RNG, applies the scheduler's
    forward process, runs the denoiser, and returns the MSE against either the noise
    ("epsilon"), the clean trajectory ("sample"), or the scheduler velocity
    ("v_prediction").

    Args:
        denoise_fn: Computes the model prediction from ``(noisy_trajectory, timesteps)``
            where ``timesteps`` is an int64 tensor of shape ``(batch_size,)``.
        noise_scheduler: A DDPM/DDIM scheduler; provides ``config.num_train_timesteps``
            and ``add_noise``.
        trajectory: Clean action trajectory of shape ``(batch_size, horizon, action_dim)``.
        prediction_type: "epsilon" (predict the noise), "sample" (predict the clean
            trajectory), or "v_prediction" (predict scheduler velocity).
        action_is_pad: Optional ``(batch_size, horizon)`` bool mask; when given, padded
            steps (edges of the dataset trajectory) are excluded from the loss mean.
        noise_like_trajectory: When True (multi_task_dit convention) noise follows the
            trajectory's dtype (``randn_like``); when False (diffusion convention) noise
            is sampled in the process default dtype regardless of the trajectory dtype.
            The two only differ for non-float32 action batches.
    """
    # Sample noise to add to the trajectory.
    if noise_like_trajectory:
        eps = torch.randn_like(trajectory)
    else:
        eps = torch.randn(trajectory.shape, device=trajectory.device)
    # Sample a random noising timestep for each item in the batch.
    timesteps = torch.randint(
        low=0,
        high=noise_scheduler.config.num_train_timesteps,
        size=(trajectory.shape[0],),
        device=trajectory.device,
    ).long()
    # Add noise to the clean trajectories according to the noise magnitude at each timestep.
    noisy_trajectory = noise_scheduler.add_noise(trajectory, eps, timesteps)

    # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
    pred = denoise_fn(noisy_trajectory, timesteps)

    # The target is either the original trajectory, or the noise.
    if prediction_type == "epsilon":
        target = eps
    elif prediction_type == "sample":
        target = trajectory
    elif prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(trajectory, eps, timesteps)
    else:
        raise ValueError(f"Unsupported prediction type {prediction_type}")

    loss = F.mse_loss(pred, target, reduction="none")

    # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
    if action_is_pad is not None:
        mask = (~action_is_pad).unsqueeze(-1)
        num_valid = mask.sum() * loss.shape[-1]
        return (loss * mask).sum() / num_valid.clamp_min(1)

    return loss.mean()


def run_conditional_sample(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    noise_scheduler: "DDPMScheduler | DDIMScheduler",
    num_inference_steps: int,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Iterative DDPM/DDIM denoising from a standard-normal prior to a clean sample.

    Args:
        denoise_fn: Computes the model prediction from ``(sample, timesteps)`` where
            ``timesteps`` is an int64 tensor of shape ``(batch_size,)`` filled with the
            current scheduler timestep.
        noise_scheduler: A DDPM/DDIM scheduler; ``set_timesteps`` is called on it.
        num_inference_steps: Number of denoising steps.
        shape: Shape of the prior sample, ``(batch_size, horizon, action_dim)``.
        device: Device to draw the prior on (ignored if ``noise`` is given).
        dtype: Dtype of the prior sample (ignored if ``noise`` is given).
        noise: Optional pre-drawn prior sample used instead of drawing one.
        generator: Optional RNG used for the prior and the scheduler's noise injection.
    """
    # Sample prior.
    sample = (
        noise
        if noise is not None
        else torch.randn(size=shape, dtype=dtype, device=device, generator=generator)
    )

    noise_scheduler.set_timesteps(num_inference_steps)

    for t in noise_scheduler.timesteps:
        # Predict model output.
        model_output = denoise_fn(
            sample, torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)
        )
        # Compute previous image: x_t -> x_t-1
        sample = noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

    return sample
