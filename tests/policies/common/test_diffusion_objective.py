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

"""Behavior-pinning tests for the shared DDPM/DDIM diffusion-objective primitives.

``compute_diffusion_loss`` and ``run_conditional_sample`` are compared against verbatim
copies of the historical diffusion/multi_task_dit code (including their RNG consumption
order): any divergence from those references is a behavior change for released
checkpoints.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

pytest.importorskip("diffusers")

from lerobot.policies.common.diffusion_objective import (  # noqa: E402
    compute_diffusion_loss,
    make_inference_scheduler,
    make_noise_scheduler,
    run_conditional_sample,
)

SCHEDULER_KWARGS = {
    "num_train_timesteps": 100,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "squaredcos_cap_v2",
    "clip_sample": True,
    "clip_sample_range": 1.0,
}


def _denoise_fn(x, t):
    """Deterministic parameter-free stand-in for the denoising network."""
    return 0.1 * x + 0.01 * t.to(x.dtype).view(-1, *([1] * (x.ndim - 1)))


def test_make_noise_scheduler_types_and_kwargs():
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    ddpm = make_noise_scheduler("DDPM", **SCHEDULER_KWARGS, prediction_type="epsilon")
    ddim = make_noise_scheduler("DDIM", **SCHEDULER_KWARGS, prediction_type="sample")
    assert type(ddpm) is DDPMScheduler and type(ddim) is DDIMScheduler
    assert ddpm.config.num_train_timesteps == 100
    assert ddpm.config.prediction_type == "epsilon"
    assert ddim.config.prediction_type == "sample"
    with pytest.raises(ValueError, match="Unsupported noise scheduler type"):
        make_noise_scheduler("PNDM")


def test_make_inference_scheduler_uses_diffusers_from_config():
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    train_scheduler = make_noise_scheduler("DDPM", **SCHEDULER_KWARGS, prediction_type="epsilon")
    assert make_inference_scheduler(train_scheduler) is train_scheduler
    inference_scheduler = make_inference_scheduler(train_scheduler, "DDIM")
    assert type(inference_scheduler) is DDIMScheduler
    assert inference_scheduler.config.num_train_timesteps == train_scheduler.config.num_train_timesteps
    assert inference_scheduler.config.prediction_type == train_scheduler.config.prediction_type
    with pytest.raises(ValueError, match="Unsupported inference noise scheduler type"):
        make_inference_scheduler(train_scheduler, "PNDM")


def test_policy_configs_accept_velocity_prediction_and_inference_solver():
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig

    diffusion_config = DiffusionConfig(prediction_type="v_prediction", inference_noise_scheduler_type="DDIM")
    dit_config = MultiTaskDiTConfig(prediction_type="v_prediction", inference_noise_scheduler_type="DDIM")
    assert diffusion_config.prediction_type == dit_config.prediction_type == "v_prediction"
    assert (
        diffusion_config.inference_noise_scheduler_type == dit_config.inference_noise_scheduler_type == "DDIM"
    )


def _reference_loss(scheduler, trajectory, prediction_type, action_is_pad):
    """Verbatim copy of the historical DiffusionModel.compute_loss diffusion block."""
    eps = torch.randn(trajectory.shape, device=trajectory.device)
    timesteps = torch.randint(
        low=0,
        high=scheduler.config.num_train_timesteps,
        size=(trajectory.shape[0],),
        device=trajectory.device,
    ).long()
    noisy_trajectory = scheduler.add_noise(trajectory, eps, timesteps)
    pred = _denoise_fn(noisy_trajectory, timesteps)
    if prediction_type == "epsilon":
        target = eps
    elif prediction_type == "sample":
        target = trajectory
    else:
        target = scheduler.get_velocity(trajectory, eps, timesteps)
    loss = F.mse_loss(pred, target, reduction="none")
    if action_is_pad is not None:
        mask = (~action_is_pad).unsqueeze(-1)
        num_valid = mask.sum() * loss.shape[-1]
        return (loss * mask).sum() / num_valid.clamp_min(1)
    return loss.mean()


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
@pytest.mark.parametrize("with_pad_mask", [False, True])
def test_compute_diffusion_loss_matches_historical(prediction_type, with_pad_mask):
    scheduler = make_noise_scheduler("DDPM", **SCHEDULER_KWARGS, prediction_type=prediction_type)
    torch.manual_seed(0)
    trajectory = torch.randn(2, 16, 4)
    action_is_pad = None
    if with_pad_mask:
        action_is_pad = torch.zeros(2, 16, dtype=torch.bool)
        action_is_pad[:, -3:] = True

    torch.manual_seed(42)
    expected = _reference_loss(scheduler, trajectory, prediction_type, action_is_pad)
    torch.manual_seed(42)
    actual = compute_diffusion_loss(
        _denoise_fn, scheduler, trajectory, prediction_type, action_is_pad=action_is_pad
    )
    assert torch.equal(actual, expected)


def test_compute_diffusion_loss_noise_dtype_conventions():
    """Pin the two historical noise conventions for non-float32 trajectories.

    The diffusion policy sampled noise with ``torch.randn(shape, device=...)`` (process
    default dtype, i.e. float32) regardless of the trajectory dtype, while
    multi_task_dit used ``randn_like``. They only differ for non-float32 batches.
    """
    scheduler = make_noise_scheduler("DDPM", **SCHEDULER_KWARGS, prediction_type="epsilon")
    torch.manual_seed(1)
    trajectory = torch.randn(2, 16, 4).to(torch.bfloat16)

    # diffusion convention: float32 noise (verbatim historical reference above).
    torch.manual_seed(7)
    expected = _reference_loss(scheduler, trajectory, "epsilon", None)
    torch.manual_seed(7)
    actual = compute_diffusion_loss(
        _denoise_fn, scheduler, trajectory, "epsilon", noise_like_trajectory=False
    )
    assert torch.equal(actual, expected)

    # multi_task_dit convention: noise follows the trajectory dtype.
    torch.manual_seed(7)
    eps = torch.randn_like(trajectory)
    assert eps.dtype == torch.bfloat16
    torch.manual_seed(7)
    like = compute_diffusion_loss(_denoise_fn, scheduler, trajectory, "epsilon")
    assert not torch.equal(like, actual)


def test_compute_diffusion_loss_rejects_unknown_prediction_type():
    scheduler = make_noise_scheduler("DDPM", **SCHEDULER_KWARGS, prediction_type="epsilon")
    with pytest.raises(ValueError, match="Unsupported prediction type"):
        compute_diffusion_loss(_denoise_fn, scheduler, torch.randn(1, 8, 2), "velocity")


def _reference_conditional_sample(scheduler, num_inference_steps, shape, generator=None, noise=None):
    """Verbatim copy of the historical DiffusionModel.conditional_sample loop."""
    sample = (
        noise
        if noise is not None
        else torch.randn(size=shape, dtype=torch.float32, device="cpu", generator=generator)
    )
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        model_output = _denoise_fn(
            sample, torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)
        )
        sample = scheduler.step(model_output, t, sample, generator=generator).prev_sample
    return sample


@pytest.mark.parametrize("scheduler_type", ["DDPM", "DDIM"])
@pytest.mark.parametrize("prior", ["global_rng", "generator", "noise"])
def test_run_conditional_sample_matches_historical(scheduler_type, prior):
    shape = (2, 16, 4)
    kwargs = {}
    if prior == "noise":
        torch.manual_seed(3)
        kwargs["noise"] = torch.randn(shape)

    def make_kwargs():
        if prior == "generator":
            return {**kwargs, "generator": torch.Generator().manual_seed(7)}
        return dict(kwargs)

    scheduler = make_noise_scheduler(scheduler_type, **SCHEDULER_KWARGS, prediction_type="epsilon")
    torch.manual_seed(11)
    expected = _reference_conditional_sample(scheduler, 10, shape, **make_kwargs())

    scheduler = make_noise_scheduler(scheduler_type, **SCHEDULER_KWARGS, prediction_type="epsilon")
    torch.manual_seed(11)
    actual = run_conditional_sample(
        _denoise_fn, scheduler, 10, shape, device="cpu", dtype=torch.float32, **make_kwargs()
    )
    assert torch.equal(actual, expected)
    assert actual.shape == shape
