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

"""Wan2.2 shift-based flow-matching schedulers shared by LingBot-VA and FastWAM.

Both policies use the same math: the Wan sigma shift ``s*u / (1 + (s-1)*u)``, linear
noising ``(1-sigma)*x0 + sigma*noise``, the velocity target ``noise - x0``, Euler steps
``x + v*(sigma_next - sigma)``, and the 'bsmntw' bell training weight
``exp(-2*((t - T/2)/T)^2) - min``. Two entry styles are provided over that shared math:

- ``WanFlowMatchScheduler``: stateful, discrete precomputed-timesteps + argmin lookup
  (DiffSynth convention, used by LingBot-VA).
- ``WanContinuousFlowMatchScheduler``: stateless continuous-time sampling
  and training primitives (used by FastWAM training and as the historical
  numerical reference for inference).
- ``WanDiffusersFlowMatchInferenceScheduler``: a thin FastWAM inference adapter
  over diffusers' ``FlowMatchEulerDiscreteScheduler``. It supplies FastWAM's
  already-shifted sigma grid so the delegated schedule remains bit-identical.

The bell-weight *normalization* intentionally differs between the two and is exposed
explicitly via :func:`normalize_bsmntw_weights`: the discrete grid convention
``y_shifted * (T / y_shifted.sum())`` and the continuous convention
``y_shifted / (mean(y_shifted_grid) + eps)`` agree mathematically (``T/sum == 1/mean``
on a T-point grid) but not bit-for-bit (float32 grid vs float64 stats, mul-vs-div, eps:
max abs diff ~2.4e-7), so each policy keeps its exact historical numerics.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.utils.import_utils import _diffusers_available, require_package

if TYPE_CHECKING or _diffusers_available:
    from diffusers import FlowMatchEulerDiscreteScheduler
else:
    FlowMatchEulerDiscreteScheduler = None


def shift_sigmas(sigmas, shift):
    """Wan sigma shift transform ``s*u / (1 + (s-1)*u)`` (elementwise; torch or numpy)."""
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def get_sampling_sigmas(sampling_steps, shift):
    # Vendored from Wan2.2 (formerly wan/utils/fm_solvers.py); computes the
    # noise-level (sigma) schedule for Wan-compatible flow-matching inference.
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    return shift_sigmas(sigma, shift)


def bsmntw_training_bell(timesteps, num_timesteps):
    """Unnormalized 'bsmntw' training-weight bell ``exp(-2*((t - T/2)/T)^2)``."""
    return torch.exp(-2 * ((timesteps - num_timesteps / 2) / num_timesteps) ** 2)


def normalize_bsmntw_weights(y_shifted, normalization, *, num_timesteps=None, grid_mean=None, eps=0.0):
    """Normalize min-shifted bell weights. The two Wan ports normalize differently and NOT
    bit-identically, so the convention is an explicit parameter (do not unify them):

    - ``"grid_sum"``: ``y_shifted * (num_timesteps / y_shifted.sum())`` over the full
      discrete grid (LingBot-VA / DiffSynth).
    - ``"grid_mean_eps"``: ``y_shifted / (grid_mean + eps)`` with ``grid_mean`` precomputed
      from a float64 reference grid (FastWAM).
    """
    if normalization == "grid_sum":
        return y_shifted * (num_timesteps / y_shifted.sum())
    if normalization == "grid_mean_eps":
        return y_shifted / (grid_mean + eps)
    raise ValueError(f"Unknown bsmntw weight normalization: {normalization!r}")


def flow_matching_add_noise(original_samples, noise, sigma):
    """Linear flow-matching noising ``(1-sigma)*x0 + sigma*noise`` (``sigma`` broadcast-ready)."""
    return (1 - sigma) * original_samples + sigma * noise


def flow_matching_target(sample, noise):
    """Flow-matching velocity training target ``noise - x0``."""
    return noise - sample


def flow_matching_euler_step(sample, model_output, delta):
    """One Euler integration step ``x + v*delta`` with ``delta = sigma_next - sigma``."""
    return sample + model_output * delta


class WanFlowMatchScheduler:
    """Discrete Wan flow-matching scheduler (precomputed timesteps + argmin lookup).

    DiffSynth-style ``FlowMatchScheduler``. LingBot-VA uses two independent instances at
    inference (one for the video-latent stream, one for the action stream), each with its
    own ``shift`` and number of denoising steps.
    """

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        shift=None,
        dynamic_shift_len=None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = (
                self.calculate_shift(dynamic_shift_len)
                if dynamic_shift_len is not None
                else self.exponential_shift_mu
            )
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = shift_sigmas(self.sigmas, self.shift)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            y = bsmntw_training_bell(self.timesteps, num_inference_steps)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = normalize_bsmntw_weights(
                y_shifted, "grid_sum", num_timesteps=num_inference_steps
            )
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        return flow_matching_euler_step(sample, model_output, sigma_ - sigma)

    def add_noise(self, original_samples, noise, timestep, t_dim=2):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep = timestep[None]
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep).abs(), dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = timestep_id.shape[0]
        sigma = self.sigmas[timestep_id].to(original_samples).view(shape)
        return flow_matching_add_noise(original_samples, noise, sigma)

    def training_target(self, sample, noise, timestep):
        return flow_matching_target(sample, noise)

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0
        )
        weights = self.linear_timesteps_weights.to(timestep.device)[timestep_id].to(timestep.device)
        return weights

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based Wan sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = shift_sigmas(u_grid, self.shift) * float(steps)
        y_grid = bsmntw_training_bell(t_grid, steps)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = shift_sigmas(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        y = bsmntw_training_bell(t, self.num_train_timesteps)
        y_shifted = y - self._y_min
        weight = normalize_bsmntw_weights(
            y_shifted, "grid_mean_eps", grid_mean=self._weight_norm_const, eps=self.eps
        )
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim != 0:
            sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return flow_matching_add_noise(original_samples, noise, sigma)

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return flow_matching_target(sample, noise)

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        sigma_steps = torch.as_tensor(
            get_sampling_sigmas(num_inference_steps, shift),
            device=device,
            dtype=torch.float32,
        )
        timesteps = sigma_steps * float(self.num_train_timesteps)
        sigma_next = torch.cat([sigma_steps[1:], sigma_steps.new_zeros(1)])
        deltas = sigma_next - sigma_steps
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim != 0:
            delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return flow_matching_euler_step(sample, model_output, delta)


class WanDiffusersFlowMatchInferenceScheduler:
    """FastWAM inference adapter over diffusers' FlowMatch Euler scheduler.

    Diffusers' default grid includes its training-time ``sigma_min`` endpoint and is
    therefore not the Wan2.2 grid. FastWAM historically uses ``N`` points from a
    ``linspace(1, 0, N + 1)`` with the terminal zero excluded, followed by Wan's
    rational shift. We pass those already-shifted sigmas to diffusers with ``shift=1``;
    diffusers then owns timestep storage, step indexing, and the Euler update without
    changing FastWAM's established numerics.

    This class is intentionally inference-only. FastWAM's continuous timestep sampling
    and BSMNTW loss weighting remain in :class:`WanContinuousFlowMatchScheduler` because
    diffusers does not implement that training objective.
    """

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self._scheduler = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> torch.Tensor:
        """Configure and return model-facing timesteps in the requested dtype."""
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        if FlowMatchEulerDiscreteScheduler is None:
            require_package("diffusers", extra="fastwam")

        shifted_sigmas = get_sampling_sigmas(num_inference_steps, shift)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps,
            # The supplied sigma grid is already Wan-shifted. Applying the shift again
            # would silently change released FastWAM inference trajectories.
            shift=1.0,
        )
        scheduler.set_timesteps(sigmas=shifted_sigmas.tolist(), device=device)
        self._scheduler = scheduler
        return scheduler.timesteps.to(device=device, dtype=dtype)

    def step(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Advance one configured step using diffusers' stateful Euler implementation."""
        if self._scheduler is None:
            raise RuntimeError("Call `set_timesteps` before `step`.")
        step_index = self._scheduler.step_index
        step_index = 0 if step_index is None else step_index
        if step_index >= len(self._scheduler.timesteps):
            raise RuntimeError("All configured inference steps have already been consumed.")
        timestep = self._scheduler.timesteps[step_index]
        return self._scheduler.step(model_output, timestep, sample, return_dict=False)[0]
