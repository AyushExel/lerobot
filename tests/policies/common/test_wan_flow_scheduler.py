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

"""Numerical regression tests for FastWAM's Wan flow-matching schedulers."""

import numpy as np
import pytest
import torch

from lerobot.policies.common.wan_flow_scheduler import (
    WanContinuousFlowMatchScheduler,
    WanDiffusersFlowMatchInferenceScheduler,
)


def _historical_shift(values, shift):
    return shift * values / (1.0 + (shift - 1.0) * values)


def _historical_schedule(num_steps, shift, num_train_timesteps, dtype):
    sigmas = np.linspace(1, 0, num_steps + 1)[:num_steps]
    sigmas = torch.as_tensor(_historical_shift(sigmas, shift), dtype=torch.float32)
    timesteps = sigmas * float(num_train_timesteps)
    deltas = torch.cat([sigmas[1:], sigmas.new_zeros(1)]) - sigmas
    return timesteps.to(dtype), deltas.to(dtype)


def _historical_weight(timestep, *, num_train_timesteps, shift, eps):
    grid = torch.linspace(1.0, 0.0, num_train_timesteps + 1, dtype=torch.float64)[:-1]
    grid_t = _historical_shift(grid, shift) * float(num_train_timesteps)
    grid_y = torch.exp(-2.0 * ((grid_t - num_train_timesteps / 2.0) / num_train_timesteps) ** 2)
    y_min = float(grid_y.min())
    norm = float((grid_y - y_min).mean())

    t = timestep.to(torch.float32)
    y = torch.exp(-2.0 * ((t - num_train_timesteps / 2.0) / num_train_timesteps) ** 2)
    return (y - y_min) / (norm + eps)


@pytest.mark.parametrize("num_train_timesteps,shift", [(1000, 5.0), (37, 2.5)])
def test_continuous_training_t_matches_historical_rng(num_train_timesteps, shift):
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=num_train_timesteps, shift=shift)

    torch.manual_seed(17)
    u = torch.rand((8,), dtype=torch.float32)
    expected = _historical_shift(u, shift) * float(num_train_timesteps)

    torch.manual_seed(17)
    actual = scheduler.sample_training_t(8, torch.device("cpu"), torch.float32)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_continuous_training_primitives_match_historical(dtype):
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    timestep = torch.tensor([125.0, 500.0, 875.0], dtype=dtype)
    sample = torch.linspace(-1, 1, 18, dtype=dtype).reshape(3, 2, 3)
    noise = torch.flip(sample, dims=[-1])

    sigma = (timestep / 1000.0).to(dtype).view(3, 1, 1)
    expected_noisy = (1 - sigma) * sample + sigma * noise
    assert torch.equal(scheduler.add_noise(sample, noise, timestep), expected_noisy)
    assert torch.equal(scheduler.training_target(sample, noise, timestep), noise - sample)

    expected_weight = _historical_weight(
        timestep,
        num_train_timesteps=1000,
        shift=5.0,
        eps=scheduler.eps,
    )
    assert torch.equal(scheduler.training_weight(timestep), expected_weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_steps,shift", [(1, 1.0), (3, 5.0), (20, 7.5)])
def test_continuous_inference_schedule_and_step_match_historical(dtype, num_steps, shift):
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=3.0)
    actual_t, actual_delta = scheduler.build_inference_schedule(
        num_steps,
        device=torch.device("cpu"),
        dtype=dtype,
        shift_override=shift,
    )
    expected_t, expected_delta = _historical_schedule(num_steps, shift, 1000, dtype)
    assert torch.equal(actual_t, expected_t)
    assert torch.equal(actual_delta, expected_delta)

    sample = torch.linspace(-1, 1, 12, dtype=dtype).reshape(2, 2, 3)
    model_output = torch.full_like(sample, 0.25)
    for delta in actual_delta:
        expected = sample + model_output * delta.to(dtype)
        sample = scheduler.step(model_output, delta, sample)
        assert sample.dtype == dtype
        assert torch.equal(sample, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_steps,shift", [(1, 1.0), (3, 5.0), (20, 7.5)])
def test_diffusers_inference_adapter_is_bit_exact_with_historical_fastwam(dtype, num_steps, shift):
    pytest.importorskip("diffusers")
    historical = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=shift)
    expected_t, expected_delta = historical.build_inference_schedule(
        num_steps,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    delegated = WanDiffusersFlowMatchInferenceScheduler(num_train_timesteps=1000, shift=shift)
    actual_t = delegated.set_timesteps(num_steps, device=torch.device("cpu"), dtype=dtype)
    assert torch.equal(actual_t, expected_t)

    torch.manual_seed(23)
    expected_sample = torch.randn(2, 3, 4, dtype=dtype)
    actual_sample = expected_sample.clone()
    for delta in expected_delta:
        model_output = torch.randn_like(expected_sample)
        expected_sample = historical.step(model_output, delta, expected_sample)
        actual_sample = delegated.step(model_output, actual_sample)
        assert actual_sample.dtype == dtype
        assert torch.equal(actual_sample, expected_sample)


def test_diffusers_inference_adapter_validates_state_and_resets():
    pytest.importorskip("diffusers")
    scheduler = WanDiffusersFlowMatchInferenceScheduler(num_train_timesteps=1000, shift=5.0)
    sample = torch.zeros(1, 2)

    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.step(torch.ones_like(sample), sample)

    timesteps = scheduler.set_timesteps(2, device=torch.device("cpu"), dtype=torch.float32)
    for _ in timesteps:
        sample = scheduler.step(torch.ones_like(sample), sample)
    with pytest.raises(RuntimeError, match="already been consumed"):
        scheduler.step(torch.ones_like(sample), sample)

    reset_timesteps = scheduler.set_timesteps(3, device=torch.device("cpu"), dtype=torch.float32)
    assert reset_timesteps.shape == (3,)
    scheduler.step(torch.ones_like(sample), sample)


@pytest.mark.parametrize(
    "scheduler_cls",
    [WanContinuousFlowMatchScheduler, WanDiffusersFlowMatchInferenceScheduler],
)
def test_wan_continuous_schedulers_validate_configuration(scheduler_cls):
    with pytest.raises(ValueError, match="num_train_timesteps"):
        scheduler_cls(num_train_timesteps=0)
    with pytest.raises(ValueError, match="shift"):
        scheduler_cls(shift=0)

    scheduler = scheduler_cls()
    method = (
        scheduler.build_inference_schedule
        if scheduler_cls is WanContinuousFlowMatchScheduler
        else scheduler.set_timesteps
    )
    with pytest.raises(ValueError, match="num_inference_steps"):
        method(0, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError, match="shift"):
        method(1, device=torch.device("cpu"), dtype=torch.float32, shift_override=0)
