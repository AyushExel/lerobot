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

import math

import torch

from lerobot.policies.common.embeddings import SinusoidalPosEmb


def test_sinusoidal_pos_emb_matches_reference_formula():
    dim = 128
    module = SinusoidalPosEmb(dim)
    x = torch.tensor([0.0, 1.0, 17.0, 999.0])
    out = module(x)
    assert out.shape == (4, dim)

    # Independent recomputation of the classic log-10000 formula
    # (the exact math historically duplicated in diffusion, multi_task_dit and wall_x).
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = x[:, None] * emb[None, :]
    expected = torch.cat((emb.sin(), emb.cos()), dim=-1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    # t=0 embeds to [sin(0)=0 ..., cos(0)=1 ...].
    assert torch.equal(out[0, :half_dim], torch.zeros(half_dim))
    assert torch.equal(out[0, half_dim:], torch.ones(half_dim))


def test_sinusoidal_pos_emb_is_stateless():
    module = SinusoidalPosEmb(16)
    assert len(list(module.parameters())) == 0
    assert module.state_dict() == {}
