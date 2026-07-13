#!/usr/bin/env python

# Copyright 2025 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""GR00T-specific DiT variants.

The shared cross-attention DiT building blocks (``TimestepEncoder``,
``AdaLayerNorm``, ``BasicTransformerBlock``, ``DiT``) live in
``lerobot.policies.common.heads.cross_attention_dit``.
"""

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from lerobot.policies.common.heads.cross_attention_dit import BasicTransformerBlock, DiT
from lerobot.utils.import_utils import _diffusers_available, require_package

if TYPE_CHECKING or _diffusers_available:
    from diffusers import ConfigMixin, ModelMixin
    from diffusers.configuration_utils import register_to_config
else:
    ConfigMixin = object
    ModelMixin = nn.Module
    register_to_config = lambda fn: fn  # noqa: E731


logger = logging.getLogger(__name__)


class AlternateVLDiT(DiT):
    """N1.7 DiT variant that alternates cross-attention over image and text tokens."""

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        require_package("diffusers", extra="groot")
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
        image_mask: torch.Tensor | None = None,
        backbone_attention_mask: torch.Tensor | None = None,
    ):
        if image_mask is None:
            raise ValueError("image_mask is required for AlternateVLDiT.")
        if backbone_attention_mask is None:
            raise ValueError("backbone_attention_mask is required for AlternateVLDiT.")

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask

        all_hidden_states = [hidden_states]
        if not self.config.interleave_self_attention:
            raise ValueError("AlternateVLDiT requires interleave_self_attention=True.")

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                curr_encoder_attention_mask = (
                    non_image_attention_mask
                    if idx % (2 * self.attend_text_every_n_blocks) == 0
                    else image_attention_mask
                )
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: str | None = "sinusoidal",
        interleave_self_attention=False,
    ):
        require_package("diffusers", extra="groot")
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        logger.debug(
            "Total number of SelfAttentionTransformer parameters: %d",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for _idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
