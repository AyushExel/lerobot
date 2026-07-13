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

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Beta

from lerobot.policies.common.heads.cross_attention_dit import DiT
from lerobot.utils.import_utils import require_package

from .configuration_vla_jepa import VLAJEPAConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        batch_size, seq_len = timesteps.shape
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device)
        exponent = exponent * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / max(half_dim, 1))
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).view(batch_size, seq_len, -1)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [batch_size].")
        timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)
        action_emb = self.layer1(actions)
        time_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        return self.layer3(F.silu(self.layer2(torch.cat([action_emb, time_emb], dim=-1))))


@dataclass
class ActionModelPreset:
    hidden_size: int
    attention_head_dim: int
    num_attention_heads: int


DIT_PRESETS = {
    "DiT-B": ActionModelPreset(hidden_size=768, attention_head_dim=64, num_attention_heads=12),
    "DiT-L": ActionModelPreset(hidden_size=1536, attention_head_dim=48, num_attention_heads=32),
    "DiT-test": ActionModelPreset(hidden_size=16, attention_head_dim=8, num_attention_heads=2),
}


class VLAJEPAActionHead(nn.Module):
    def __init__(self, config: VLAJEPAConfig, cross_attention_dim: int) -> None:
        require_package("diffusers", extra="vla_jepa")
        super().__init__()
        preset = DIT_PRESETS[config.action_model_type]
        self.config = config
        num_heads = config.action_num_heads or preset.num_attention_heads
        head_dim = config.action_attention_head_dim or preset.attention_head_dim
        inner_dim = num_heads * head_dim  # e.g. DiT-B: 12 × 64 = 768

        self.input_embedding_dim = inner_dim
        self.action_horizon = config.chunk_size
        self.num_inference_timesteps = config.num_inference_timesteps

        hidden_size = config.action_hidden_size
        # Pin the shared DiT to VLA-JEPA's historical architecture: ada_norm blocks that
        # alternate cross- and self-attention, no positional embeddings, feed-forward
        # final dropout but no extra dropout on the attention output.
        self.model = DiT(
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            output_dim=hidden_size,
            num_layers=config.action_num_layers,
            dropout=config.action_dropout,
            cross_attention_dim=cross_attention_dim,
            attention_bias=True,
            activation_fn="gelu-approximate",
            norm_type="ada_norm",
            final_dropout=True,
            positional_embeddings=None,
            interleave_self_attention=True,
            attn_output_dropout=False,
        )
        self.action_encoder = ActionEncoder(config.action_dim, inner_dim)
        self.action_decoder = nn.Sequential(
            OrderedDict(
                [
                    ("layer1", nn.Linear(hidden_size, hidden_size)),
                    ("relu", nn.ReLU()),
                    ("layer2", nn.Linear(hidden_size, config.action_dim)),
                ]
            )
        )
        self.state_encoder = (
            nn.Sequential(
                OrderedDict(
                    [
                        ("layer1", nn.Linear(config.state_dim, hidden_size)),
                        ("relu", nn.ReLU()),
                        ("layer2", nn.Linear(hidden_size, inner_dim)),
                    ]
                )
            )
            if config.state_dim > 0
            else None
        )
        self.future_tokens = nn.Embedding(config.num_embodied_action_tokens_per_instruction, inner_dim)
        self.position_embedding = nn.Embedding(
            max(1024, config.chunk_size + config.num_action_tokens_per_timestep + 4),
            inner_dim,
        )
        self.beta_dist = Beta(config.action_noise_beta_alpha, config.action_noise_beta_beta)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (self.config.action_noise_s - sample) / self.config.action_noise_s

    def _build_inputs(
        self,
        conditioning_tokens: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        action_features = self.action_encoder(actions, timesteps)
        pos_ids = torch.arange(action_features.shape[1], device=actions.device)
        action_features = action_features + self.position_embedding(pos_ids)[None]

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(actions.shape[0], -1, -1)
        seq = [future_tokens, action_features]
        if state is not None and self.state_encoder is not None:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            seq.insert(0, self.state_encoder(state))
        return torch.cat(seq, dim=1)

    def forward(
        self,
        conditioning_tokens: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy_actions = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        velocity = actions - noise
        t_discretized = (t * self.config.action_num_timestep_buckets).long()

        hidden_states = self._build_inputs(conditioning_tokens, noisy_actions, state, t_discretized)
        pred = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=conditioning_tokens,
            timestep=t_discretized,
        )
        pred_actions = self.action_decoder(pred[:, -actions.shape[1] :])

        if action_is_pad is None:
            action_is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)

        loss = F.mse_loss(pred_actions, velocity, reduction="none")  # [B, T, action_dim]
        valid_mask = ~action_is_pad.unsqueeze(-1)  # [B, T, 1]
        num_valid = valid_mask.sum() * loss.shape[-1]
        return (loss * valid_mask).sum() / num_valid.clamp_min(1)

    @torch.no_grad()
    def predict_action(
        self,
        conditioning_tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = conditioning_tokens.shape[0]
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.config.action_dim,
            dtype=conditioning_tokens.dtype,
            device=conditioning_tokens.device,
        )
        dt = 1.0 / max(self.num_inference_timesteps, 1)
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(max(self.num_inference_timesteps, 1))
            t_value = int(t_cont * self.config.action_num_timestep_buckets)
            timesteps = torch.full(
                (batch_size,), t_value, device=conditioning_tokens.device, dtype=torch.long
            )
            hidden_states = self._build_inputs(conditioning_tokens, actions, state, timesteps)
            pred = self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=conditioning_tokens,
                timestep=timesteps,
            )
            pred_velocity = self.action_decoder(pred[:, -self.action_horizon :])
            actions = actions + dt * pred_velocity
        return actions
