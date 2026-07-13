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

"""Backward compatibility of XVLA checkpoints saved with the old vendored Florence-2 layout.

XVLA used to vendor the Microsoft remote-code Florence-2 implementation; existing Hub
checkpoints (e.g. ``lerobot/xvla-widowx``) store that module layout in their state dicts
(``model.vlm.image_projection``, ``model.vlm.language_model.model.encoder...``,
``...spatial_block.window_attn.fn.qkv...``). Since the policy now builds on the native
``transformers.models.florence2`` implementation, those checkpoints are detected and
remapped at load time by ``_remap_vendored_florence_state_dict``.

These tests fabricate an old-layout state dict from a tiny new-layout policy by applying
the exact inverse renames (the old key patterns below were verified against the real
``lerobot/xvla-widowx`` safetensors index) and assert that detection, remapping, strict
loading, and the full ``from_pretrained`` path all work.
"""

from __future__ import annotations

import re

import pytest
import torch

pytest.importorskip("transformers")

import safetensors.torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import (
    XVLAPolicy,
    _is_vendored_florence_state_dict,
    _remap_vendored_florence_state_dict,
)
from lerobot.utils.constants import ACTION, OBS_STATE

SEED = 20260712
STATE_DIM = 14
ACTION_DIM = 20
CHUNK_SIZE = 8
IMAGE_KEYS = ("observation.images.top", "observation.images.wrist")


def _tiny_florence_config() -> dict:
    """Tiny Florence2 config in the original Microsoft remote-code format (as on the Hub)."""
    return {
        "vision_config": {
            "model_type": "davit",
            "drop_path_rate": 0.0,
            "patch_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "patch_prenorm": [False, True, True, True],
            "dim_embed": [16, 32, 64, 128],
            "num_heads": [2, 4, 8, 16],
            "num_groups": [2, 4, 8, 16],
            "depths": [1, 1, 1, 1],
            "window_size": 2,
            "projection_dim": 64,
            "visual_temporal_embedding": {"type": "COSINE", "max_temporal_embeddings": 8},
            "image_pos_embed": {"type": "learned_abs_2d", "max_pos_embeddings": 16},
            "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
        },
        "text_config": {
            "vocab_size": 128,
            "d_model": 64,
            "encoder_layers": 2,
            "decoder_layers": 1,
            "encoder_ffn_dim": 128,
            "decoder_ffn_dim": 128,
            "encoder_attention_heads": 4,
            "decoder_attention_heads": 4,
            "max_position_embeddings": 128,
            "dropout": 0.1,
        },
        "vocab_size": 128,
        "projection_dim": 64,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
    }


def _make_config() -> XVLAConfig:
    return XVLAConfig(
        device="cpu",
        dtype="float32",
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        florence_config=_tiny_florence_config(),
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_domains=4,
        len_soft_prompts=4,
        dim_time=32,
        max_len_seq=256,
        action_mode="ee6d",
        num_denoising_steps=5,
        use_proprio=True,
        max_state_dim=32,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            IMAGE_KEYS[0]: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            IMAGE_KEYS[1]: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )


def _build_policy(seed: int) -> XVLAPolicy:
    torch.manual_seed(seed)
    policy = XVLAPolicy(_make_config())
    policy.eval()
    return policy


# Inverse of _remap_vendored_florence_state_dict: (new pattern -> old vendored pattern).
# These OLD key patterns match the real `lerobot/xvla-widowx` checkpoint layout.
_INVERSE_RULES = [
    (r"(model\.vlm\.vision_tower\.convs\.\d+\.)conv\.", r"\1proj."),
    (r"(model\.vlm\.vision_tower\.blocks\.\d+\.\d+\.spatial_block\.)norm1\.", r"\1window_attn.norm."),
    (r"(model\.vlm\.vision_tower\.blocks\.\d+\.\d+\.channel_block\.)norm1\.", r"\1channel_attn.norm."),
    (
        r"(model\.vlm\.vision_tower\.blocks\.\d+\.\d+\.(?:spatial_block|channel_block)\.)norm2\.",
        r"\1ffn.norm.",
    ),
    (
        r"(model\.vlm\.vision_tower\.blocks\.\d+\.\d+\.(?:spatial_block|channel_block)\.)"
        r"(window_attn|channel_attn)\.(qkv|proj)\.",
        r"\1\2.fn.\3.",
    ),
    (
        r"(model\.vlm\.vision_tower\.blocks\.\d+\.\d+\.(?:spatial_block|channel_block)\.)ffn\.(fc1|fc2)\.",
        r"\1ffn.fn.net.\2.",
    ),
    (
        r"(model\.vlm\.vision_tower\.blocks\.\d+\.\d+\.(?:spatial_block|channel_block)\.)(conv1|conv2)\.",
        r"\1\2.fn.dw.",
    ),
    (r"model\.vlm\.multi_modal_projector\.image_proj_norm\.", r"model.vlm.image_proj_norm."),
    (r"model\.vlm\.multi_modal_projector\.image_position_embed\.", r"model.vlm.image_pos_embed."),
    (
        r"model\.vlm\.multi_modal_projector\.visual_temporal_embed\.",
        r"model.vlm.visual_temporal_embed.",
    ),
    (r"model\.vlm\.language_model\.", r"model.vlm.language_model.model."),
]


def _to_old_vendored_layout(new_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a new-layout state dict into the old vendored checkpoint layout."""
    old: dict[str, torch.Tensor] = {}
    vocab_size = new_state_dict["model.vlm.language_model.shared.weight"].shape[0]
    for key, value in new_state_dict.items():
        if key == "model.vlm.language_model.shared.weight":
            # the real checkpoints only store encoder.embed_tokens.weight (safetensors
            # deduplicated the tied alias on save)
            continue
        if key == "model.vlm.multi_modal_projector.image_projection.weight":
            # old layout stored the raw matmul parameter, transposed w.r.t. nn.Linear
            old["model.vlm.image_projection"] = value.transpose(0, 1).contiguous()
            continue
        new_key = key
        for pattern, replacement in _INVERSE_RULES:
            new_key, count = re.subn(pattern, replacement, new_key, count=1)
            if count:
                break
        old[new_key] = value.clone()
    # generation-only buffer that the vendored language model registered
    old["model.vlm.language_model.final_logits_bias"] = torch.zeros(1, vocab_size)
    return old


def test_vendored_layout_detection() -> None:
    policy = _build_policy(SEED)
    new_sd = policy.state_dict()
    assert not _is_vendored_florence_state_dict(new_sd)

    old_sd = _to_old_vendored_layout(new_sd)
    assert _is_vendored_florence_state_dict(old_sd)


def test_remap_old_layout_loads_strict_and_preserves_weights() -> None:
    donor = _build_policy(SEED)
    old_sd = _to_old_vendored_layout(donor.state_dict())

    remapped = _remap_vendored_florence_state_dict(old_sd)
    # from_pretrained restores the tied shared/embed_tokens alias after the remap
    remapped["model.vlm.language_model.shared.weight"] = remapped[
        "model.vlm.language_model.encoder.embed_tokens.weight"
    ]

    target = _build_policy(SEED + 1)
    target.load_state_dict(remapped, strict=True)

    donor_sd = donor.state_dict()
    for key, value in target.state_dict().items():
        assert torch.equal(value, donor_sd[key]), f"weight mismatch after remap round-trip: {key}"


def test_from_pretrained_loads_old_vendored_checkpoint(tmp_path) -> None:
    donor = _build_policy(SEED)
    old_sd = _to_old_vendored_layout(donor.state_dict())

    donor.config._save_pretrained(tmp_path)
    safetensors.torch.save_file(old_sd, str(tmp_path / "model.safetensors"))

    loaded = XVLAPolicy.from_pretrained(str(tmp_path))
    donor_sd = donor.state_dict()
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, donor_sd[key]), f"weight mismatch via from_pretrained: {key}"


def test_from_pretrained_roundtrip_new_layout(tmp_path) -> None:
    donor = _build_policy(SEED)
    donor.save_pretrained(tmp_path)

    loaded = XVLAPolicy.from_pretrained(str(tmp_path))
    donor_sd = donor.state_dict()
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, donor_sd[key]), f"weight mismatch after save/load round-trip: {key}"
