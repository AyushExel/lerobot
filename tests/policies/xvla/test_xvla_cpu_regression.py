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

"""CPU regression tests for XVLA's real inference and loss paths. No downloads.

``florence_config`` is given tiny ``vision_config``/``text_config`` dicts so the whole
Florence2 backbone (the native ``transformers.models.florence2`` implementation) is built
locally with random init, and ``OBS_LANGUAGE_TOKENS`` is provided directly so no tokenizer
is instantiated. ``predict_action_chunk`` therefore runs the REAL flow-matching Euler loop
end-to-end on CPU. The tiny config is written in the original Microsoft remote-code format
on purpose: it exercises the same MS->native config translation that existing Hub
checkpoints go through.

The vendored-to-native mapping is covered separately by
``test_xvla_checkpoint_compat.py``. Here, the native implementation is guarded by a
compact frozen numerical signature, a frozen loss, and a hash of the state-dict key/shape
schema. Those checks remain meaningful after this commit, unlike comparing two fresh
instances of the same implementation to each other.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE

SEED = 20260709
BATCH_SIZE = 2
CHUNK_SIZE = 8
STATE_DIM = 14
ACTION_DIM = 20  # ee6d action space model dim
IMAGE_KEYS = ("observation.images.top", "observation.images.wrist")
EXPECTED_CHUNK_SIGNATURE = torch.tensor(
    [
        -0.1862837076,
        0.0543350130,
        0.3003459871,
        0.4870811105,
        -0.1783014685,
        0.1781928539,
        -0.3613993227,
        0.4825531244,
        0.2864531577,
        0.4261035621,
        0.5686659813,
        0.0573174208,
        41.8987503052,
        0.1309335977,
        0.3302442133,
        6.3463764191,
        -0.8187647462,
        0.6641827822,
    ]
)
EXPECTED_FORWARD_LOSS = 270.4609680176
EXPECTED_STATE_DICT_SCHEMA = "8bedd2f255eeb676de9e9fab8b37a2ac7f62a79c8c02e07e3099491ee57ced7b"


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _gen(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _tensor_signature(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.flatten()
    summaries = torch.stack(
        [tensor.sum(), tensor.mean(), tensor.std(), tensor.norm(), tensor.min(), tensor.max()]
    )
    return torch.cat([flat[:12], summaries]).cpu()


def _state_dict_schema(policy: XVLAPolicy) -> str:
    schema = "\n".join(f"{key}:{tuple(value.shape)}" for key, value in sorted(policy.state_dict().items()))
    return hashlib.sha256(schema.encode()).hexdigest()


def _tiny_florence_config() -> dict:
    """Tiny Florence2: 64x64 images -> DaViT stages /4,/2,/2,/2 -> 2x2 (square) final feature map."""
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
            "d_model": 64,  # must equal projection_dim (encoder output feeds the transformer head)
            "encoder_layers": 2,
            "decoder_layers": 1,  # the decoder is deleted by XVLAModel right after init
            "encoder_ffn_dim": 128,
            "decoder_ffn_dim": 128,
            "encoder_attention_heads": 4,
            "decoder_attention_heads": 4,
            "max_position_embeddings": 128,
            "dropout": 0.1,  # inactive: the tests run in eval mode
        },
        # MS-format top-level keys, as stored in existing Hub checkpoints; vocab_size and
        # projection_dim are ignored by the MS->native translation (the native config reads
        # them from text_config/vision_config), the token ids are passed through.
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


def _build_policy() -> XVLAPolicy:
    """Deterministically build a fresh random-init policy."""
    set_seed_all(SEED)
    policy = XVLAPolicy(_make_config())
    policy.eval()
    return policy


def _make_batch() -> dict[str, torch.Tensor]:
    generator = _gen(SEED + 41)
    return {
        OBS_LANGUAGE_TOKENS: torch.randint(4, 120, (BATCH_SIZE, 10), generator=generator),
        IMAGE_KEYS[0]: torch.rand(BATCH_SIZE, 3, 64, 64, dtype=torch.float32, generator=generator),
        IMAGE_KEYS[1]: torch.rand(BATCH_SIZE, 3, 64, 64, dtype=torch.float32, generator=generator),
        OBS_STATE: torch.rand(BATCH_SIZE, STATE_DIM, dtype=torch.float32, generator=generator),
        ACTION: torch.rand(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM, dtype=torch.float32, generator=generator),
    }


def test_predict_action_chunk_matches_frozen_regression() -> None:
    """The real flow-matching loop retains its seeded numerical contract."""
    policy = _build_policy()
    batch = _make_batch()
    with torch.no_grad():
        set_seed_all(SEED + 4)  # seeds the internal noise of the flow-matching Euler loop
        chunk = policy.predict_action_chunk(dict(batch))

    assert chunk.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    assert chunk.dtype == torch.float32
    assert torch.isfinite(chunk).all()
    torch.testing.assert_close(
        _tensor_signature(chunk),
        EXPECTED_CHUNK_SIGNATURE,
        rtol=1e-5,
        atol=1e-6,
    )


def test_forward_loss_matches_frozen_regression() -> None:
    """The real training objective retains its seeded numerical contract."""
    policy = _build_policy()
    batch = _make_batch()
    with torch.no_grad():
        set_seed_all(SEED + 5)  # seeds the internal time/noise draws of the training objective
        loss, _ = policy.forward(dict(batch))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(EXPECTED_FORWARD_LOSS, rel=1e-5, abs=1e-5)


def test_state_dict_key_shape_schema_is_stable() -> None:
    """Guard the native Florence checkpoint contract independently of round-trips."""
    policy = _build_policy()
    assert len(policy.state_dict()) == 223
    assert _state_dict_schema(policy) == EXPECTED_STATE_DICT_SCHEMA
