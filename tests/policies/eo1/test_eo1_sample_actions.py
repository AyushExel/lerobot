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

"""CPU regression tests for EO1's real sampling and loss paths. No downloads.

Unlike ``test_eo1.py`` (which stubs the backbone and fakes ``sample_actions``), these
tests build a tiny REAL Qwen2.5-VL backbone locally from a ``Qwen2_5_VLConfig``
(random init, injected by monkeypatching the ``Qwen2_5_VLForConditionalGeneration``
symbol used by ``modeling_eo1``) so the genuine
``EO1VisionFlowMatchingModel.sample_actions`` path runs end-to-end on CPU:
``get_rope_index``, the KV-cache prefix encode, and ``past_key_values.crop`` per
denoise step. ``vlm_config={}`` keeps ``EO1Config`` from downloading anything.

The seeded CPU run is checked against a compact frozen numerical signature with a
small tolerance, and the state-dict key/shape schema is hashed. Unlike rebuilding the
same implementation twice, these assertions survive the commit that introduced them
and catch deterministic numerical changes and checkpoint-key drift.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.eo1.modeling_eo1 import EO1Policy
from lerobot.utils.constants import ACTION, OBS_STATE

SEED = 20260709
BATCH_SIZE = 2
STATE_DIM = 4
ACTION_DIM = 3
CHUNK_SIZE = 4
MAX_STATE_DIM = 6
MAX_ACTION_DIM = 8
NUM_DENOISE_STEPS = 4
STATE_TOKEN_ID = 5
ACTION_TOKEN_ID = 6
EXPECTED_CHUNK_SIGNATURE = torch.tensor(
    [
        0.8466535807,
        -0.4898085594,
        1.2745913267,
        1.8823019266,
        1.3760800362,
        2.4176251888,
        1.8392460346,
        -0.3630225062,
        1.2481100559,
        -0.8001103401,
        1.7478247881,
        0.2530531287,
        4.0029802322,
        0.1667908430,
        1.2649837732,
        6.1214284897,
        -2.2369163036,
        2.4176251888,
    ]
)
EXPECTED_FORWARD_LOSS = 1.4293539524
EXPECTED_STATE_DICT_SCHEMA = "ec101d5a98a3159d795ab7ae75e83d40de74b2668299b38bd97b7c49182f4bfb"


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


def _state_dict_schema(policy: EO1Policy) -> str:
    schema = "\n".join(f"{key}:{tuple(value.shape)}" for key, value in sorted(policy.state_dict().items()))
    return hashlib.sha256(schema.encode()).hexdigest()


def _tiny_qwen_vl_config() -> Qwen2_5_VLConfig:
    """Tiny REAL Qwen2.5-VL config so the genuine rope/KV-cache machinery is exercised."""
    return Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 256,
            "rope_theta": 1000000.0,
            # head_dim = 64/4 = 16 -> mrope sections must sum to 8
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 28,
            "fullatt_block_indexes": [0, 1],
        },
        image_token_id=997,
        video_token_id=998,
        vision_start_token_id=995,
        vision_end_token_id=996,
        attn_implementation="eager",
    )


def _make_config() -> EO1Config:
    return EO1Config(
        device="cpu",
        dtype="float32",
        vlm_base="tiny-random-qwen2.5-vl (patched, no download)",
        vlm_config={},  # non-None -> EO1Config.__post_init__ skips the Hub config download
        chunk_size=CHUNK_SIZE,
        n_action_steps=2,
        max_state_dim=MAX_STATE_DIM,
        max_action_dim=MAX_ACTION_DIM,
        num_denoise_steps=NUM_DENOISE_STEPS,
        attn_implementation="eager",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )


def _build_policy(monkeypatch: pytest.MonkeyPatch) -> EO1Policy:
    """Deterministically build a fresh policy around a fresh tiny random-init backbone."""
    set_seed_all(SEED)  # deterministic random init of the tiny backbone
    tiny_vlm = Qwen2_5_VLForConditionalGeneration(_tiny_qwen_vl_config()).float()

    class _PatchedQwen:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> Qwen2_5_VLForConditionalGeneration:
            return tiny_vlm

    monkeypatch.setattr("lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration", _PatchedQwen)
    set_seed_all(SEED)  # deterministic init of the EO1 flow head (state/action projections)
    policy = EO1Policy(_make_config())
    policy.eval()
    return policy


def _make_batch(include_action: bool) -> dict[str, torch.Tensor | int]:
    """Text-only rollout prompt: [text, STATE, text, ACTION x chunk, text] (pixel_values=None)."""
    input_ids = torch.tensor(
        [[11, STATE_TOKEN_ID, 12] + [ACTION_TOKEN_ID] * CHUNK_SIZE + [13]] * BATCH_SIZE, dtype=torch.long
    )
    seq_len = input_ids.shape[1]
    generator = _gen(SEED + 20)
    batch: dict[str, torch.Tensor | int] = {
        OBS_STATE: torch.rand(BATCH_SIZE, STATE_DIM, dtype=torch.float32, generator=generator),
        "input_ids": input_ids,
        "attention_mask": torch.ones(BATCH_SIZE, seq_len, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(BATCH_SIZE, seq_len, dtype=torch.int32),
        "state_token_id": STATE_TOKEN_ID,
        "action_token_id": ACTION_TOKEN_ID,
    }
    if include_action:
        batch[ACTION] = torch.rand(
            BATCH_SIZE, CHUNK_SIZE, ACTION_DIM, dtype=torch.float32, generator=generator
        )
    return batch


def test_real_sample_actions_end_to_end_shape_and_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real sample_actions path (rope index + KV-cache prefix + crop) runs and is finite."""
    policy = _build_policy(monkeypatch)
    batch = _make_batch(include_action=False)

    with torch.no_grad():
        set_seed_all(SEED + 2)  # seeds model.sample_noise inside predict_action_chunk
        chunk = policy.predict_action_chunk(dict(batch))
    assert chunk.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    assert torch.isfinite(chunk).all()

    with torch.no_grad():
        set_seed_all(SEED + 2)
        direct = policy.model.sample_actions(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            states=policy.prepare_state(batch[OBS_STATE]),
            state_token_id=STATE_TOKEN_ID,
            action_token_id=ACTION_TOKEN_ID,
        )
    assert direct.shape == (BATCH_SIZE, CHUNK_SIZE, MAX_ACTION_DIM)
    assert torch.isfinite(direct).all()
    # predict_action_chunk must delegate to the same sampler: same seed in, cropped actions out.
    assert torch.equal(chunk, direct[..., :ACTION_DIM])
    torch.testing.assert_close(
        _tensor_signature(chunk),
        EXPECTED_CHUNK_SIGNATURE,
        rtol=1e-5,
        atol=1e-6,
    )


def test_state_dict_key_shape_schema_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard the checkpoint contract independently of runtime-generated checkpoints."""
    policy = _build_policy(monkeypatch)
    assert len(policy.state_dict()) == 69
    assert _state_dict_schema(policy) == EXPECTED_STATE_DICT_SCHEMA


def test_forward_loss_matches_frozen_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real training path retains its seeded numerical contract."""
    policy = _build_policy(monkeypatch)
    batch = _make_batch(include_action=True)
    with torch.no_grad():
        set_seed_all(SEED + 3)  # seeds the internal sample_time/sample_noise draws
        loss, loss_dict = policy.forward(dict(batch))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss_dict["loss"] == pytest.approx(loss.item())
    assert loss.item() == pytest.approx(EXPECTED_FORWARD_LOSS, rel=1e-5, abs=1e-6)
