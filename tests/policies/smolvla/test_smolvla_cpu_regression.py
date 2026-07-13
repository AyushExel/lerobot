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

"""Offline CPU regression tests for SmolVLA's real inference entry points.

A tiny local ``SmolVLMConfig`` exercises the real ``predict_action_chunk`` and
``model.sample_actions`` flow-matching paths without Hub access. The seeded output is
checked against a compact frozen numerical signature, and the state-dict key/shape
schema is hashed, so the tests catch deterministic behavior and checkpoint-contract
regressions after the commit that introduced them.
"""

from __future__ import annotations

import hashlib
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from transformers import SmolVLMConfig

import lerobot.policies.smolvla.smolvlm_with_expert as smolvlm_with_expert
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

SEED = 20260709
BATCH_SIZE = 1
STATE_DIM = 4
ACTION_DIM = 4
CHUNK_SIZE = 4
N_ACTION_STEPS = 2
IMAGE_KEY = "observation.images.camera0"
EXPECTED_CHUNK_SIGNATURE = torch.tensor(
    [
        -1.1366491318,
        -0.2570063174,
        -0.0708454847,
        -0.4162012041,
        -1.1738150120,
        0.2739891112,
        -0.1169109046,
        -2.0858731270,
        1.7228362560,
        -0.6142331958,
        1.3207587004,
        0.3407114446,
        -2.1168515682,
        -0.1323032230,
        0.9975754619,
        3.8996689320,
        -2.0858731270,
        1.7228362560,
    ]
)
EXPECTED_STATE_DICT_SCHEMA = "d92be3ed6269d9a1e9b27566396e92ba2a5d958ac4dd8b3f8f55ac34f12ec475"


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


def _state_dict_schema(policy: SmolVLAPolicy) -> str:
    schema = "\n".join(f"{key}:{tuple(value.shape)}" for key, value in sorted(policy.state_dict().items()))
    return hashlib.sha256(schema.encode()).hexdigest()


def _tiny_smolvlm_config() -> SmolVLMConfig:
    return SmolVLMConfig(
        image_token_id=127,
        pad_token_id=0,
        scale_factor=2,
        text_config={
            "model_type": "llama",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 256,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        vision_config={
            "model_type": "smolvlm_vision",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 32,
            "patch_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "attention_dropout": 0.0,
            "layer_norm_eps": 1e-6,
        },
    )


def _make_config() -> SmolVLAConfig:
    return SmolVLAConfig(
        device="cpu",
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        num_vlm_layers=2,
        num_expert_layers=2,
        load_vlm_weights=False,
        resize_imgs_with_padding=(32, 32),
        num_steps=2,
        expert_width_multiplier=1.0,
        max_state_dim=8,
        max_action_dim=8,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )


def _build_policy_and_batch() -> tuple[SmolVLAPolicy, dict[str, Any]]:
    """Build a deterministic tiny policy and batch without touching the Hub."""
    config = _make_config()

    class LocalAutoConfig:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> SmolVLMConfig:
            return _tiny_smolvlm_config()

    class LocalAutoProcessor:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> SimpleNamespace:
            tokenizer = SimpleNamespace(fake_image_token_id=125, global_image_token_id=126)
            return SimpleNamespace(tokenizer=tokenizer)

    set_seed_all(SEED)  # deterministic random init of VLM + expert + projections
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(smolvlm_with_expert, "AutoConfig", LocalAutoConfig)
        monkeypatch.setattr(smolvlm_with_expert, "AutoProcessor", LocalAutoProcessor)
        policy = SmolVLAPolicy(config)
    policy.eval()

    generator = _gen(SEED + 10)
    batch = {
        OBS_STATE: torch.randn(BATCH_SIZE, STATE_DIM, dtype=torch.float32, generator=generator),
        IMAGE_KEY: torch.rand(BATCH_SIZE, 3, 32, 32, dtype=torch.float32, generator=generator),
        ACTION: torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM, dtype=torch.float32, generator=generator),
        OBS_LANGUAGE_TOKENS: torch.randint(3, 120, (BATCH_SIZE, 6), generator=generator),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(BATCH_SIZE, 6, dtype=torch.bool),
    }
    return policy, batch


@pytest.fixture(scope="module")
def shared_policy_and_batch() -> tuple[SmolVLAPolicy, dict[str, Any]]:
    """One tiny policy shared by tests that do not require a fresh instance."""
    return _build_policy_and_batch()


def test_predict_action_chunk_matches_frozen_regression(
    shared_policy_and_batch: tuple[SmolVLAPolicy, dict[str, Any]],
) -> None:
    """The real cached Euler loop retains its seeded numerical contract."""
    policy, batch = shared_policy_and_batch
    noise = torch.randn(
        BATCH_SIZE,
        CHUNK_SIZE,
        policy.config.max_action_dim,
        dtype=torch.float32,
        generator=_gen(SEED + 11),
    )
    with torch.no_grad():
        chunk = policy.predict_action_chunk(dict(batch), noise=noise)

    assert chunk.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    assert chunk.dtype == torch.float32
    assert torch.isfinite(chunk).all()
    torch.testing.assert_close(
        _tensor_signature(chunk),
        EXPECTED_CHUNK_SIGNATURE,
        rtol=1e-5,
        atol=1e-6,
    )


def test_sample_actions_with_explicit_noise_matches_policy_wrapper(
    shared_policy_and_batch: tuple[SmolVLAPolicy, dict[str, Any]],
) -> None:
    """The public policy wrapper delegates to the same cached sampler."""
    policy, batch = shared_policy_and_batch
    max_action_dim = policy.config.max_action_dim
    noise = torch.randn(
        BATCH_SIZE,
        CHUNK_SIZE,
        max_action_dim,
        dtype=torch.float32,
        generator=_gen(SEED + 11),
    )
    with torch.no_grad():
        prep_batch = dict(batch)
        images, img_masks = policy.prepare_images(prep_batch)
        state = policy.prepare_state(prep_batch)
        sampled = policy.model.sample_actions(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            noise=noise.clone(),
        )
        chunk = policy.predict_action_chunk(dict(batch), noise=noise.clone())

    assert sampled.shape == (BATCH_SIZE, CHUNK_SIZE, max_action_dim)
    assert torch.isfinite(sampled).all()
    assert torch.isfinite(chunk).all()
    # predict_action_chunk must delegate to the same sampler: same noise in, cropped actions out.
    assert torch.equal(chunk, sampled[..., :ACTION_DIM])


def test_state_dict_key_shape_schema_is_stable(
    shared_policy_and_batch: tuple[SmolVLAPolicy, dict[str, Any]],
) -> None:
    """Guard the published SmolVLA checkpoint contract across cache refactors."""
    policy, _ = shared_policy_and_batch
    assert len(policy.state_dict()) == 72
    assert _state_dict_schema(policy) == EXPECTED_STATE_DICT_SCHEMA


def test_select_action_queue_consumes_one_chunk(
    shared_policy_and_batch: tuple[SmolVLAPolicy, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """n_action_steps consecutive select_action calls consume exactly one sampled chunk."""
    policy, batch = shared_policy_and_batch
    policy.reset()

    calls = {"count": 0}
    original_sample_actions = policy.model.sample_actions

    def counting_sample_actions(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls["count"] += 1
        return original_sample_actions(*args, **kwargs)

    monkeypatch.setattr(policy.model, "sample_actions", counting_sample_actions)

    set_seed_all(SEED + 1)
    actions = [policy.select_action(dict(batch)) for _ in range(N_ACTION_STEPS)]
    assert calls["count"] == 1, "the first n_action_steps calls must be served by a single chunk"

    actions.append(policy.select_action(dict(batch)))
    assert calls["count"] == 2, "draining the queue must trigger exactly one new chunk computation"

    for action in actions:
        assert action.shape == (BATCH_SIZE, ACTION_DIM)
        assert torch.isfinite(action).all()
