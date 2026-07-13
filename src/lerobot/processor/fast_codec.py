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

"""Canonical FAST action-token codec (DCT + BPE + PaliGemma vocab-offset mapping).

FAST (https://huggingface.co/physical-intelligence/fast) encodes continuous action
chunks as ``dct -> quantize -> chr() -> BPE`` and maps the resulting action-token ids
into the tail of the PaliGemma vocabulary. These helpers are the canonical versions of
logic that historically was copy-pasted between ``pi0_fast`` and the
``ActionTokenizerProcessorStep`` (and is vendored, untouched, in
``molmoact2.molmoact2_hf_model.action_tokenizer`` for Hub remote-code parity).

The scipy import is kept lazy because scipy is an optional dependency.
"""

import logging

import numpy as np
import torch


def fast_paligemma_token_offset(
    tokens: torch.Tensor, *, vocab_size: int, fast_skip_tokens: int
) -> torch.Tensor:
    """Map FAST action-token ids <-> PaliGemma token ids.

    FAST action tokens are stored in the tail of the PaliGemma vocabulary, skipping the
    last ``fast_skip_tokens`` special tokens. The mapping
    ``t -> vocab_size - 1 - fast_skip_tokens - t`` is an involution, so the same
    function converts in both directions.

    Args:
        tokens: Action-token ids (or PaliGemma token ids) to convert.
        vocab_size: Size of the PaliGemma vocabulary.
        fast_skip_tokens: Number of tokens at the end of the vocabulary to skip.

    Returns:
        The converted token ids.
    """
    return vocab_size - 1 - fast_skip_tokens - tokens


def decode_actions_with_fast(
    token_ids: list[int],
    *,
    bpe_tokenizer,
    min_token: int,
    scale: float,
    time_horizon: int,
    action_dim: int,
    relaxed_decoding: bool = True,
) -> np.ndarray:
    """Decodes action token IDs back to continuous action values using the FAST tokenizer.

    Per sequence: BPE decode -> ``ord()`` -> ``+ min_token`` -> (relaxed: pad/truncate to
    ``time_horizon * action_dim``) -> ``idct(coeff / scale, norm="ortho")``. Sequences that
    fail to decode fall back to zeros.

    Args:
        token_ids: List of token-id sequences to decode.
        bpe_tokenizer: The FAST BPE tokenizer (e.g. ``action_tokenizer.bpe_tokenizer``).
        min_token: Quantization offset of the FAST tokenizer (``action_tokenizer.min_token``).
        scale: DCT coefficient scale of the FAST tokenizer (``action_tokenizer.scale``).
        time_horizon: The number of timesteps for actions.
        action_dim: The dimensionality of each action.
        relaxed_decoding: Whether to use relaxed decoding (allows partial sequences).

    Returns:
        A numpy array of decoded actions with shape ``(len(token_ids), time_horizon, action_dim)``.
    """
    from scipy.fftpack import idct

    decoded_actions = []

    for token in token_ids:
        try:
            decoded_tokens = bpe_tokenizer.decode(token)
            decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + min_token

            if relaxed_decoding:
                # expected sequence length
                expected_seq_len = time_horizon * action_dim
                diff = expected_seq_len - decoded_dct_coeff.shape[0]

                # apply truncation if too long
                if diff < 0:
                    decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]  # truncate on the right

                # apply padding if too short
                elif diff > 0:
                    decoded_dct_coeff = np.pad(
                        decoded_dct_coeff, (0, diff), mode="constant", constant_values=0
                    )

            decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
            assert decoded_dct_coeff.shape == (
                time_horizon,
                action_dim,
            ), (
                f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({time_horizon}, {action_dim})"
            )

        except Exception as e:
            logging.warning(f"Error decoding tokens: {e}")
            logging.warning(f"Tokens: {token}")
            decoded_dct_coeff = np.zeros((time_horizon, action_dim))

        decoded_actions.append(idct(decoded_dct_coeff / scale, axis=0, norm="ortho"))

    return np.stack(decoded_actions)
