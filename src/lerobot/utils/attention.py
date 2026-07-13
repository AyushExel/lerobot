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

"""Shared helpers for loading optional attention backends."""

from collections.abc import Callable


def load_flash_attn_func(
    unavailable_hint: str = "Install it, or use the default torch attention backend.",
) -> Callable:
    """Load ``flash_attn_func``, preferring the FA3 ``flash_attn_interface`` package.

    Falls back to the classic ``flash_attn`` package, and raises ImportError if neither
    is installed. ``unavailable_hint`` lets callers name the exact config knob their
    users should change instead of the generic remediation.
    """
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        try:
            from flash_attn import flash_attn_func
        except ImportError as e:
            raise ImportError(
                f"Flash attention was requested but the `flash_attn` package is not installed. "
                f"{unavailable_hint}"
            ) from e
    return flash_attn_func
