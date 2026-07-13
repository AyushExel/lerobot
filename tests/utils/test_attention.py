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

import sys
from types import ModuleType

import pytest

from lerobot.utils.attention import load_flash_attn_func


def _fake_flash_module(name: str, func) -> ModuleType:
    module = ModuleType(name)
    module.flash_attn_func = func
    return module


def test_load_flash_attn_func_prefers_fa3(monkeypatch: pytest.MonkeyPatch) -> None:
    def fa3_func():
        return "fa3"

    def fa2_func():
        return "fa2"

    monkeypatch.setitem(
        sys.modules, "flash_attn_interface", _fake_flash_module("flash_attn_interface", fa3_func)
    )
    monkeypatch.setitem(sys.modules, "flash_attn", _fake_flash_module("flash_attn", fa2_func))

    assert load_flash_attn_func() is fa3_func


def test_load_flash_attn_func_falls_back_to_fa2(monkeypatch: pytest.MonkeyPatch) -> None:
    def fa2_func():
        return "fa2"

    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    monkeypatch.setitem(sys.modules, "flash_attn", _fake_flash_module("flash_attn", fa2_func))

    assert load_flash_attn_func() is fa2_func


def test_load_flash_attn_func_reports_caller_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    monkeypatch.setitem(sys.modules, "flash_attn", None)

    with pytest.raises(ImportError, match="disable use_flash_attention"):
        load_flash_attn_func("Install flash-attn or disable use_flash_attention.")
