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

"""Shared action-head building blocks reused across policies.

Unlike the top-level ``lerobot.policies.common`` modules, modules in this
subpackage may require optional extras (e.g. ``diffusers``) at instantiation
time. Importing them remains safe with base dependencies only: heavy imports
are gated behind ``TYPE_CHECKING``/availability flags and enforced with
``require_package`` at construction time.
"""
