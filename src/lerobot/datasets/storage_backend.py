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
"""Storage backend seam for LeRobotDataset.

`LeRobotDataset` is the single public dataset class; the on-disk/remote format
(Parquet/MP4 today, Lance here, others later) is a `StorageBackend` selected
behind `DatasetReader`. `DatasetReader` keeps the format-independent LeRobot
semantics (episode handling, delta timestamps, transforms, task resolution, the
final sample structure); the backend only retrieves the underlying data.

The one perf-critical contract is `get_items`: a batch of indices resolves to a
single batched read, so backends with columnar/batched access (Lance) keep that
advantage under a standard PyTorch DataLoader:

    DataLoader -> LeRobotDataset.__getitems__ -> DatasetReader.get_items -> backend.get_items
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Retrieval interface a storage format implements to plug in under LeRobotDataset.

    A backend owns nothing LeRobot-semantic; it returns fully-built item dicts for
    a batch of relative indices (the reader's format-independent layer wraps it).
    Kept deliberately small — exactly what DatasetReader delegates — so a new format
    is "implement a backend", not "add a public Dataset class and branches through
    the codebase".
    """

    image_transforms: Callable | None

    @property
    def num_frames(self) -> int: ...

    @property
    def num_episodes(self) -> int: ...

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        """Absolute frame index -> relative row position; None unless episode-filtered."""
        ...

    def get_item(self, idx: int) -> dict:
        """Single-row read (typically get_items([idx])[0])."""
        ...

    def get_items(self, indices: list[int]) -> list[dict]:
        """Batched read — the perf-critical path. One batched fetch per call, not a
        loop of per-item reads, so columnar backends keep their batching win."""
        ...


# The Lance backend is the current map-style reader; it already satisfies the
# StorageBackend contract (get_item/get_items + the metadata properties). Import
# is guarded so `storage_backend` stays importable without the optional lancedb dep.
try:
    from .lancedb_dataset import LanceDBDataset as LanceStorageBackend  # noqa: F401
except Exception:  # pragma: no cover - lancedb is an optional extra
    LanceStorageBackend = None  # type: ignore[assignment,misc]
