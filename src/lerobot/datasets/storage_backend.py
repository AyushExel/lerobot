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

`LeRobotDataset` stays the single public dataset class; the storage format
(Parquet/MP4, Lance, ...) plugs in as a `StorageBackend` behind the reader:

    DataLoader -> LeRobotDataset.__getitems__ -> reader.get_items -> backend.get_items
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .dataset_reader import DatasetReader


@runtime_checkable
class StorageBackend(Protocol):
    """What a storage format implements to plug in under LeRobotDataset.

    A backend returns fully-built item dicts for relative indices; everything
    format-independent stays in the reader/metadata layer.
    """

    image_transforms: Callable | None

    @property
    def num_frames(self) -> int: ...

    @property
    def num_episodes(self) -> int: ...

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None: ...

    def get_item(self, idx: int) -> dict: ...

    def get_items(self, indices: list[int]) -> list[dict]:
        """Batched read — the perf-critical path: one fetch per call, not per item."""
        ...


class BackendDatasetReader(DatasetReader):
    """DatasetReader whose retrieval is served by a StorageBackend.

    The backend opens its own tables lazily per worker, so there is nothing to
    load; ``hf_dataset`` stays None.
    """

    def __init__(self, backend: StorageBackend, **kwargs):
        super().__init__(**kwargs)
        self._backend = backend

    def try_load(self) -> bool:
        return True

    def load_and_activate(self) -> None:
        pass

    @property
    def num_frames(self) -> int:
        return self._backend.num_frames

    @property
    def num_episodes(self) -> int:
        return self._backend.num_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        return self._backend.absolute_to_relative_idx

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        super().set_image_transforms(image_transforms)
        self._backend.image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        self.set_image_transforms(None)

    def get_item(self, idx: int) -> dict:
        return self._backend.get_item(idx)

    def get_items(self, indices: list[int]) -> list[dict]:
        return self._backend.get_items(indices)


try:
    from .lancedb_dataset import LanceDBDataset as LanceStorageBackend  # noqa: F401
except Exception:  # pragma: no cover - lancedb is an optional extra
    LanceStorageBackend = None  # type: ignore[assignment,misc]
