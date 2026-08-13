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
"""Pluggable storage backends for :class:`~lerobot.datasets.lerobot_dataset.LeRobotDataset`.

``LeRobotDataset`` is the single public dataset class, whatever format a dataset
is stored in. The default format — parquet tables plus mp4 videos — is built in.
Alternative formats plug in underneath as *storage backends*: ``LeRobotDataset``
reads ``storage_format`` from ``meta/info.json`` and delegates data access to the
backend registered for that format, keeping metadata, episode selection, and item
semantics identical across formats.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .dataset_metadata import LeRobotDatasetMetadata

DEFAULT_STORAGE_FORMAT = "parquet"

# Formats shipped with lerobot, imported lazily so their optional dependencies
# stay optional. Importing the module runs its ``@register_storage_backend``.
_STORAGE_BACKEND_MODULES = {"lance": "lerobot.datasets.lance_backend"}

# An object-store root (e.g. ``hf://datasets/{repo_id}``) carries no local ``meta/``
# to read the format from before a backend localizes it. Lance is currently the
# only URI-addressable format; revisit if another one appears.
_REMOTE_ROOT_FORMAT = "lance"

_STORAGE_BACKENDS: dict[str, Callable[..., StorageBackend]] = {}


class StorageBackend(Protocol):
    """Read-side data access for one storage format.

    A backend owns row fetching and video decoding for its format and returns
    fully assembled frame dicts — tabular features, delta-timestamp windows,
    padding masks, decoded video frames — identical to the default parquet/mp4
    pipeline's output. ``LeRobotDataset`` delegates ``__getitem__`` and
    ``__getitems__`` to it and keeps everything else (metadata, episode
    selection, the public API). Batches go through :meth:`get_items` so a
    backend can serve them in a single round trip. Instances must be picklable
    so ``DataLoader`` workers can reopen their own connections.
    """

    def __len__(self) -> int: ...

    def get_item(self, idx: int) -> dict: ...

    def get_items(self, indices: list[int]) -> list[dict]: ...

    def set_image_transforms(self, image_transforms: Callable | None) -> None: ...

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None: ...


def is_remote_uri(root: str | Path) -> bool:
    """True for object-store style roots (``hf://…``, ``file://…``, …)."""
    return "://" in str(root)


def register_storage_backend(storage_format: str) -> Callable:
    """Class (or factory) decorator registering a backend for a storage format.

    The registered callable is invoked with the keyword arguments ``meta``,
    ``root``, ``episodes``, ``delta_timestamps``, ``image_transforms``,
    ``tolerance_s``, ``revision``, ``return_uint8`` and ``depth_output_unit``,
    and must return a :class:`StorageBackend`.
    """

    def decorator(factory: Callable[..., StorageBackend]) -> Callable[..., StorageBackend]:
        _STORAGE_BACKENDS[storage_format] = factory
        return factory

    return decorator


def _backend_module(storage_format: str):
    module_name = _STORAGE_BACKEND_MODULES.get(storage_format)
    if module_name is None:
        raise ValueError(
            f"Unknown storage_format {storage_format!r}. Built-in formats: "
            f"{[DEFAULT_STORAGE_FORMAT, *_STORAGE_BACKEND_MODULES]}. Third-party formats must be "
            "registered with `register_storage_backend` before loading the dataset."
        )
    return importlib.import_module(module_name)


def make_storage_backend(storage_format: str, **kwargs) -> StorageBackend:
    """Instantiate the backend registered for ``storage_format``."""
    if storage_format not in _STORAGE_BACKENDS:
        _backend_module(storage_format)
    return _STORAGE_BACKENDS[storage_format](**kwargs)


def localize_remote_root(repo_id: str | None, root: str | Path, revision: str | None = None) -> Path:
    """Materialize ``meta/`` for an object-store dataset and return the local dir holding it.

    Data files are never downloaded — the storage backend reads them in place.
    """
    return _backend_module(_REMOTE_ROOT_FORMAT).localize_root(repo_id, root, revision)


def load_dataset_metadata(
    repo_id: str,
    root: str | Path | None = None,
    revision: str | None = None,
    repo_type: str = "dataset",
) -> LeRobotDatasetMetadata:
    """Load dataset metadata wherever the dataset lives.

    Same as constructing :class:`LeRobotDatasetMetadata` directly, except that a
    remote object-store ``root`` has its ``meta/`` localized first.
    """
    from .dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415  (import cycle)

    if root is not None and is_remote_uri(root):
        root = localize_remote_root(repo_id, root, revision)
    return LeRobotDatasetMetadata(repo_id, root=root, revision=revision, repo_type=repo_type)
