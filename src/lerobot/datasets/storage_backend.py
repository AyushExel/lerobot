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

`LeRobotDataset` stays the single public dataset class. A storage backend owns
only data retrieval (batched row reads and batched video decoding); everything
format-independent — episode handling, delta windows and padding, transforms,
task resolution, the final sample structure — lives in `BackendDatasetReader`:

    DataLoader -> LeRobotDataset.__getitems__ -> BackendDatasetReader.get_items
               -> backend.get_rows(...) + backend.get_video_frames(...)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import datasets
import numpy as np
import torch

from .dataset_reader import DatasetReader
from .depth_utils import dequantize_depth
from .io_utils import hf_transform_to_torch

if TYPE_CHECKING:
    from .dataset_metadata import LeRobotDatasetMetadata

# A video file's identity in LeRobot metadata: (video_key, chunk_index, file_index).
VideoFileKey = tuple[str, int, int]


@runtime_checkable
class StorageBackend(Protocol):
    """Retrieval interface a storage format implements to plug in under LeRobotDataset.

    A backend holds no LeRobot semantics: it returns raw batched data and the
    reader builds the samples. ``meta`` is the standard dataset metadata the
    backend transported (e.g. materialized from an object store).
    """

    meta: LeRobotDatasetMetadata

    def get_rows(self, rows: list[int]) -> dict[str, np.ndarray | list]:
        """One batched columnar read of the given absolute frame rows.

        Returns one entry per tabular feature: numpy arrays (2D for vector
        features), or python lists for string/language features.
        """
        ...

    def get_video_frames(
        self,
        requests: dict[VideoFileKey, list[tuple[Hashable, list[float]]]],
        *,
        tolerance_s: float,
        return_uint8: bool,
    ) -> dict[Hashable, torch.Tensor]:
        """Batched video decode: per file, a list of (request_id, timestamps).

        Timestamps are relative to the video file (already shifted by the
        episode's ``from_timestamp``). Returns decoded frame stacks keyed by
        the opaque request_id; depth streams are returned raw (undequantized),
        mirroring :func:`lerobot.datasets.video_utils.decode_video_frames`.
        """
        ...

    def prefetch_videos(self, windows: dict[VideoFileKey, list[tuple[int, int]]]) -> None:
        """Optional overlap hint: frame spans the next get_video_frames will need.

        Lets a remote backend start fetching video bytes while the caller is
        still reading rows. Correctness must not depend on it (no-op is valid).
        """
        ...


class BackendDatasetReader(DatasetReader):
    """DatasetReader over a StorageBackend.

    Owns the format-independent semantics once, for every backend: episode
    handling, delta windows and padding, timestamp shifting, depth
    dequantization, image transforms, task resolution, and final sample
    assembly. The backend is only asked for raw rows and decoded frames.
    ``hf_dataset`` stays None: the backend reads its own storage lazily
    per worker.
    """

    def __init__(self, backend: StorageBackend, **kwargs):
        super().__init__(**kwargs)
        self._backend = backend
        meta = self._meta
        if self.episodes is not None:
            self.episodes = sorted(self.episodes)

        self._ep_from = self._episode_numpy("dataset_from_index", np.int64)
        self._ep_to = self._episode_numpy("dataset_to_index", np.int64)
        # episode ranges must tile [0, total_frames) exactly
        if len(self._ep_from) and (
            int(self._ep_from[0]) != 0
            or int(self._ep_to[-1]) != meta.total_frames
            or (self._ep_from[1:] != self._ep_to[:-1]).any()
        ):
            raise ValueError(
                "Episode boundaries in meta do not tile [0, total_frames) contiguously; "
                "the dataset metadata is inconsistent."
            )

        if self.episodes is not None:
            self._rel_to_abs = np.concatenate(
                [np.arange(self._ep_from[ep], self._ep_to[ep]) for ep in self.episodes]
            )
            self._absolute_to_relative_idx = {
                int(abs_idx): rel_idx for rel_idx, abs_idx in enumerate(self._rel_to_abs)
            }
        else:
            self._rel_to_abs = None

        self._task_names = list(meta.tasks.index)
        self._tabular_keys = [
            key for key in meta.features if key not in meta.video_keys and key not in meta.image_keys
        ]
        self._feature_shapes = {
            key: tuple(meta.features[key].get("shape") or ()) for key in self._tabular_keys
        }
        # String features pass through as python strings, language columns
        # (list<struct>) as python lists of dicts, like the Parquet path.
        self._string_keys = {key for key in self._tabular_keys if meta.features[key].get("dtype") == "string"}
        self._language_keys = {
            key for key in self._tabular_keys if meta.features[key].get("dtype") == "language"
        }
        # Which (chunk, file) mp4 holds each episode and where it starts inside
        # it (episodes share files in v3.0; timestamps shift by from_timestamp).
        self._video_locator = {
            key: (
                self._episode_numpy(f"videos/{key}/chunk_index", np.int64),
                self._episode_numpy(f"videos/{key}/file_index", np.int64),
                self._episode_numpy(f"videos/{key}/from_timestamp", np.float64),
            )
            for key in meta.video_keys
        }

    def _episode_numpy(self, name: str, dtype: type[np.generic]) -> np.ndarray:
        # Read straight from the underlying Arrow column, not HF Dataset __getitem__
        column = self._meta.episodes.data.column(name).to_numpy(zero_copy_only=False)
        return column.astype(dtype, copy=False)

    # ── lifecycle ─────────────────────────────────────────────────────

    def try_load(self) -> bool:
        return True

    def load_and_activate(self) -> None:
        """Materialize the tabular data as a real HF dataset (compatibility path).

        The batched read path never calls this. It exists so existing consumers
        of ``hf_dataset`` (replay's select_columns, visualization and stats
        tools, ...) work unchanged on backend datasets: one batched read of the
        same tabular table the Parquet path loads eagerly at init.
        """
        if self.hf_dataset is not None:
            return
        rows = (
            [int(row) for row in self._rel_to_abs]
            if self._rel_to_abs is not None
            else list(range(self._meta.total_frames))
        )
        columns = self._backend.get_rows(rows)
        out = {}
        for key in self._tabular_keys:
            data = columns[key]
            shape = self._feature_shapes[key]
            if isinstance(data, np.ndarray) and len(shape) > 1:
                data = data.reshape(len(data), *shape)
            # lists of 1-D numpy rows keep the storage dtype through Arrow
            out[key] = list(data) if getattr(data, "ndim", 1) > 1 else data
        view = datasets.Dataset.from_dict(out)
        view.set_transform(hf_transform_to_torch)
        self.hf_dataset = view

    # ── counts and index mapping ──────────────────────────────────────

    @property
    def num_frames(self) -> int:
        return len(self._rel_to_abs) if self._rel_to_abs is not None else self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        return self._absolute_to_relative_idx

    # ── batched read path ─────────────────────────────────────────────

    def get_item(self, idx) -> dict:
        return self.get_items([idx])[0]

    def get_items(self, indices: list[int]) -> list[dict]:
        """Batched fetch: one backend row read and one backend video decode per batch."""
        plans = self._plan_batch(indices)
        rows = sorted({row for plan in plans for row in plan["rows"]})
        row_pos = {row: pos for pos, row in enumerate(rows)}

        # The video byte prefetch needs only episode metadata, so it overlaps
        # the row fetch below.
        if self._meta.video_keys:
            self._backend.prefetch_videos(self._plan_file_windows(plans))

        columns = self._backend.get_rows(rows)
        items = [self._build_item(plan, columns, row_pos) for plan in plans]

        if self._meta.video_keys:
            requests = self._build_video_requests(plans, columns["timestamp"], row_pos)
            decoded = self._backend.get_video_frames(
                requests, tolerance_s=self._tolerance_s, return_uint8=self._return_uint8
            )
            for (sample_idx, key), frames in decoded.items():
                if key in self._meta.depth_keys:
                    config = self._depth_encoder_configs[key]
                    frames = dequantize_depth(
                        frames,
                        depth_min=config.depth_min,
                        depth_max=config.depth_max,
                        shift=config.shift,
                        use_log=config.use_log,
                        output_unit=self._depth_output_unit,
                    )
                items[sample_idx][key] = frames.squeeze(0)

        if self._image_transforms is not None:
            for item in items:
                for cam_key in self._meta.camera_keys:
                    if cam_key in self._meta.depth_keys:
                        continue
                    item[cam_key] = self._image_transforms(item[cam_key])
        return items

    def _plan_batch(self, indices: list[int]) -> list[dict]:
        """Resolve each sample to the absolute rows it needs and its padding masks."""
        plans = []
        for idx in indices:
            abs_idx = self._resolve_abs_idx(idx)
            ep_idx = self._episode_index_for_abs_idx(abs_idx)
            start, end = self._episode_bounds(ep_idx)
            plan = {"abs_idx": abs_idx, "ep_idx": ep_idx, "rows": {abs_idx}, "windows": {}, "padding": {}}
            if self.delta_indices is not None:
                for key, deltas in self.delta_indices.items():
                    window = [min(max(abs_idx + delta, start), end - 1) for delta in deltas]
                    plan["windows"][key] = window
                    plan["rows"].update(window)
                    plan["padding"][f"{key}_is_pad"] = torch.BoolTensor(
                        [not (start <= abs_idx + delta < end) for delta in deltas]
                    )
            plans.append(plan)
        return plans

    def _resolve_abs_idx(self, idx: int) -> int:
        return int(self._rel_to_abs[idx]) if self._rel_to_abs is not None else int(idx)

    def _episode_index_for_abs_idx(self, abs_idx: int) -> int:
        return int(np.searchsorted(self._ep_from, abs_idx, side="right") - 1)

    def _episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        return int(self._ep_from[ep_idx]), int(self._ep_to[ep_idx])

    def _video_file_key(self, key: str, ep_idx: int) -> VideoFileKey:
        chunk_arr, file_arr, _ = self._video_locator[key]
        return key, int(chunk_arr[ep_idx]), int(file_arr[ep_idx])

    def _plan_file_windows(self, plans: list[dict]) -> dict[VideoFileKey, list[tuple[int, int]]]:
        """Map each batch sample's video windows to (file key -> frame spans).

        Positions come from episode metadata alone, so this runs before any row
        is fetched; the decode stage re-derives exact ranges from real
        timestamps and fetches anything missed.
        """
        fps = float(self._meta.fps)
        windows: dict[VideoFileKey, list[tuple[int, int]]] = {}
        for plan in plans:
            ep_idx = plan["ep_idx"]
            ep_start, _ = self._episode_bounds(ep_idx)
            for key in self._meta.video_keys:
                _, _, from_ts_arr = self._video_locator[key]
                window = plan["windows"].get(key, [plan["abs_idx"]])
                base = round(float(from_ts_arr[ep_idx]) * fps) - ep_start
                first = base + min(window) - (1 if key in self._meta.depth_keys else 0)
                windows.setdefault(self._video_file_key(key, ep_idx), []).append((first, base + max(window)))
        return windows

    def _build_video_requests(
        self,
        plans: list[dict],
        timestamps: np.ndarray,
        row_pos: dict[int, int],
    ) -> dict[VideoFileKey, list[tuple[Hashable, list[float]]]]:
        requests: dict[VideoFileKey, list[tuple[Hashable, list[float]]]] = defaultdict(list)
        for sample_idx, plan in enumerate(plans):
            ep_idx = plan["ep_idx"]
            for key in self._meta.video_keys:
                _, _, from_ts_arr = self._video_locator[key]
                base = float(from_ts_arr[ep_idx])
                window = plan["windows"].get(key, [plan["abs_idx"]])
                shifted_ts = [base + float(timestamps[row_pos[row]]) for row in window]
                requests[self._video_file_key(key, ep_idx)].append(((sample_idx, key), shifted_ts))
        return requests

    def _tabular_value(self, key: str, data, pos: int, windows: dict | None = None, row_pos=None):
        """One item's value for a tabular feature (window-aware when given)."""
        shape = self._feature_shapes[key]
        if key in self._language_keys:
            return data[pos]
        if key in self._string_keys:
            value = data[pos]
            return value if isinstance(value, str) or value is None else str(value)
        if windows is not None and key in windows:
            window = data[[row_pos[row] for row in windows[key]]]
            if len(shape) > 1:
                window = window.reshape(len(window), *shape)
            return torch.from_numpy(np.ascontiguousarray(window))
        if getattr(data, "ndim", 1) > 1:
            value = data[pos]
            if len(shape) > 1:
                value = value.reshape(shape)
            return torch.from_numpy(value.copy())
        return torch.tensor(data[pos])

    def _build_item(self, plan: dict, columns: dict, row_pos: dict[int, int]) -> dict:
        base = row_pos[plan["abs_idx"]]
        item = {
            key: self._tabular_value(key, columns[key], base, plan["windows"], row_pos)
            for key in self._tabular_keys
        }
        item.update(plan["padding"])
        item["task"] = self._task_names[int(item["task_index"].item())]
        return item


def make_storage_backend(storage_format: str, **kwargs) -> StorageBackend:
    """Registry mapping a non-default ``storage_format`` to its backend implementation."""
    if storage_format == "lance":
        from .lancedb_dataset import LanceStorageBackend  # noqa: PLC0415 - optional lancedb extra

        return LanceStorageBackend(**kwargs)
    raise ValueError(f"No storage backend registered for storage_format={storage_format!r}.")


def make_backend_metadata(storage_format: str, repo_id: str | None, root, revision: str | None):
    """Metadata via the backend's transport (e.g. from an object-store URI the default
    loader cannot reach). Returns None for the default Parquet/MP4 loader."""
    if storage_format == "lance":
        from .lancedb_dataset import lance_metadata  # noqa: PLC0415 - optional lancedb extra

        return lance_metadata(repo_id, root, revision)
    return None
