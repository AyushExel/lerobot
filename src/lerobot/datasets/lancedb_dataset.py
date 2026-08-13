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

from __future__ import annotations

import bisect
import io
import json
import multiprocessing
import os
import re
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import av
import huggingface_hub
import numpy as np
import torch

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import _lancedb_available, require_package

if TYPE_CHECKING or _lancedb_available:
    import lancedb
    from lancedb.permutation import Permutation

from .dataset_metadata import LeRobotDatasetMetadata
from .utils import resolve_storage_format
from .video_utils import FrameTimestampError, decode_video_frames_pyav

FRAMES_TABLE = "frames"
VIDEOS_TABLE = "videos"
META_TABLE = "meta"
VIDEO_BLOB_COLUMN = "video_bytes"
# Byte-index columns on the videos table: map a frame window to its byte ranges so a
# batch's video fetch can be batched. Assume constant frame rate; mp4-only.
VIDEO_INDEX_COLUMNS = ("file_size", "moov_offset", "moov_size", "kf_indices", "kf_positions")
# ffmpeg reads more bytes than the frames requested. Padding each prefetched range
# to cover those known reads keeps them off the slow fallback path.
_OPEN_PROBE_BYTES = 256 * 1024
_RANGE_SLACK = 64 * 1024


def _merge_spans(spans: list[tuple[int, int]], gap: int = _RANGE_SLACK) -> list[tuple[int, int]]:
    """Coalesce overlapping or nearby byte ranges into fewer, larger requests."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _find_moov(read_at, file_size: int) -> tuple[int, int]:
    """Locate the mp4 ``moov`` box by walking top-level box headers."""
    offset = 0
    while offset < file_size:
        header = read_at(offset, 16)
        box_size = int.from_bytes(header[:4], "big")
        box_type = header[4:8]
        if box_size == 1:
            box_size = int.from_bytes(header[8:16], "big")
        elif box_size == 0:
            box_size = file_size - offset
        if box_type == b"moov":
            return offset, box_size
        offset += box_size
    raise ValueError("no moov box found")


def build_video_byte_index(path: str | Path) -> dict:
    """Compute the byte-index columns for one video file. mp4-only"""
    path = Path(path)
    file_size = path.stat().st_size
    kf_entries = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for packet in container.demux(stream):
            if packet.pts is None or not packet.is_keyframe or packet.pos is None:
                continue
            kf_entries.append((round(float(packet.pts * packet.time_base) * fps), packet.pos))
    kf_entries.sort()
    with open(path, "rb") as f:

        def read_at(offset: int, length: int) -> bytes:
            f.seek(offset)
            return f.read(length)

        moov_offset, moov_size = _find_moov(read_at, file_size)
    return {
        "file_size": file_size,
        "moov_offset": moov_offset,
        "moov_size": moov_size,
        "kf_indices": [index for index, _ in kf_entries],
        "kf_positions": [position for _, position in kf_entries],
    }


class _SparseBlobSource(io.RawIOBase):
    """Adapter between range fetches and the decoders' file API."""

    def __init__(self, size: int, fallback):
        super().__init__()
        self._size = size
        self._fallback = fallback
        self._starts: list[int] = []
        self._chunks: list[bytes] = []
        self._pos = 0
        self.buffered = 0
        self.fallback_bytes = 0

    def add(self, offset: int, data: bytes) -> None:
        end = offset + len(data)
        lo = bisect.bisect_left(self._starts, offset)
        if lo > 0 and self._starts[lo - 1] + len(self._chunks[lo - 1]) >= offset:
            lo -= 1
        hi = bisect.bisect_right(self._starts, end)
        if lo == hi:
            self._starts.insert(lo, offset)
            self._chunks.insert(lo, data)
        else:
            merged_start = min(offset, self._starts[lo])
            merged_end = max(end, max(self._starts[i] + len(self._chunks[i]) for i in range(lo, hi)))
            merged = bytearray(merged_end - merged_start)
            for i in range(lo, hi):
                at = self._starts[i] - merged_start
                merged[at : at + len(self._chunks[i])] = self._chunks[i]
            merged[offset - merged_start : offset - merged_start + len(data)] = data
            del self._starts[lo:hi]
            del self._chunks[lo:hi]
            self._starts.insert(lo, merged_start)
            self._chunks.insert(lo, bytes(merged))
        self.buffered = sum(len(chunk) for chunk in self._chunks)

    def covers(self, start: int, end: int) -> bool:
        index = bisect.bisect_right(self._starts, start) - 1
        return index >= 0 and self._starts[index] + len(self._chunks[index]) >= end

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, buffer) -> int:
        if self._pos >= self._size:
            return 0
        want = min(len(buffer), self._size - self._pos)
        index = bisect.bisect_right(self._starts, self._pos) - 1
        if index >= 0:
            start, chunk = self._starts[index], self._chunks[index]
            inside = self._pos - start
            if inside < len(chunk):
                data = chunk[inside : inside + want]
                buffer[: len(data)] = data
                self._pos += len(data)
                return len(data)
        # Cap the miss at the next buffered range so we never re-fetch bytes
        # we already hold.
        next_index = bisect.bisect_right(self._starts, self._pos)
        if next_index < len(self._starts):
            want = min(want, self._starts[next_index] - self._pos)
        data = self._fallback.read_range(self._pos, want)
        self.fallback_bytes += len(data)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


class _VideoDecoderLRU:
    """Per-worker LRU of torchcodec decoders keyed by (video_key, chunk, file).
    eviction is bounded by ``byte_budget`` too, not just count.
    """

    def __init__(self, capacity: int, byte_budget: int | None = None):
        self.capacity = capacity
        self.byte_budget = byte_budget
        self._items: OrderedDict[tuple, tuple[object, int]] = OrderedDict()
        self._total_bytes = 0

    def __contains__(self, key: tuple) -> bool:
        return key in self._items

    def get(self, key: tuple):
        self._items.move_to_end(key)
        return self._items[key][0]

    def put(self, key: tuple, decoder, nbytes: int = 0) -> None:
        if key in self._items:
            self._total_bytes -= self._items[key][1]
        self._items[key] = (decoder, nbytes)
        self._items.move_to_end(key)
        self._total_bytes += nbytes
        while len(self._items) > 1 and (
            len(self._items) > self.capacity
            or (self.byte_budget is not None and self._total_bytes > self.byte_budget)
        ):
            _, (_, evicted_bytes) = self._items.popitem(last=False)
            self._total_bytes -= evicted_bytes


def to_lance_column(key: str) -> str:
    return key.replace(".", "_")


def _is_remote_uri(path) -> bool:
    return "://" in str(path)


def _storage_options(db_uri: str, storage_options: dict | None, revision: str | None) -> dict:
    options = dict(storage_options or {})
    if db_uri.startswith("hf://"):
        if "token" not in options:
            token = huggingface_hub.get_token()
            if token:
                options["token"] = token
        if revision and "revision" not in options:
            options["revision"] = revision
    return {key: value for key, value in options.items() if value is not None}


def _connect(db_uri: str, storage_options: dict | None, revision: str | None = None):
    require_package("lancedb", extra="lancedb")  # earliest common site: also reached via lance_metadata()
    if _is_remote_uri(db_uri):
        os.environ.setdefault("LANCE_IO_THREADS", "256")
    options = _storage_options(db_uri, storage_options, revision)
    return lancedb.connect(db_uri, **({"storage_options": options} if options else {}))


def _materialize_meta(db, local_root: Path) -> None:
    """Write ``meta/`` from the meta table to a local cache, once."""
    meta_dir = local_root / "meta"
    if meta_dir.exists():
        return
    try:
        table = db.open_table(META_TABLE)
    except Exception as error:
        raise FileNotFoundError(
            f"Dataset has no '{META_TABLE}' table. Re-convert it with a converter that "
            "ingests meta/, or create the table in place from an existing meta/ copy."
        ) from error

    tmp_dir = local_root / f"meta.tmp-{os.getpid()}"
    try:
        tmp_resolved = tmp_dir.resolve()
        for batch in table.search().select(["path", "data"]).to_batches():
            paths = batch.column("path").to_pylist()
            for i, rel_path in enumerate(paths):
                dst = (tmp_dir / rel_path).resolve()
                if not dst.is_relative_to(tmp_resolved):
                    raise ValueError(f"meta table entry escapes the cache directory: {rel_path!r}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(batch.column("data")[i].as_py())
        try:
            tmp_dir.rename(meta_dir)
        except OSError:
            if not meta_dir.exists():
                raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def lance_mp_context() -> str:
    return "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"


def is_lance_dataset(
    repo_id: str | None = None, root: str | Path | None = None, revision: str | None = None
) -> bool:
    """True if the dataset's storage format resolves to Lance."""
    return resolve_storage_format(repo_id, root, revision) == "lance"


def resolve_lance_root(
    repo_id: str | None,
    root: str | Path | None,
    storage_options: dict | None = None,
    revision: str | None = None,
    force_refresh: bool = False,
) -> tuple[str, Path]:
    """Resolve a Lance dataset to its connect URI and the local root holding ``meta/``"""
    if root is not None and _is_remote_uri(root):
        db_uri = str(root).rstrip("/")
        # Key the cache by revision too: an hf:// root at a non-default revision must not
        # reuse (or overwrite) another revision's materialized meta.
        cache_key = f"{db_uri}@{revision}" if revision else db_uri
        local_root = HF_LEROBOT_HOME / "remote" / re.sub(r"[^A-Za-z0-9._-]+", "_", cache_key)
        if force_refresh and (local_root / "meta").exists():
            shutil.rmtree(local_root / "meta")
        if not (local_root / "meta").exists():
            _materialize_meta(_connect(db_uri, storage_options, revision), local_root)
        return db_uri, local_root
    root_path = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if (root_path / f"{FRAMES_TABLE}.lance").exists():
        return str(root_path), root_path
    if repo_id is not None:
        return f"hf://datasets/{repo_id}", root_path
    raise FileNotFoundError(f"No '{FRAMES_TABLE}.lance' table under {root_path}.")


def lance_metadata(
    repo_id: str | None,
    root: str | Path | None,
    revision: str | None = None,
    storage_options: dict | None = None,
) -> LeRobotDatasetMetadata:
    _, local_root = resolve_lance_root(repo_id, root, storage_options, revision)
    return LeRobotDatasetMetadata(
        repo_id if repo_id is not None else str(local_root), root=local_root, revision=revision
    )


class LanceStorageBackend:
    """Lance retrieval backend for :class:`LeRobotDataset`.

    Holds no LeRobot semantics: it transports the standard ``meta/`` tree and
    serves batched raw reads (``get_rows``/``get_column``/``get_video_frames``)
    to :class:`~lerobot.datasets.storage_backend.BackendDatasetReader`, which
    owns episode handling, delta windows, transforms, and sample assembly.

    Args:
        repo_id: Hub dataset repo; tables stream over ``hf://``, only ``meta/`` downloads.
        root: Local dir with ``meta/`` and ``.lance`` tables, or an object-store
            URI (``s3://...``) with the same layout. Local tables win when both given.
        revision: Hub revision for the ``meta/`` download.
        storage_options: Extra options forwarded to ``lancedb.connect``.
        token: Hub authentication token, as in :class:`LeRobotDataset`. A string
            token is also forwarded to the object store for ``hf://`` roots.
        force_cache_sync: Re-download / re-materialize the cached ``meta/`` tree.
        video_decoder_cache_size: Max decoders per worker (default 16, also
            bounded by a 2 GiB per-worker byte budget).
    """

    def __init__(
        self,
        repo_id: str | None = None,
        root: str | Path | None = None,
        revision: str | None = None,
        storage_options: dict | None = None,
        token: str | bool | None = None,
        force_cache_sync: bool = False,
        video_decoder_cache_size: int | None = None,
    ):
        require_package("lancedb", extra="lancedb")
        if repo_id is None and root is None:
            raise ValueError("Provide `repo_id`, `root`, or both.")

        self.repo_id = repo_id
        if isinstance(token, str) or token is False:
            # An explicit string token reaches the object store; token=False pins the
            # entry so no ambient token is injected (None values are dropped later).
            storage_options = {**(storage_options or {})}
            storage_options.setdefault("token", token if isinstance(token, str) else None)
        self._storage_options = storage_options

        self._db_uri, self.root = resolve_lance_root(
            repo_id, root, self._storage_options, revision, force_refresh=force_cache_sync
        )

        # meta/ comes from the Hub only in the repo_id case; remote roots re-materialize
        # it above and local roots read it in place, so there is nothing to sync there.
        meta_from_hub = self._db_uri == f"hf://datasets/{repo_id}"
        self.meta = LeRobotDatasetMetadata(
            repo_id if repo_id is not None else str(self.root),
            root=self.root,
            revision=revision,
            force_cache_sync=force_cache_sync and meta_from_hub,
            token=token,
        )
        self._hub_revision = self.meta.revision if meta_from_hub else None

        if self.meta.image_keys:
            raise NotImplementedError(
                f"Image-backed features are not supported by the Lance backend: {self.meta.image_keys}. "
                "Re-encode them as video."
            )

        # Storage-side column mapping (dots -> underscores) and value decoding.
        self._tabular_keys = [
            key
            for key in self.meta.features
            if key not in self.meta.video_keys and key not in self.meta.image_keys
        ]
        self._fetch_columns = [to_lance_column(key) for key in self._tabular_keys]
        self._string_keys = {
            key for key in self._tabular_keys if self.meta.features[key].get("dtype") == "string"
        }
        self._language_keys = {
            key for key in self._tabular_keys if self.meta.features[key].get("dtype") == "language"
        }

        self._frames_table = None
        self._frames_perm = None
        self._videos_table = None
        self._video_row_ids: dict[tuple, int] | None = None
        self._file_meta: OrderedDict[tuple, dict] = OrderedDict()
        self._prefetch_pool: ThreadPoolExecutor | None = None
        self._decode_pool: ThreadPoolExecutor | None = None
        self._pending_prepare = None
        if video_decoder_cache_size is None:
            video_decoder_cache_size = 16
        self._decoder_cache = _VideoDecoderLRU(video_decoder_cache_size, byte_budget=2 << 30)  # 2GB cap

    def _ensure_open(self) -> None:
        if self._frames_perm is not None:
            return
        if self.meta.video_keys:
            self._prefetch_pool = ThreadPoolExecutor(max_workers=1)
            self._decode_pool = ThreadPoolExecutor(max_workers=16)
        db = _connect(self._db_uri, self._storage_options, revision=self._hub_revision)
        table = db.open_table(FRAMES_TABLE)
        n_rows = table.count_rows()
        if n_rows != self.meta.total_frames:
            raise ValueError(
                f"frames table has {n_rows} rows but meta declares "
                f"{self.meta.total_frames} frames; the dataset is truncated or corrupt."
            )
        self._frames_table = table
        self._frames_perm = (
            Permutation.identity(table).select_columns(self._fetch_columns).with_format("arrow")
        )
        if self.meta.video_keys:
            self._videos_table = db.open_table(VIDEOS_TABLE)
            # future TODO: resolve row ids lazily per batch.
            index = (
                self._videos_table.search()
                .select(["video_key", "chunk_index", "file_index"])
                .with_row_id(True)
                .to_arrow()
            )
            self._video_row_ids = {
                (row["video_key"], row["chunk_index"], row["file_index"]): row["_rowid"]
                for row in index.to_pylist()
            }
            episodes_data = self.meta.episodes.data
            referenced = {
                (key, int(chunk), int(file))
                for key in self.meta.video_keys
                for chunk, file in zip(
                    episodes_data.column(f"videos/{key}/chunk_index").to_pylist(),
                    episodes_data.column(f"videos/{key}/file_index").to_pylist(),
                    strict=True,
                )
            }
            missing = referenced - self._video_row_ids.keys()
            if missing:
                raise ValueError(
                    f"videos table is missing {len(missing)} file(s) referenced by episode "
                    f"metadata, e.g. {sorted(missing)[:3]}. The dataset is incomplete or "
                    "was converted against different metadata."
                )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_frames_table"] = None
        state["_frames_perm"] = None
        state["_videos_table"] = None
        state["_video_row_ids"] = None
        state["_file_meta"] = OrderedDict()
        state["_prefetch_pool"] = None
        state["_decode_pool"] = None
        state["_pending_prepare"] = None
        state["_decoder_cache"] = _VideoDecoderLRU(
            self._decoder_cache.capacity, byte_budget=self._decoder_cache.byte_budget
        )
        return state

    def get_rows(self, rows: list[int]) -> dict[str, np.ndarray | list]:
        """One batched columnar read of the given absolute frame rows."""
        self._ensure_open()
        batch = self._frames_perm.__getitems__(rows)
        columns = {}
        for key, lance_name in zip(self._tabular_keys, self._fetch_columns, strict=True):
            array = batch.column(lance_name)
            if hasattr(array, "combine_chunks"):
                array = array.combine_chunks()
            if key in self._language_keys:
                rows = array.to_pylist()
                for row in rows:
                    for msg in row or ():
                        if msg.get("tool_calls"):
                            msg["tool_calls"] = [
                                json.loads(call) if isinstance(call, str) else call
                                for call in msg["tool_calls"]
                            ]
                columns[key] = rows
            elif hasattr(array, "flatten") and hasattr(array.type, "value_type"):
                values = array.flatten().to_numpy(zero_copy_only=False)
                columns[key] = values.reshape(len(array), -1)
            else:
                columns[key] = array.to_numpy(zero_copy_only=False)
        return columns

    def prefetch_videos(self, windows: dict[tuple, list[tuple[int, int]]]) -> None:
        """Overlap hint: start fetching video bytes for the given (file -> frame spans)."""
        self._ensure_open()
        self._pending_prepare = self._prefetch_pool.submit(self._prepare_files, sorted(windows), windows)

    def get_video_frames(
        self,
        requests: dict[tuple, list[tuple, list[float]]],
        *,
        tolerance_s: float,
        return_uint8: bool,
    ) -> dict:
        """Batched decode: one blob fetch + one decode pass per video file.

        Depth streams decode through pyav and are returned raw (undequantized),
        mirroring upstream's ``decode_video_frames`` contract.
        """
        self._ensure_open()
        pending, self._pending_prepare = self._pending_prepare, None
        entries = pending.result() if pending is not None else self._prepare_files(sorted(requests))

        results: dict = {}

        def _decode_file(file_key: tuple, file_requests: list[tuple[tuple, list[float]]]) -> None:
            key, chunk_idx, file_idx = file_key
            decoder, source = entries[file_key]
            if key in self.meta.depth_keys:
                for request_id, shifted_ts in file_requests:
                    results[request_id] = self._decode_depth_window(source, shifted_ts, tolerance_s)
                return
            fps = decoder.metadata.average_fps
            for request_id, shifted_ts in file_requests:
                indices = [round(ts * fps) for ts in shifted_ts]
                batch = decoder.get_frames_at(indices=indices)
                distance = (torch.tensor(shifted_ts, dtype=torch.float64) - batch.pts_seconds).abs()
                if (distance >= tolerance_s).any():
                    raise FrameTimestampError(
                        f"Query timestamps violate tolerance_s={tolerance_s} for video "
                        f"'{key}' (chunk {chunk_idx}, file {file_idx}): queried {shifted_ts}, "
                        f"loaded {batch.pts_seconds.tolist()}."
                    )
                frames = batch.data
                if not return_uint8:
                    frames = (frames / 255.0).type(torch.float32)
                results[request_id] = frames

        futures = [self._decode_pool.submit(_decode_file, k, r) for k, r in requests.items()]
        for future in futures:
            future.result()
        return results

    def _prepare_files(
        self, file_keys: list[tuple], windows: dict[tuple, list[tuple[int, int]]] | None = None
    ) -> dict[tuple, tuple]:
        """Stage 1 of video decoding: everything that doesn't need timestamps"""
        # Lazy load torchcodec
        from torchcodec.decoders import VideoDecoder

        self._load_file_meta([key for key in file_keys if key not in self._file_meta])

        prepared: dict[tuple, tuple] = {}
        new_files = []
        for key in file_keys:
            if key in self._decoder_cache:
                prepared[key] = self._decoder_cache.get(key)
            else:
                new_files.append(key)

        if new_files:
            handles = self._videos_table.fetch_blob_files(
                VIDEO_BLOB_COLUMN, [self._video_row_ids[key] for key in new_files]
            )
            sources = {
                key: _SparseBlobSource(self._file_meta[key]["file_size"], handle)
                for key, handle in zip(new_files, handles, strict=True)
            }
            spans_by_key: dict[tuple, list[tuple[int, int]]] = {}
            for key in new_files:
                meta = self._file_meta[key]
                spans = [
                    (0, min(_OPEN_PROBE_BYTES, meta["file_size"])),
                    # Slack past the moov covers the next box header ffmpeg reads.
                    (
                        meta["moov_offset"],
                        min(meta["moov_offset"] + meta["moov_size"] + _RANGE_SLACK, meta["file_size"]),
                    ),
                ]
                if len(meta["kf_positions"]):
                    first_packet = int(meta["kf_positions"][0])
                    spans.append((first_packet, min(first_packet + _OPEN_PROBE_BYTES, meta["file_size"])))
                spans_by_key[key] = spans
            self._fetch_spans(spans_by_key, sources)

            rgb_files = [key for key in new_files if key[0] not in self.meta.depth_keys]
            created = self._decode_pool.map(
                lambda key: VideoDecoder(sources[key], seek_mode="approximate"), rgb_files
            )
            for key, decoder in zip(rgb_files, created, strict=True):
                prepared[key] = (decoder, sources[key])
            for key in new_files:
                if key[0] in self.meta.depth_keys:
                    prepared[key] = (None, sources[key])

        if windows:
            window_spans: dict[tuple, list[tuple[int, int]]] = {}
            for key, frame_windows in windows.items():
                meta = self._file_meta[key]
                source = prepared[key][1]
                spans = [
                    span
                    for first, last in frame_windows
                    for span in [self._window_byte_range(key[0], meta, first, last)]
                    if not source.covers(*span)
                ]
                if spans:
                    window_spans[key] = spans
            self._fetch_spans(window_spans, {key: prepared[key][1] for key in window_spans})

        # Insert/refresh each file in the decoder cache.
        for key, (decoder, source) in prepared.items():
            self._decoder_cache.put(key, (decoder, source), nbytes=source.buffered)
        return prepared

    def _decode_depth_window(self, source, shifted_ts: list[float], tolerance_s: float) -> torch.Tensor:
        """Decode one depth window with upstream's pyav decoder over our sparse source."""
        source.seek(0)
        return decode_video_frames_pyav(source, shifted_ts, tolerance_s, return_uint8=False, is_depth=True)

    def _fetch_spans(
        self, spans_by_key: dict[tuple, list[tuple[int, int]]], sources: dict[tuple, _SparseBlobSource]
    ) -> None:
        range_requests: list[tuple[int, int, int]] = []
        range_targets: list[tuple[tuple, int]] = []
        for key, spans in spans_by_key.items():
            for start, end in _merge_spans(spans):
                range_requests.append((self._video_row_ids[key], start, end - start))
                range_targets.append((key, start))
        if not range_requests:
            return
        payloads = self._videos_table.fetch_blob_ranges(VIDEO_BLOB_COLUMN, range_requests)
        for (key, offset), payload in zip(range_targets, payloads, strict=True):
            sources[key].add(offset, payload.as_py())

    def _load_file_meta(self, missing: list[tuple]) -> None:
        """Fetch byte-index columns for files not yet in the per-worker cache."""
        if not missing:
            return
        row_ids = [self._video_row_ids[file_key] for file_key in missing]
        batch = (
            self._videos_table.take_row_ids(row_ids)
            .select(["video_key", "chunk_index", "file_index", *VIDEO_INDEX_COLUMNS])
            .to_arrow()
        )
        if batch.num_rows != len(missing):
            raise ValueError(
                "videos table is missing byte-index columns or rows "
                f"({batch.num_rows} matches for {len(missing)} files). Re-convert the dataset "
                f"with a converter that writes {VIDEO_INDEX_COLUMNS}."
            )
        scalars = {
            name: batch.column(name).to_pylist()
            for name in ("video_key", "chunk_index", "file_index", "file_size", "moov_offset", "moov_size")
        }
        kf_index_column = batch.column("kf_indices").combine_chunks()
        kf_position_column = batch.column("kf_positions").combine_chunks()
        index_offsets = kf_index_column.offsets.to_numpy(zero_copy_only=False)
        index_values = kf_index_column.values.to_numpy(zero_copy_only=False)
        position_offsets = kf_position_column.offsets.to_numpy(zero_copy_only=False)
        position_values = kf_position_column.values.to_numpy(zero_copy_only=False)
        for i in range(batch.num_rows):
            file_key = (scalars["video_key"][i], scalars["chunk_index"][i], scalars["file_index"][i])
            self._file_meta[file_key] = {
                "file_size": scalars["file_size"][i],
                "moov_offset": scalars["moov_offset"][i],
                "moov_size": scalars["moov_size"][i],
                "kf_indices": index_values[index_offsets[i] : index_offsets[i + 1]],
                "kf_positions": position_values[position_offsets[i] : position_offsets[i + 1]],
            }
        while len(self._file_meta) > 2048:
            self._file_meta.popitem(last=False)

    def _window_byte_range(self, key: str, meta: dict, first_frame: int, last_frame: int) -> tuple[int, int]:
        """Byte range covering frames [first, last]: preceding keyframe to next keyframe."""
        kf_indices, kf_positions = meta["kf_indices"], meta["kf_positions"]
        start_idx = max(int(np.searchsorted(kf_indices, first_frame, side="right")) - 1, 0)
        end_idx = int(np.searchsorted(kf_indices, last_frame, side="right"))
        end = int(kf_positions[end_idx]) if end_idx < len(kf_positions) else meta["file_size"]
        slack = _RANGE_SLACK * 4 if key in self.meta.depth_keys else _RANGE_SLACK
        return int(kf_positions[start_idx]), min(end + slack, meta["file_size"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  repo_id={self.repo_id},\n"
            f"  uri={self._db_uri},\n"
            f"  frames={self.meta.total_frames} / episodes={self.meta.total_episodes},\n"
        )
