"""Shared fixtures: synthetic virtual stores with controllable chunk layout.

We build a ManifestArray whose chunks are uncompressed (raw ``bytes`` codec)
little-endian int32, so a chunk's on-disk bytes are exactly ``chunk.tobytes()``.
That lets us lay the chunks out in backing files at arbitrary offsets — with
gaps, or split across "files" — and know precisely what bytes each chunk should
decode to. This is how we exercise the coalescing planner's merge/split and
over-read behaviour against a byte-exact ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr.manifests import ChunkManifest, ManifestArray, ManifestGroup
from virtualizarr.manifests.utils import create_v3_array_metadata

from coalescing_zarr.manifest_store import CoalescingManifestStore

# 4 rows x 4 cols int32 == 64 bytes per chunk.
CHUNK_ROWS = 4
CHUNK_COLS = 4
BYTES_PER_CHUNK = CHUNK_ROWS * CHUNK_COLS * 4


@dataclass
class SyntheticStore:
    store: CoalescingManifestStore
    expected: np.ndarray
    array_name: str = "data"


def _raw_bytes_codec() -> list[dict[str, object]]:
    return [{"name": "bytes", "configuration": {"endian": "little"}}]


def build_synthetic_store(
    tmp_path: Path,
    *,
    gaps: list[int],
    files: list[int] | None = None,
    inline: set[int] | None = None,
) -> SyntheticStore:
    """Build a 1-row-of-chunks virtual array with a chosen on-disk layout.

    Parameters
    ----------
    gaps
        ``gaps[i]`` is the number of filler bytes written *before* chunk ``i``
        within its backing file. ``len(gaps)`` is the number of chunks.
    files
        ``files[i]`` is the index of the backing file chunk ``i`` lives in.
        Defaults to all chunks in file 0. Use distinct file indices to model
        the time-series "one chunk per file" case.
    inline
        Indices of chunks to store *inlined* in the manifest (real bytes held
        in memory, no backing-file offset) instead of as a file reference.
    """
    n_chunks = len(gaps)
    if files is None:
        files = [0] * n_chunks
    inline = inline or set()

    expected = np.arange(CHUNK_ROWS * CHUNK_COLS * n_chunks, dtype="<i4").reshape(
        CHUNK_ROWS, CHUNK_COLS * n_chunks
    )

    # Encoded bytes for chunk j: columns [4j, 4j+4).
    chunk_bytes = [
        np.ascontiguousarray(
            expected[:, j * CHUNK_COLS : (j + 1) * CHUNK_COLS]
        ).tobytes()
        for j in range(n_chunks)
    ]

    # Lay chunks into their backing files, honouring the requested gaps.
    file_buffers: dict[int, bytearray] = {}
    paths = np.empty((1, n_chunks), dtype=np.dtypes.StringDType())
    offsets = np.zeros((1, n_chunks), dtype=np.uint64)
    lengths = np.zeros((1, n_chunks), dtype=np.uint64)
    inlined: dict[tuple[int, ...], bytes] = {}

    for j in range(n_chunks):
        if j in inline:
            inlined[(0, j)] = chunk_bytes[j]
            continue
        f = files[j]
        buf = file_buffers.setdefault(f, bytearray())
        buf.extend(b"\x00" * gaps[j])
        offset = len(buf)
        buf.extend(chunk_bytes[j])
        file_path = tmp_path / f"backing_{f}.bin"
        paths[0, j] = file_path.as_uri()
        offsets[0, j] = offset
        lengths[0, j] = len(chunk_bytes[j])

    for f, buf in file_buffers.items():
        (tmp_path / f"backing_{f}.bin").write_bytes(bytes(buf))

    manifest = ChunkManifest.from_arrays(
        paths=paths,
        offsets=offsets,
        lengths=lengths,
        validate_paths=False,
        inlined=inlined or None,
    )
    metadata = create_v3_array_metadata(
        shape=(CHUNK_ROWS, CHUNK_COLS * n_chunks),
        data_type=np.dtype("<i4"),
        chunk_shape=(CHUNK_ROWS, CHUNK_COLS),
        codecs=_raw_bytes_codec(),
    )
    marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
    group = ManifestGroup(arrays={"data": marr})

    registry = ObjectStoreRegistry()
    registry.register(tmp_path.as_uri() + "/", LocalStore(prefix=tmp_path))
    store = CoalescingManifestStore(group, registry=registry)
    return SyntheticStore(store=store, expected=expected)


@pytest.fixture
def adjacent_store(tmp_path: Path) -> SyntheticStore:
    """4 chunks, zero gaps — one contiguous run (collapses even at max_gap=0)."""
    return build_synthetic_store(tmp_path, gaps=[0, 0, 0, 0])


@pytest.fixture
def gapped_store(tmp_path: Path) -> SyntheticStore:
    """4 chunks with 1 KiB gaps between them in one file."""
    return build_synthetic_store(tmp_path, gaps=[0, 1024, 1024, 1024])


@pytest.fixture
def time_series_store(tmp_path: Path) -> SyntheticStore:
    """4 chunks, each in its own file — nothing to coalesce (worst case)."""
    return build_synthetic_store(tmp_path, gaps=[0, 0, 0, 0], files=[0, 1, 2, 3])


@pytest.fixture
def inlined_store(tmp_path: Path) -> SyntheticStore:
    """4 chunks, one of which is inlined in the manifest (real data, no offset).

    Guards against treating an uncoalescable inlined chunk as missing (which
    would decode to the fill value — silent corruption).
    """
    return build_synthetic_store(tmp_path, gaps=[0, 1024, 1024, 1024], inline={2})
