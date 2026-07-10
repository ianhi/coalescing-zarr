"""``read_region`` — a cross-array coalesced reader over an Icechunk repo.

This is the non-zarr sibling of :class:`~coalescing_zarr.pipeline.
CoalescingCodecPipeline`. That pipeline can only coalesce *within* a single
array read, because zarr drives one array at a time. ``read_region`` instead
builds **one** flat request list spanning *several* arrays and hands it to
Icechunk's native ``get_many_chunks`` in a single call. Chunks from different
arrays that live in the same backing object then coalesce together — the win
zarr's per-array read path structurally can't get (think GOES bands
block-interleaved in one NetCDF).

Decode is not reimplemented: each chunk's bytes are replayed through the stock
``BatchedCodecPipeline.read_batch`` via a :class:`~coalescing_zarr.pipeline.
CachedGetter`, so a chunk decodes byte-for-byte identically to a normal zarr
read. We only own the fan-out (one native call across arrays) and the assembly
(placing each decoded chunk into its array's window output, clipped to the
window bounds; missing/``None`` chunks stay at the fill value).
"""

from __future__ import annotations

import asyncio
import itertools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from zarr.core.array import _get_chunk_spec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from coalescing_zarr.config import DEFAULT_MAX_COALESCED_BYTES, DEFAULT_MAX_GAP
from coalescing_zarr.pipeline import CachedGetter

if TYPE_CHECKING:
    import icechunk

__all__ = ["read_region"]


def _normalize_window(
    window: tuple[Any, ...] | Any, shape: tuple[int, ...]
) -> tuple[tuple[int, int, bool], ...]:
    """Resolve ``window`` against ``shape`` to per-dim ``(start, stop, is_int)``.

    An int selector keeps ``start == coord`` and ``is_int=True`` so we can drop
    that axis from the output (matching numpy fancy-free basic indexing). A slice
    is clamped to ``[0, size]``; ``step`` is out of scope (chunk geometry assumes
    step 1, as the perf harness patterns do).
    """
    if not isinstance(window, tuple):
        window = (window,)
    if len(window) > len(shape):
        raise IndexError("window has more dimensions than the array")
    resolved: list[tuple[int, int, bool]] = []
    for dim, size in enumerate(shape):
        sel = window[dim] if dim < len(window) else slice(None)
        if isinstance(sel, slice):
            if sel.step not in (None, 1):
                raise NotImplementedError("read_region supports step-1 slices only")
            start = 0 if sel.start is None else max(0, sel.start)
            stop = size if sel.stop is None else min(size, sel.stop)
            resolved.append((start, max(start, stop), False))
        else:
            idx = int(sel)
            if idx < 0:
                idx += size
            resolved.append((idx, idx + 1, True))
    return tuple(resolved)


def _window_coords(
    resolved: tuple[tuple[int, int, bool], ...], chunks: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """Chunk-grid coords the resolved window touches (per-dim product).

    Same geometry as ``benchmarks/read_goes_coalesced.py``: a slice spans
    ``range(start // cs, (stop - 1) // cs + 1)``, an int a single chunk.
    """
    ranges = [
        range(start // cs, (stop - 1) // cs + 1)
        for (start, stop, _), cs in zip(resolved, chunks, strict=True)
    ]
    return [tuple(c) for c in itertools.product(*ranges)]


def read_region(
    session: icechunk.Session,
    arrays: list[str],
    window: tuple[Any, ...] | Any,
    *,
    max_gap: int = DEFAULT_MAX_GAP,
    max_coalesced_bytes: int | None = DEFAULT_MAX_COALESCED_BYTES,
    decode_workers: int | None = None,
) -> dict[str, np.ndarray]:
    """Read ``window`` from several arrays in ONE bulk call, decode + assemble.

    Builds one flat request list spanning all arrays and hands it to the native
    ``get_many_chunks`` in a single call (which coalesces *within* each manifest
    and pipelines across manifests), then decodes + assembles per array. Note
    Icechunk coalesces per-manifest, so chunks from different arrays (different
    manifests) are fetched correctly but not merged into shared range GETs — the
    win here is the single bulk call + per-manifest coalescing, not cross-array
    merging.

    Parameters
    ----------
    session
        An open ``icechunk.Session`` whose store exposes the native
        ``get_many_chunks`` (see :class:`~coalescing_zarr.icechunk_store.
        CoalescingIcechunkStore`).
    arrays
        Array paths within the session's root group, e.g.
        ``["grp/CMI_C01", "grp/CMI_C02"]``. All are indexed with the same
        ``window`` (they must share a grid for that to be meaningful, but each
        array's own shape/chunks are used, so mismatched grids still read
        correctly per array).
    window
        Ints/slices applied to every array. Ints drop their axis in the output.
    decode_workers
        Threads used to decode fetched chunks. Decode (numcodecs decompression)
        is CPU-bound and releases the GIL, so running it on a thread pool spreads
        it across cores — the fetch happens on the shared IO loop, but decode does
        not (on the event loop it serializes to ~1 core, which dominates the wall
        time for many-small-chunk arrays). ``None`` → ``min(32, os.cpu_count())``;
        ``1`` restores the serial, on-loop behaviour.

    Returns
    -------
    dict
        ``{array_path: ndarray}`` — each ndarray is the ``window`` slice of that
        array, identical to ``zarr.open_group(session.store)[path][window]``.
    """
    from coalescing_zarr.icechunk_store import _MISSING_NATIVE_MSG

    store = session.store
    if not hasattr(store, "get_many_chunks"):
        raise NotImplementedError(_MISSING_NATIVE_MSG)

    proto = default_buffer_prototype()
    group = zarr.open_group(store, mode="r")

    # Per array: resolve the window, pre-allocate its output, and record every
    # chunk coord it needs. `plans[path]` carries what assembly later needs.
    plans: dict[str, dict[str, Any]] = {}
    requests: list[tuple[str, tuple[int, ...]]] = []
    # request index -> (array_path, coords), so completion-order results route back.
    index_map: list[tuple[str, tuple[int, ...]]] = []

    for path in arrays:
        arr = group[path]
        if not isinstance(arr, zarr.Array):
            raise TypeError(f"{path!r} is a group, not an array")
        async_arr = arr._async_array
        meta = async_arr.metadata
        chunks = arr.chunks
        resolved = _normalize_window(window, arr.shape)

        # Output shape drops int-indexed axes (basic-indexing semantics).
        out_shape = tuple(
            stop - start for (start, stop, is_int) in resolved if not is_int
        )
        native_dtype = arr.dtype
        out = np.full(out_shape, arr.fill_value, dtype=native_dtype)

        plans[path] = {
            "async_arr": async_arr,
            "meta": meta,
            "chunk_grid": async_arr._chunk_grid,
            "config": async_arr.config,
            "codec_pipeline": async_arr.codec_pipeline,
            "chunks": chunks,
            "resolved": resolved,
            "out": out,
        }
        for coords in _window_coords(resolved, chunks):
            index_map.append((path, coords))
            requests.append((path, coords))

    if not requests:
        return {path: plans[path]["out"] for path in arrays}

    # ONE native call across ALL arrays; Icechunk coalesces within each manifest
    # and pipelines manifests (cross-array chunks are separate manifests -> not
    # merged, but fetched correctly). Collect the bytes on the IO loop first, then
    # decode off it: decode is the wall-time cost for many-small-chunk arrays and
    # only parallelizes across cores when run on threads, not the event loop.
    fetched: list[bytes | None] = [None] * len(requests)

    async def _fetch() -> None:
        chunks_iter = store.get_many_chunks(
            requests, max_gap=max_gap, max_coalesced_bytes=max_coalesced_bytes
        )
        try:
            async for index, data in chunks_iter:
                fetched[index] = None if data is None else bytes(data)
        finally:
            await chunks_iter.aclose()

    sync(_fetch())

    workers = (
        decode_workers if decode_workers is not None else min(32, os.cpu_count() or 1)
    )
    workers = max(1, min(workers, len(requests)))

    # Each chunk writes a disjoint window of its array's output, so decoding them
    # concurrently is race-free. Each group runs on its OWN worker thread with its
    # own event loop: a fresh thread has no running loop (so asyncio.run is legal
    # even when the caller is inside one, e.g. Jupyter), and separate loops avoid
    # re-serializing the GIL-releasing decode back onto the shared IO loop.
    def _decode_group(idxs: list[int]) -> None:
        async def _run() -> None:
            for index in idxs:
                path, coords = index_map[index]
                await _decode_into(plans[path], coords, fetched[index], proto)

        asyncio.run(_run())

    groups = [list(range(k, len(requests), workers)) for k in range(workers)]
    groups = [g for g in groups if g]
    with ThreadPoolExecutor(len(groups)) as pool:
        list(pool.map(_decode_group, groups))
    return {path: plans[path]["out"] for path in arrays}


async def _decode_into(
    plan: dict[str, Any],
    coords: tuple[int, ...],
    data: bytes | None,
    proto: Any,
) -> None:
    """Decode one chunk's bytes and place it into the array's window output.

    Decodes a *whole* chunk through the stock ``read_batch`` (byte-identical to
    a normal read), then copies the overlap of that chunk with the window into
    the output — clipping partial edge chunks. ``data is None`` means the chunk
    is uninitialized, so we leave the output at its fill value.
    """
    if data is None:
        return  # output was pre-filled with fill_value

    meta = plan["meta"]
    config = plan["config"]
    chunks = plan["chunks"]
    resolved = plan["resolved"]

    spec = _get_chunk_spec(meta, plan["chunk_grid"], coords, config, proto)
    native_dtype = spec.dtype.to_native_dtype()
    full_sel = tuple(slice(None) for _ in chunks)
    chunk_out = proto.nd_buffer.create(
        shape=spec.shape,
        dtype=native_dtype,
        fill_value=meta.fill_value,
        order=config.order,
    )
    buf = proto.buffer.from_bytes(data)
    await plan["codec_pipeline"].read_batch(
        [(CachedGetter(buf), spec, full_sel, full_sel, True)], chunk_out
    )
    decoded = np.asarray(chunk_out.as_numpy_array())

    # For each dim work out the slice of this chunk that lands in the window and
    # where in the (int-axes-dropped) output it goes.
    chunk_src: list[slice] = []
    out_dst: list[slice] = []
    for (start, stop, is_int), cs, coord in zip(resolved, chunks, coords, strict=True):
        chunk_lo = coord * cs
        lo = max(start, chunk_lo)
        hi = min(stop, chunk_lo + cs)
        chunk_src.append(slice(lo - chunk_lo, hi - chunk_lo))
        if not is_int:  # int axes are squeezed out of the output
            out_dst.append(slice(lo - start, hi - start))

    plan["out"][tuple(out_dst)] = decoded[tuple(chunk_src)]
