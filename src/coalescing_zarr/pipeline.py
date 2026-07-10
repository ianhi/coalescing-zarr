"""``CoalescingCodecPipeline`` — the mandatory glue that calls ``get_many_chunks``.

zarr's read path never calls a bulk-get store method on its own (see
``design.md`` §Framing): the built-in pipeline fetches one chunk per
``getter.get()``. So a store method alone changes nothing. This pipeline is what
teaches zarr to use it.

It overrides ``read`` — the hook that receives the *entire* ``batch_info`` before
zarr splits it into size-1 batches — and:

1. if the store does not expose ``get_many_chunks``, delegates to the stock
   pipeline unchanged;
2. otherwise streams chunks from ``get_many_chunks`` and **kicks off the decode
   of each chunk the moment its bytes arrive**, so decoding overlaps the fetches
   still in flight (the fetch<->decode overlap the design treats as first-order).

Decode itself is *not* reimplemented: each arrived chunk is replayed through the
stock ``read_batch`` via a :class:`CachedGetter`, so the decode/assembly path is
byte-for-byte identical to a normal read. The only thing we change is *when*
each decode starts and how the bytes were fetched.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.config import config as zarr_config

from coalescing_zarr.icechunk_native import (
    is_native_icechunk_store,
    stream_icechunk_chunks,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.abc.codec import GetResult
    from zarr.abc.store import ByteGetter, ByteRequest
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.indexing import SelectorTuple
    from zarr.storage import StorePath

    BatchEntry = tuple[
        "ByteGetter", "ArraySpec", "SelectorTuple", "SelectorTuple", bool
    ]
    BatchInfo = Iterable[BatchEntry]

# Coalescing knobs: defaults + the zarr-config keys they are read from. They live
# in this leaf module so both `config.py` and this pipeline can import them
# without a cycle (config.py registers this pipeline class, so it can't be
# imported here). ``max_gap`` is unwanted bytes tolerated between two chunks
# before their range GETs merge; ``max_coalesced_bytes`` caps one merged request.
# 256 KiB matches the gap used in the prior NDPI measurements.
DEFAULT_MAX_GAP = 256 * 1024
DEFAULT_MAX_COALESCED_BYTES: int | None = None
MAX_GAP_KEY = "coalescing.max_gap"
MAX_COALESCED_BYTES_KEY = "coalescing.max_coalesced_bytes"


@dataclass
class CachedGetter:
    """A ByteGetter that serves bytes already fetched by ``get_many_chunks``.

    Lets us replay prefetched bytes through the stock ``read_batch`` so decode is
    unchanged. ``buffer`` is ``None`` for a missing chunk, which the stock path
    turns into the fill value.
    """

    buffer: Buffer | None

    async def get(
        self,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # The non-sharded virtual read path requests whole chunks
        # (byte_range is None). Partial/sharded decode is out of MVP scope, so
        # fail loudly rather than silently return the wrong bytes if it appears.
        if byte_range is not None:
            raise NotImplementedError("CachedGetter serves whole chunks only")
        return self.buffer


class CoalescingCodecPipeline(BatchedCodecPipeline):
    """A codec pipeline that fetches via ``get_many_chunks`` and decodes on arrival."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Snapshot the coalescing knobs from config at construction. zarr builds
        # the pipeline instance once, when the array is opened (verified) — so
        # an opener that sets these in config only for the open window (see
        # ``open_zarr_coalesced``) still binds them here, and a later config
        # reset does not lose them. Only the native-Icechunk path uses these;
        # the pure-Python store carries its own per-instance knobs.
        max_gap = zarr_config.get(MAX_GAP_KEY, DEFAULT_MAX_GAP)
        mcb = zarr_config.get(MAX_COALESCED_BYTES_KEY, DEFAULT_MAX_COALESCED_BYTES)
        self.max_gap = int(max_gap)
        self.max_coalesced_bytes = None if mcb is None else int(mcb)

    async def read(
        self,
        batch_info: BatchInfo,
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        entries = list(batch_info)
        if not entries:
            return ()

        # The read path always passes StorePath byte-getters (store / key).
        first_getter = cast("StorePath", entries[0][0])
        store = getattr(first_getter, "store", None)
        # Two bulk paths: a native Icechunk store (we adapt its native
        # get_many_chunks here) and the pure-Python CoalescingManifestStore
        # (which exposes a (keys, prototype) -> (key, buffer) get_many_chunks).
        native = store is not None and is_native_icechunk_store(store)
        generic = store is not None and not native and hasattr(store, "get_many_chunks")
        if (
            (not native and not generic)
            # Sharded / partial-decode arrays drive byte_range reads that a
            # whole-chunk CachedGetter cannot serve; let the stock pipeline
            # handle them (coalescing targets non-sharded virtual arrays).
            or self.supports_partial_decode
        ):
            return await super().read(entries, out, drop_axes)

        # Map each chunk key to the batch entries that want it. A key normally
        # appears once, but mapping to a list keeps us correct (and avoids a
        # dropped result) if an indexer ever emits the same chunk twice.
        by_key: dict[str, list[tuple[int, BatchEntry]]] = {}
        for i, entry in enumerate(entries):
            key = cast("StorePath", entry[0]).path
            by_key.setdefault(key, []).append((i, entry))

        assert store is not None  # guaranteed by the native/generic guard above
        prototype: BufferPrototype = entries[0][1].prototype
        # Indices preserve the input order the caller maps results back by.
        results: list[GetResult | None] = [None] * len(entries)

        # Bind the parent method now; ``super()`` won't resolve inside the
        # nested coroutine below. Bound the number of *concurrent* decodes the
        # same way the stock pipeline does, so a large selection can't spawn an
        # unbounded decode fan-out.
        parent_read_batch = super().read_batch
        decode_sem = asyncio.Semaphore(int(zarr_config.get("async.concurrency")))

        async def decode_one(index: int, entry: BatchEntry, buf: Buffer | None) -> None:
            async with decode_sem:
                # Replace only the byte-getter; reuse the rest of the batch entry.
                single: BatchEntry = (CachedGetter(buf), *entry[1:])
                res = await parent_read_batch([single], out, drop_axes)
            results[index] = res[0]

        decode_tasks: list[asyncio.Task[None]] = []
        if native:
            # Adapt Icechunk's native (array_path, coords) -> (index, bytes)
            # getter to the (key, buffer) stream this pipeline decodes from.
            chunks = stream_icechunk_chunks(
                store,
                list(by_key),
                prototype=prototype,
                max_gap=self.max_gap,
                max_coalesced_bytes=self.max_coalesced_bytes,
            )
        else:
            # Pure-Python store: it carries its own per-instance knobs.
            chunks = store.get_many_chunks(list(by_key), prototype=prototype)
        try:
            async for key, buf in chunks:
                for index, entry in by_key[key]:
                    decode_tasks.append(
                        asyncio.create_task(decode_one(index, entry, buf))
                    )
            if decode_tasks:
                await asyncio.gather(*decode_tasks)
        finally:
            # On any failure (a span fetch or a sibling decode raising), stop the
            # remaining decodes so they don't keep writing into `out` after we
            # return/raise, and close the generator so it cancels its fetches.
            for task in decode_tasks:
                if not task.done():
                    task.cancel()
            if decode_tasks:
                await asyncio.gather(*decode_tasks, return_exceptions=True)
            aclose = getattr(chunks, "aclose", None)
            if aclose is not None:
                await aclose()
        return cast("tuple[GetResult, ...]", tuple(results))
