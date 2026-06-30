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
from typing import TYPE_CHECKING, cast

from zarr.core.codec_pipeline import BatchedCodecPipeline

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
        if store is None or not hasattr(store, "get_many_chunks"):
            # Not a coalescing store — behave exactly like the stock pipeline.
            return await super().read(entries, out, drop_axes)

        # Map each chunk key back to its (index, batch entry). Index preserves
        # the input order that the caller maps results back by position.
        by_key: dict[str, tuple[int, BatchEntry]] = {}
        for i, entry in enumerate(entries):
            key = cast("StorePath", entry[0]).path
            by_key[key] = (i, entry)

        prototype: BufferPrototype = entries[0][1].prototype
        results: list[GetResult | None] = [None] * len(entries)

        # Bind the parent method now; ``super()`` won't resolve inside the
        # nested coroutine below.
        parent_read_batch = super().read_batch

        async def decode_one(index: int, entry: BatchEntry, buf: Buffer | None) -> None:
            # Replace only the byte-getter; reuse the rest of the batch entry.
            single: BatchEntry = (CachedGetter(buf), *entry[1:])
            res = await parent_read_batch([single], out, drop_axes)
            results[index] = res[0]

        decode_tasks: list[asyncio.Task[None]] = []
        async for key, buf in store.get_many_chunks(list(by_key), prototype=prototype):
            index, entry = by_key[key]
            decode_tasks.append(asyncio.create_task(decode_one(index, entry, buf)))

        if decode_tasks:
            await asyncio.gather(*decode_tasks)
        return cast("tuple[GetResult, ...]", tuple(results))
