"""Concurrent span fetch + completion-order streaming.

Shared by the pure-Python coalescing store: once
:func:`~coalescing_zarr.manifest_store.planning.plan_spans`
has grouped chunks into byte-range spans, this fetches the spans concurrently and
yields each member chunk's bytes as its span's range GET completes — so the caller
can decode a chunk the instant it arrives, overlapping decode with in-flight fetches.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from zarr.core.config import config as zarr_config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from zarr.core.buffer import Buffer, BufferPrototype

    from coalescing_zarr.manifest_store.planning import Span
    from coalescing_zarr.manifest_store.store import CoalescingStats


async def stream_span_members(
    spans: Sequence[Span],
    *,
    prototype: BufferPrototype,
    stats: CoalescingStats | None = None,
) -> AsyncIterator[tuple[str, Buffer]]:
    """Fetch ``spans`` concurrently; yield ``(chunk_key, buffer)`` in completion order.

    Concurrency is bounded by zarr's ``async.concurrency`` so coalescing never
    fans out wider than the stock per-chunk read would. Each span is one range
    GET; its member chunks are sliced out as zero-copy views and released together
    when that GET completes.

    If ``stats`` is given, records the pure fetch wall (first GET dispatched to
    last byte in) into ``stats.download_seconds`` — decode-independent even
    though members are yielded to a consumer that decodes between pulls.
    """
    if not spans:
        return

    concurrency = int(zarr_config.get("async.concurrency"))
    sem = asyncio.Semaphore(concurrency)
    starts: list[float] = []
    ends: list[float] = []

    async def fetch(span: Span) -> tuple[Span, Any]:
        async with sem:
            # Time the actual GET (after acquiring the semaphore slot).
            starts.append(time.perf_counter())
            # obstore returns a zero-copy buffer-protocol object; we slice views
            # out of it below without ever copying the span bytes.
            raw = await span.store.get_range_async(
                span.path, start=span.start, end=span.end
            )
            ends.append(time.perf_counter())
        return span, raw

    tasks = [asyncio.create_task(fetch(span)) for span in spans]
    try:
        for completed in asyncio.as_completed(tasks):
            span, raw = await completed
            view = memoryview(raw)
            for member in span.members:
                rel = member.offset - span.start
                chunk_view = view[rel : rel + member.length]
                yield member.key, prototype.buffer.from_bytes(chunk_view)
        if stats is not None and ends:
            stats.download_seconds += max(ends) - min(starts)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
