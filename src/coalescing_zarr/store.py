"""``CoalescingManifestStore`` — a ManifestStore that can serve many chunks at once.

This subclasses VirtualiZarr's :class:`~virtualizarr.manifests.ManifestStore`
(which already knows how to resolve a chunk key to a byte range in a backing
file via the manifest, and fetch it through obstore) and adds one method:
``get_many_chunks``. That method resolves all requested keys, plans coalesced
byte-range spans (:func:`coalescing_zarr.planning.plan_spans`), fetches the
spans concurrently, and streams the per-chunk bytes back **in completion
order** so the caller can decode each chunk the instant it arrives.

The resolution logic in :meth:`_resolve` mirrors ``ManifestStore.get`` but stops
just before fetching — this is the "derive the effective shard index" step. In
the eventual Icechunk-native implementation this all happens in Rust over the
in-memory manifest; here it is plain Python, which is exactly the per-key
overhead the design warns about (see ``design.md`` §Open questions).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from virtualizarr.manifests import ManifestArray, ManifestGroup
from virtualizarr.manifests.store import ManifestStore, _get_deepest_group_or_array
from virtualizarr.manifests.utils import parse_manifest_index
from zarr.core.config import config as zarr_config

from coalescing_zarr.planning import ResolvedChunk, Span, plan_spans

if TYPE_CHECKING:
    from obspec_utils.registry import ObjectStoreRegistry
    from zarr.core.buffer import Buffer, BufferPrototype


@dataclass
class CoalescingStats:
    """Per-store counters, handy for tests and benchmarks.

    These count what the coalescing layer actually issued — *not* what the
    underlying object store did. ``source_gets`` is the number of coalesced
    range requests; ``over_read_bytes`` is bytes fetched but never handed back.
    """

    calls: int = 0
    chunks_requested: int = 0
    spans: int = 0
    useful_bytes: int = 0
    over_read_bytes: int = 0

    def reset(self) -> None:
        self.calls = 0
        self.chunks_requested = 0
        self.spans = 0
        self.useful_bytes = 0
        self.over_read_bytes = 0


class CoalescingManifestStore(ManifestStore):
    """A ManifestStore with a bulk, streaming ``get_many_chunks`` method."""

    def __init__(
        self,
        group: ManifestGroup,
        *,
        registry: ObjectStoreRegistry[Any] | None = None,
    ) -> None:
        super().__init__(group, registry=registry)
        self.stats = CoalescingStats()

    def _resolve(self, key: str) -> ResolvedChunk | None:
        """Resolve a chunk key to a byte range, without fetching.

        Returns ``None`` if the key is not a present data chunk (missing entry,
        a metadata key, or an inlined chunk — inlined chunks are not coalescable
        and are left to the regular ``get`` path).
        """
        node, suffix = _get_deepest_group_or_array(self._group, key)
        # Only data chunks are coalescable; metadata and group keys are not.
        metadata_suffixes = ("zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata")
        if suffix.endswith(metadata_suffixes):
            return None
        if not isinstance(node, ManifestArray):
            return None
        manifest = node.manifest

        separator: Literal[".", "/"] = getattr(
            node.metadata.chunk_key_encoding, "separator", "."
        )
        chunk_indexes = parse_manifest_index(key, separator, expand_pattern=True)
        if chunk_indexes in manifest._inlined:
            return None

        entry = manifest.get_entry(chunk_indexes)
        if entry is None:
            return None
        path = entry["path"]
        offset = int(entry["offset"])
        length = int(entry["length"])

        store, _ = self._registry.resolve(path)
        if not store:
            raise ValueError(f"No store registered for {path}")
        path_in_store = self._path_in_store(store, path)
        return ResolvedChunk(
            key=key,
            store=store,
            path=path_in_store,
            offset=offset,
            length=length,
        )

    @staticmethod
    def _path_in_store(store: object, path: str) -> str:
        # Mirrors ManifestStore.get: strip the store's own prefix/url path so we
        # are left with the file path the object store expects.
        path_in_store = urlparse(path).path
        store_prefix = getattr(store, "prefix", None)
        store_url = getattr(store, "url", None)
        if store_prefix:
            prefix = str(store_prefix).lstrip("/")
        elif store_url:
            prefix = urlparse(str(store_url)).path.lstrip("/")
        else:
            prefix = ""
        return path_in_store.lstrip("/").removeprefix(prefix).lstrip("/")

    async def get_many_chunks(
        self,
        keys: Sequence[str],
        *,
        prototype: BufferPrototype,
        max_gap: int | None = None,
        max_coalesced_bytes: int | None = None,
    ) -> AsyncIterator[tuple[str, Buffer | None]]:
        """Fetch many chunks, coalescing nearby ranges; yield in completion order.

        Yields ``(key, buffer)`` pairs as the bytes for each key become
        available. ``buffer`` is ``None`` for a missing/uncoalescable key. The
        iteration order is *not* the input order — it is whatever order the
        underlying span fetches complete in — so the consumer can start decoding
        the first chunk without waiting for the slowest fetch.
        """
        from coalescing_zarr import config as _cfg

        if max_gap is None:
            max_gap = _cfg.settings.max_gap
        if max_coalesced_bytes is None:
            max_coalesced_bytes = _cfg.settings.max_coalesced_bytes

        self.stats.calls += 1
        self.stats.chunks_requested += len(keys)

        resolved: list[ResolvedChunk] = []
        for key in keys:
            rc = self._resolve(key)
            if rc is None:
                # Missing or uncoalescable: hand back None immediately.
                yield key, None
            else:
                resolved.append(rc)

        spans = plan_spans(
            resolved, max_gap=max_gap, max_coalesced_bytes=max_coalesced_bytes
        )
        self.stats.spans += len(spans)
        for span in spans:
            self.stats.useful_bytes += span.useful_bytes
            self.stats.over_read_bytes += span.over_read

        if not spans:
            return

        # Bound fetch concurrency by the same knob zarr uses for its per-chunk
        # fan-out, so coalescing never *reduces* concurrency below the baseline.
        concurrency = int(zarr_config.get("async.concurrency"))
        sem = asyncio.Semaphore(concurrency)

        async def fetch(span: Span) -> tuple[Span, bytes]:
            async with sem:
                raw = await span.store.get_range_async(
                    span.path, start=span.start, end=span.end
                )
            return span, bytes(raw)

        tasks = [asyncio.create_task(fetch(span)) for span in spans]
        try:
            for completed in asyncio.as_completed(tasks):
                span, raw = await completed
                view = memoryview(raw)
                for member in span.members:
                    rel = member.offset - span.start
                    chunk_bytes = bytes(view[rel : rel + member.length])
                    yield member.key, prototype.buffer.from_bytes(chunk_bytes)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
