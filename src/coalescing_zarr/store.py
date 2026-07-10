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

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from virtualizarr.manifests import ManifestArray, ManifestGroup
from virtualizarr.manifests.store import ManifestStore, _get_deepest_group_or_array
from virtualizarr.manifests.utils import parse_manifest_index

from coalescing_zarr.config import DEFAULT_MAX_COALESCED_BYTES, DEFAULT_MAX_GAP
from coalescing_zarr.fetch import stream_span_members
from coalescing_zarr.planning import ResolvedChunk, plan_spans

if TYPE_CHECKING:
    import icechunk
    from obspec_utils.registry import ObjectStoreRegistry
    from zarr.core.buffer import Buffer, BufferPrototype

# Keys ending in any of these are metadata/group documents, not data chunks.
_METADATA_SUFFIXES = ("zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata")


@dataclass
class CoalescingStats:
    """Per-store counters, handy for tests and benchmarks.

    These count what the coalescing layer actually issued — *not* what the
    underlying object store did. ``spans`` is the number of coalesced range
    requests; ``over_read_bytes`` is bytes fetched but never handed back.
    """

    calls: int = 0
    chunks_requested: int = 0
    spans: int = 0
    useful_bytes: int = 0
    over_read_bytes: int = 0
    resolve_seconds: float = 0.0
    """Time resolving chunk keys to byte ranges (in the store backend)."""
    coalesce_seconds: float = 0.0
    """Time in ``plan_spans`` (our coalescing algorithm)."""
    download_seconds: float = 0.0
    """Pure download wall: first GET dispatched to last byte in (decode-independent)."""

    def reset(self) -> None:
        self.calls = 0
        self.chunks_requested = 0
        self.spans = 0
        self.useful_bytes = 0
        self.over_read_bytes = 0
        self.resolve_seconds = 0.0
        self.coalesce_seconds = 0.0
        self.download_seconds = 0.0


class CoalescingManifestStore(ManifestStore):
    """A ManifestStore with a bulk, streaming ``get_many_chunks`` method.

    The coalescing knobs are per-store policy: ``max_gap`` (unwanted bytes
    tolerated between two chunks before merging their requests) and
    ``max_coalesced_bytes`` (hard cap on one request). They are plain public
    attributes so a benchmark or test can vary them per store, and so the
    eventual Rust port has an obvious home for the same policy.
    """

    def __init__(
        self,
        group: ManifestGroup,
        *,
        registry: ObjectStoreRegistry[Any] | None = None,
        max_gap: int = DEFAULT_MAX_GAP,
        max_coalesced_bytes: int | None = DEFAULT_MAX_COALESCED_BYTES,
    ) -> None:
        super().__init__(group, registry=registry)
        self.max_gap = max_gap
        self.max_coalesced_bytes = max_coalesced_bytes
        self.stats = CoalescingStats()

    @classmethod
    def from_manifest_store(
        cls,
        store: ManifestStore,
        *,
        max_gap: int = DEFAULT_MAX_GAP,
        max_coalesced_bytes: int | None = DEFAULT_MAX_COALESCED_BYTES,
    ) -> CoalescingManifestStore:
        """Wrap an existing VirtualiZarr ``ManifestStore``, reusing its manifest.

        Coalescing needs the same resolved manifest (group) and object-store
        registry the plain store already holds; this just re-homes them under the
        bulk ``get_many_chunks`` path with the given knobs.
        """
        return cls(
            store._group,
            registry=store._registry,
            max_gap=max_gap,
            max_coalesced_bytes=max_coalesced_bytes,
        )

    @classmethod
    def from_icechunk_session(
        cls,
        session: icechunk.Session,
        registry: ObjectStoreRegistry[Any],
        *,
        native_chunks_prefix: str,
        group: str | None = None,
        skip_variables: Any = None,
        max_gap: int = DEFAULT_MAX_GAP,
        max_coalesced_bytes: int | None = DEFAULT_MAX_COALESCED_BYTES,
    ) -> CoalescingManifestStore:
        """Build a coalescing store from an open Icechunk ``Session``.

        Parses the session's manifest into a ``ManifestStore`` (via VirtualiZarr's
        ``IcechunkParser``) and wraps it. ``registry`` must hold obstores for the
        *backing* data the manifest points at (e.g. the source S3 bucket) — that
        is what coalescing actually fetches, so it cannot be inferred from the
        session. ``native_chunks_prefix`` locates Icechunk's own (non-virtual)
        chunks, per the ``IcechunkParser`` contract.
        """
        from virtualizarr.parsers import IcechunkParser

        ms = IcechunkParser(group=group, skip_variables=skip_variables).parse_session(
            session, registry=registry, native_chunks_prefix=native_chunks_prefix
        )
        return cls.from_manifest_store(
            ms, max_gap=max_gap, max_coalesced_bytes=max_coalesced_bytes
        )

    def _resolve(self, key: str) -> ResolvedChunk | None:
        """Resolve a chunk key to a byte range, without fetching.

        Returns ``None`` if the key is not a present data chunk (missing entry,
        a metadata key, or an inlined chunk — inlined chunks are not coalescable
        and are left to the regular ``get`` path).
        """
        node, suffix = _get_deepest_group_or_array(self._group, key)
        # Only data chunks are coalescable; metadata and group keys are not.
        if suffix.endswith(_METADATA_SUFFIXES):
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

        # ``resolve`` finds the object store for this file's URL and returns the
        # store-relative path as its second element — the same prefix-stripping
        # ManifestStore.get does, so we reuse it rather than recompute it. It
        # raises ValueError itself if no store matches.
        store, path_in_store = self._registry.resolve(entry["path"])
        return ResolvedChunk(
            key=key,
            store=store,
            path=str(path_in_store),
            offset=int(entry["offset"]),
            length=int(entry["length"]),
        )

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

        ``max_gap`` / ``max_coalesced_bytes`` default to the store's configured
        policy; pass them to override per call.
        """
        if max_gap is None:
            max_gap = self.max_gap
        if max_coalesced_bytes is None:
            max_coalesced_bytes = self.max_coalesced_bytes

        self.stats.calls += 1
        self.stats.chunks_requested += len(keys)

        resolved: list[ResolvedChunk] = []
        for key in keys:
            rc = self._resolve(key)
            if rc is None:
                # Uncoalescable (an inlined chunk) or genuinely missing.
                # Delegate to the stock single-key get, which returns the
                # inlined bytes or None. Yielding None unconditionally here would
                # turn inlined chunks — real data — into fill values: silent
                # corruption.
                yield key, await self.get(key, prototype=prototype)
            else:
                resolved.append(rc)

        t0 = time.perf_counter()
        spans = plan_spans(
            resolved, max_gap=max_gap, max_coalesced_bytes=max_coalesced_bytes
        )
        self.stats.coalesce_seconds += time.perf_counter() - t0
        self.stats.spans += len(spans)
        for span in spans:
            useful = span.useful_bytes
            self.stats.useful_bytes += useful
            self.stats.over_read_bytes += span.nbytes - useful

        # Fetch spans concurrently and stream members in completion order (shared
        # with the Icechunk-native store — see ``fetch.stream_span_members``).
        async for key, buf in stream_span_members(
            spans, prototype=prototype, stats=self.stats
        ):
            yield key, buf
