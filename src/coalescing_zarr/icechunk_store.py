"""``CoalescingIcechunkStore`` — bulk coalesced reads via Icechunk's native getter.

Wraps the Icechunk zarr store: metadata/listing delegate straight through, and
``get_many_chunks`` maps the batch's chunk keys to ``(array_path, coords)`` and
hands them to Icechunk's **native** ``get_many_chunks``, which resolves +
coalesces + fetches through Icechunk's own client (virtual *and* native chunks,
across arrays) and streams ``(request_index, bytes)`` in completion order. We
just re-key those to ``(chunk_key, buffer)`` for the codec pipeline.

This replaces the earlier Python path (resolve_chunk_refs -> plan_spans ->
obstore fetch), which serialized resolve before download and used a second,
un-tuned client. Requires an Icechunk build exposing
``IcechunkStore.get_many_chunks``; :meth:`from_session` fails loudly otherwise.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from zarr.abc.store import Store

from coalescing_zarr.config import DEFAULT_MAX_COALESCED_BYTES, DEFAULT_MAX_GAP
from coalescing_zarr.store import CoalescingStats

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    import icechunk
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype

# zarr v3 chunk keys carry a literal "c" component before the grid coords, e.g.
# "group/array/c/0/3/1" (sep "/") or "group/array/c.0.3.1" (sep ".").
_COORD_SPLIT = re.compile(r"[./]")


def _split_chunk_key(key: str) -> tuple[str, tuple[int, ...]] | None:
    """Split a chunk key into ``(array_path, coords)``; ``None`` if not a chunk.

    Returns ``None`` for metadata keys (``zarr.json``) and anything that doesn't
    look like ``.../c/<i>/<j>/...`` so the caller falls back to a plain ``get``.
    """
    parts = key.split("/")
    # "/"-separated coords: a standalone "c" component, coords after it.
    if "c" in parts[1:]:
        ci = len(parts) - 1 - parts[::-1].index("c")
        coord_parts = parts[ci + 1 :]
        if coord_parts and all(p.lstrip("-").isdigit() for p in coord_parts):
            return "/".join(parts[:ci]), tuple(int(p) for p in coord_parts)
    # "."-separated coords: last component like "c.0.3.1".
    if parts[-1].startswith("c.") and len(parts[-1]) > 2:
        coord_parts = _COORD_SPLIT.split(parts[-1])[1:]
        if coord_parts and all(p.lstrip("-").isdigit() for p in coord_parts):
            return "/".join(parts[:-1]), tuple(int(p) for p in coord_parts)
    return None


class CoalescingIcechunkStore(Store):
    """Read-only zarr store over an Icechunk session, with bulk coalesced reads.

    Everything except chunk data (metadata reads, listing) is delegated to the
    wrapped Icechunk store unchanged. ``get_many_chunks`` is the coalescing path
    the :class:`~coalescing_zarr.pipeline.CoalescingCodecPipeline` calls; it just
    adapts Icechunk's native bulk getter to the pipeline's ``(key, buffer)`` form.
    """

    def __init__(
        self,
        inner: Store,
        *,
        max_gap: int = DEFAULT_MAX_GAP,
        max_coalesced_bytes: int | None = DEFAULT_MAX_COALESCED_BYTES,
    ) -> None:
        super().__init__(read_only=True)
        self._inner = inner
        self.max_gap = max_gap
        self.max_coalesced_bytes = max_coalesced_bytes
        self.stats = CoalescingStats()

    @classmethod
    def from_session(
        cls,
        session: icechunk.Session,
        *,
        max_gap: int = DEFAULT_MAX_GAP,
        max_coalesced_bytes: int | None = DEFAULT_MAX_COALESCED_BYTES,
    ) -> CoalescingIcechunkStore:
        """Wrap an open Icechunk ``Session``.

        Icechunk's native getter fetches virtual chunks through the repo's own
        (authorized) virtual-chunk containers, so no external registry is needed.
        Fails loudly if the running Icechunk lacks ``get_many_chunks``.
        """
        if not hasattr(session.store, "get_many_chunks"):
            raise NotImplementedError(
                "CoalescingIcechunkStore needs icechunk.IcechunkStore."
                "get_many_chunks (native coalescing). Point this project's "
                "icechunk dependency at a build that has it and rebuild "
                "(uv sync --reinstall-package icechunk)."
            )
        return cls(
            session.store,
            max_gap=max_gap,
            max_coalesced_bytes=max_coalesced_bytes,
        )

    async def get_many_chunks(
        self,
        keys: Sequence[str],
        *,
        prototype: BufferPrototype,
        max_gap: int | None = None,
        max_coalesced_bytes: int | None = None,
    ) -> AsyncIterator[tuple[str, Buffer | None]]:
        """Delegate to Icechunk's native bulk getter; yield ``(key, buffer)``.

        Chunk keys become ``(array_path, coords)`` requests (possibly spanning
        arrays); Icechunk coalesces by backing object and streams
        ``(request_index, bytes | None)`` in completion order. Metadata / non-chunk
        keys fall back to a single-key ``get``.
        """
        if max_gap is None:
            max_gap = self.max_gap
        if max_coalesced_bytes is None:
            max_coalesced_bytes = self.max_coalesced_bytes

        self.stats.calls += 1
        self.stats.chunks_requested += len(keys)

        requests: list[tuple[str, tuple[int, ...]]] = []
        key_by_index: list[str] = []
        for key in keys:
            split = _split_chunk_key(key)
            if split is None:
                yield key, await self.get(key, prototype=prototype)
                continue
            requests.append(split)
            key_by_index.append(key)

        if not requests:
            return

        chunks = self._inner.get_many_chunks(
            requests,
            max_gap=max_gap,
            max_coalesced_bytes=max_coalesced_bytes,
        )
        try:
            async for index, data in chunks:
                buf = None if data is None else prototype.buffer.from_bytes(data)
                yield key_by_index[index], buf
        finally:
            aclose = getattr(chunks, "aclose", None)
            if aclose is not None:
                await aclose()

    # --- zarr Store protocol: read paths delegate to the wrapped store ---------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        return await self._inner.get(key, prototype, byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return await self._inner.get_partial_values(prototype, key_ranges)

    async def exists(self, key: str) -> bool:
        return await self._inner.exists(key)

    def list(self) -> AsyncIterator[str]:
        return self._inner.list()

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        return self._inner.list_prefix(prefix)

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        return self._inner.list_dir(prefix)

    # --- write paths: read-only ------------------------------------------------

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("CoalescingIcechunkStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("CoalescingIcechunkStore is read-only")

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return self._inner.supports_listing

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CoalescingIcechunkStore)
            and other._inner == self._inner
            and other.max_gap == self.max_gap
            and other.max_coalesced_bytes == self.max_coalesced_bytes
        )

    def __hash__(self) -> int:
        return hash((id(self._inner), self.max_gap, self.max_coalesced_bytes))
