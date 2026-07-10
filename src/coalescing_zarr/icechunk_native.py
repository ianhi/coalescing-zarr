"""Adapt Icechunk's native ``get_many_chunks`` to the pipeline's chunk stream.

Icechunk's zarr store exposes a native bulk getter that resolves + coalesces +
fetches through Icechunk's own client (virtual *and* native chunks, across
arrays) and streams ``(request_index, bytes)`` in completion order. Its request
shape is ``(array_path, coords)`` tuples, not zarr chunk *keys*. This module is
the thin translation layer: chunk keys -> native requests, and the native
``(index, bytes)`` stream back into the ``(key, buffer)`` pairs the
:class:`~coalescing_zarr.pipeline.CoalescingCodecPipeline` decodes from.

Keeping this here (rather than in a wrapping ``Store`` the caller must construct)
is what lets a plain ``session.store`` be read through the coalescing pipeline
with no wrapper — the pipeline detects a native store and calls this directly.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from zarr.abc.store import Store
    from zarr.core.buffer import Buffer, BufferPrototype

# zarr v3 chunk keys carry a literal "c" component before the grid coords, e.g.
# "group/array/c/0/3/1" (sep "/") or "group/array/c.0.3.1" (sep ".").
_COORD_SPLIT = re.compile(r"[./]")

# Shown when the installed icechunk lacks the native bulk getter this needs.
# get_many_chunks is not in a released icechunk yet (see README "Requirements").
_MISSING_NATIVE_MSG = (
    "This installed icechunk has no IcechunkStore.get_many_chunks, which "
    "coalescing needs (native bulk coalesced reads). It is not in a released "
    "icechunk yet.\n"
    "  - Contributors: `uv sync` in this repo builds the required fork "
    "automatically (see [tool.uv.sources] in pyproject.toml).\n"
    "  - Otherwise: install the forked icechunk build first — see the README "
    "'Requirements' section — then reinstall this package."
)


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


def is_native_icechunk_store(store: object) -> bool:
    """True if ``store`` is an Icechunk store exposing the native bulk getter.

    Distinguishes the native path (adapted by :func:`stream_icechunk_chunks`)
    from the pure-Python ``CoalescingManifestStore``, which is not an Icechunk
    store but exposes its own ``get_many_chunks``.
    """
    try:
        import icechunk
    except ImportError:  # pragma: no cover - icechunk is a hard dependency
        return False
    return isinstance(store, icechunk.IcechunkStore) and hasattr(
        store, "get_many_chunks"
    )


async def stream_icechunk_chunks(
    store: Store,
    keys: Sequence[str],
    *,
    prototype: BufferPrototype,
    max_gap: int,
    max_coalesced_bytes: int | None,
) -> AsyncIterator[tuple[str, Buffer | None]]:
    """Fetch ``keys`` via Icechunk's native getter; yield ``(key, buffer)``.

    Chunk keys become ``(array_path, coords)`` requests (possibly spanning
    arrays); Icechunk coalesces by backing object and streams
    ``(request_index, bytes | None)`` in completion order, which we re-key to the
    original chunk key. Metadata / non-chunk keys fall back to a single-key
    ``get`` (they should not appear on the array read path, but yielding ``None``
    for real data would silently corrupt it).
    """
    requests: list[tuple[str, tuple[int, ...]]] = []
    key_by_index: list[str] = []
    for key in keys:
        split = _split_chunk_key(key)
        if split is None:
            yield key, await store.get(key, prototype=prototype)
            continue
        requests.append(split)
        key_by_index.append(key)

    if not requests:
        return

    chunks = cast("Any", store).get_many_chunks(
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
