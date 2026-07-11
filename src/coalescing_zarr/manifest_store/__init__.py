"""Pure-Python coalescing store — the no-forked-icechunk fallback.

The main path in :mod:`coalescing_zarr` runs the coalescing codec pipeline over a
forked-icechunk store whose native (Rust) ``get_many_chunks`` does the range
coalescing. This subpackage is the alternative for when you can't install that
fork: a VirtualiZarr :class:`~virtualizarr.manifests.store.ManifestStore` that
implements ``get_many_chunks`` itself, in Python. Same interface, slower planner
(:func:`plan_spans` runs in Python, not Rust).

Build one with :meth:`CoalescingManifestStore.from_icechunk_session`.
"""

from __future__ import annotations

from coalescing_zarr.manifest_store.planning import ResolvedChunk, Span, plan_spans
from coalescing_zarr.manifest_store.store import (
    CoalescingManifestStore,
    CoalescingStats,
)

__all__ = [
    "CoalescingManifestStore",
    "CoalescingStats",
    "ResolvedChunk",
    "Span",
    "plan_spans",
]
