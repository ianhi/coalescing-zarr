"""Store-level range coalescing for virtual Zarr stores."""

from __future__ import annotations

from coalescing_zarr.config import (
    DEFAULT_MAX_COALESCED_BYTES,
    DEFAULT_MAX_GAP,
    register_pipeline,
    use_default_pipeline,
)
from coalescing_zarr.dataset import open_coalesced
from coalescing_zarr.icechunk_store import CoalescingIcechunkStore
from coalescing_zarr.pipeline import CachedGetter, CoalescingCodecPipeline
from coalescing_zarr.planning import ResolvedChunk, Span, plan_spans
from coalescing_zarr.region import read_region
from coalescing_zarr.store import CoalescingManifestStore, CoalescingStats

__all__ = [
    "DEFAULT_MAX_COALESCED_BYTES",
    "DEFAULT_MAX_GAP",
    "CachedGetter",
    "CoalescingCodecPipeline",
    "CoalescingIcechunkStore",
    "CoalescingManifestStore",
    "CoalescingStats",
    "ResolvedChunk",
    "Span",
    "open_coalesced",
    "plan_spans",
    "read_region",
    "register_pipeline",
    "use_default_pipeline",
]
