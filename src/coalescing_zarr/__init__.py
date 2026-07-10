"""Store-level range coalescing for virtualized Zarr data."""

from __future__ import annotations

from coalescing_zarr.config import (
    DEFAULT_MAX_COALESCED_BYTES,
    DEFAULT_MAX_GAP,
    register_pipeline,
    use_default_pipeline,
)
from coalescing_zarr.dataset import open_zarr_coalesced
from coalescing_zarr.pipeline import CachedGetter, CoalescingCodecPipeline
from coalescing_zarr.planning import ResolvedChunk, Span, plan_spans
from coalescing_zarr.region import read_region
from coalescing_zarr.store import CoalescingManifestStore, CoalescingStats

__all__ = [
    "DEFAULT_MAX_COALESCED_BYTES",
    "DEFAULT_MAX_GAP",
    "CachedGetter",
    "CoalescingCodecPipeline",
    "CoalescingManifestStore",
    "CoalescingStats",
    "ResolvedChunk",
    "Span",
    "open_zarr_coalesced",
    "plan_spans",
    "read_region",
    "register_pipeline",
    "use_default_pipeline",
]
