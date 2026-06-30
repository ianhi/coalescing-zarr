"""Store-level range coalescing for virtual Zarr stores."""

from __future__ import annotations

from coalescing_zarr.config import (
    register_pipeline,
    settings,
    use_default_pipeline,
)
from coalescing_zarr.pipeline import CachedGetter, CoalescingCodecPipeline
from coalescing_zarr.planning import ResolvedChunk, Span, plan_spans
from coalescing_zarr.store import CoalescingManifestStore, CoalescingStats

__all__ = [
    "CachedGetter",
    "CoalescingCodecPipeline",
    "CoalescingManifestStore",
    "CoalescingStats",
    "ResolvedChunk",
    "Span",
    "plan_spans",
    "register_pipeline",
    "settings",
    "use_default_pipeline",
]
