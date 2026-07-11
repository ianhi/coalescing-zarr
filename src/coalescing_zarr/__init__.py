"""Store-level range coalescing for virtualized Zarr data.

The main entry points open Icechunk data through the coalescing codec pipeline,
which drives a store's native ``get_many_chunks``. The pure-Python fallback store
(for when you can't install the forked icechunk) lives in the
:mod:`coalescing_zarr.manifest_store` subpackage.
"""

from __future__ import annotations

from coalescing_zarr.config import (
    DEFAULT_MAX_COALESCED_BYTES,
    DEFAULT_MAX_GAP,
    register_pipeline,
    use_default_pipeline,
)
from coalescing_zarr.dataset import open_zarr_coalesced
from coalescing_zarr.pipeline import CachedGetter, CoalescingCodecPipeline
from coalescing_zarr.region import read_region

__all__ = [
    "DEFAULT_MAX_COALESCED_BYTES",
    "DEFAULT_MAX_GAP",
    "CachedGetter",
    "CoalescingCodecPipeline",
    "open_zarr_coalesced",
    "read_region",
    "register_pipeline",
    "use_default_pipeline",
]
