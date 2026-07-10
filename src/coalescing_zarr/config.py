"""Coalescing knob defaults and pipeline registration.

The knob defaults and their zarr-config keys live in
:mod:`coalescing_zarr.pipeline` (a leaf module, so the pipeline can read them
without importing this module, which registers the pipeline class). They are
re-exported here for the public ``coalescing_zarr.config`` surface.
"""

from __future__ import annotations

import zarr
from zarr.registry import register_pipeline as _register_pipeline_class

from coalescing_zarr.pipeline import (
    DEFAULT_MAX_COALESCED_BYTES,
    DEFAULT_MAX_GAP,
    MAX_COALESCED_BYTES_KEY,
    MAX_GAP_KEY,
    CoalescingCodecPipeline,
)

__all__ = [
    "DEFAULT_MAX_COALESCED_BYTES",
    "DEFAULT_MAX_GAP",
    "MAX_COALESCED_BYTES_KEY",
    "MAX_GAP_KEY",
    "PIPELINE_PATH",
    "register_pipeline",
    "use_default_pipeline",
]

PIPELINE_PATH = "coalescing_zarr.pipeline.CoalescingCodecPipeline"

# Make the class resolvable by ``codec_pipeline.path``. Registering only adds it
# to the registry; it becomes active only once ``register_pipeline()`` (or a
# direct ``zarr.config.set``) points the config at it.
_register_pipeline_class(CoalescingCodecPipeline)


def register_pipeline() -> None:
    """Install the coalescing codec pipeline as zarr's default."""
    zarr.config.set({"codec_pipeline.path": PIPELINE_PATH})


def use_default_pipeline() -> None:
    """Restore zarr's built-in batched codec pipeline."""
    zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    )
