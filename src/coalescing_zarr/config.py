"""Default coalescing knobs and pipeline registration.

The two knobs are intentionally minimal for the MVP. ``max_gap`` is the lever
the design calls out as *two* tensions at once (round-trips vs over-read, and
round-trips vs pipelineability); ``max_coalesced_bytes`` is a safety cap. These
are only *defaults* — actual policy lives per-store on
:class:`~coalescing_zarr.store.CoalescingManifestStore`, so a future cost model
slots in there (or as an injected planner) rather than as a global swap.
"""

from __future__ import annotations

import zarr
from zarr.registry import register_pipeline as _register_pipeline_class

from coalescing_zarr.pipeline import CoalescingCodecPipeline

# 256 KiB matches the gap used in the prior NDPI measurements; 0 would mean
# "merge only strictly adjacent chunks" (zero over-read, more round-trips).
DEFAULT_MAX_GAP = 256 * 1024
DEFAULT_MAX_COALESCED_BYTES: int | None = None

PIPELINE_PATH = "coalescing_zarr.pipeline.CoalescingCodecPipeline"

# Make the class resolvable by ``codec_pipeline.path``. Registering only adds it
# to the registry; it does not become active until ``register_pipeline()`` (or a
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
