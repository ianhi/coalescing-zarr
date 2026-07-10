"""``open_zarr_coalesced`` — open an Icechunk session as an xarray Dataset that
reads through the coalescing pipeline, with a bulk-friendly chunking default so
the fast path fires without the caller knowing anything about coalescing.

This is ``xarray.open_zarr`` with two things arranged for you:

1. **The coalescing codec pipeline is active** for the open. Reads through the
   returned Dataset call ``get_many_chunks`` on the (native Icechunk) store, so a
   read is fetched as one bulk, range-coalesced request and decoded on arrival.
   No store wrapper is needed — the pipeline detects a native Icechunk store and
   adapts to its native getter directly.
2. **Chunking is set so a read spans many chunks.** This is the part
   ``xarray.open_zarr`` cannot do for you, and the reason this function exists:
   the default xarray open makes an array dask-backed *at its native chunk grid*,
   so a read is one dask task per chunk — the coalescing pipeline then sees only
   one chunk at a time and can't coalesce, and the per-chunk task overhead (the
   very cost this targets) returns. ``chunks=None`` (the default here) reads
   eagerly in one shot; ``chunks={"time": N}`` stays lazy/dask-backed while
   batching many chunks per task.

Limitation: a plain ``xarray.open_zarr(session.store)`` will *not* coalesce even
with the pipeline registered globally, for the chunking reason above. Fixing that
(coalescing across the native chunk grid without an explicit coarse ``chunks``)
is future work; for now, open through this function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import zarr

from coalescing_zarr.config import register_pipeline, use_default_pipeline
from coalescing_zarr.icechunk_native import _MISSING_NATIVE_MSG
from coalescing_zarr.pipeline import (
    DEFAULT_MAX_COALESCED_BYTES,
    DEFAULT_MAX_GAP,
    MAX_COALESCED_BYTES_KEY,
    MAX_GAP_KEY,
)

if TYPE_CHECKING:
    import icechunk
    import xarray as xr

__all__ = ["open_zarr_coalesced"]


def open_zarr_coalesced(
    session: icechunk.Session,
    *,
    chunks: Any = None,
    max_gap: int = DEFAULT_MAX_GAP,
    max_coalesced_bytes: int | None = DEFAULT_MAX_COALESCED_BYTES,
    **open_zarr_kwargs: Any,
) -> xr.Dataset:
    """Open an Icechunk ``session`` as a coalescing-backed xarray ``Dataset``.

    Ordinary xarray then works unchanged (``ds[var].sel(...).compute()``); reads
    go through the coalescing codec pipeline over the session's native store.

    Parameters
    ----------
    session
        An open ``icechunk.Session`` (e.g. ``repo.readonly_session("main")``)
        whose store exposes the native ``get_many_chunks`` (see the README
        "Requirements": this needs a forked icechunk build).
    chunks
        Passed to :func:`xarray.open_zarr`. **Default ``None``** reads eagerly
        (numpy backed) so each read is one bulk request — the fast path. Use
        ``chunks={"time": N}`` (a block larger than the native chunk grid) to stay
        lazy/dask-backed while still batching many chunks per task. Avoid
        ``chunks={}`` / the xarray default: one dask task per native chunk, which
        reintroduces the per-chunk overhead and prevents coalescing.
    max_gap, max_coalesced_bytes
        Coalescing policy. ``max_gap`` is the most unwanted bytes tolerated
        between two chunks before merging their range GETs; ``max_coalesced_bytes``
        caps one merged request. They are bound into the pipeline at open time
        (via zarr config) and captured by the Dataset's arrays.
    **open_zarr_kwargs
        Forwarded to :func:`xarray.open_zarr` (``group``, ``mask_and_scale``, ...).
        ``zarr_format=3`` and ``consolidated=False`` are set unless overridden.

    Returns
    -------
    xarray.Dataset
        Backed by the coalescing pipeline. The arrays capture the pipeline (and
        the knobs) at open time, so the global pipeline is reset before returning
        — other stores opened later are unaffected.
    """
    import xarray as xr

    store = session.store
    if not hasattr(store, "get_many_chunks"):
        raise NotImplementedError(_MISSING_NATIVE_MSG)

    open_zarr_kwargs.setdefault("zarr_format", 3)
    open_zarr_kwargs.setdefault("consolidated", False)

    # Set the knobs in config only for the open window: the pipeline instance
    # zarr builds for each array snapshots them in its __init__, so the reset
    # afterwards doesn't lose them, and other backends stay unaffected.
    with zarr.config.set(
        {MAX_GAP_KEY: max_gap, MAX_COALESCED_BYTES_KEY: max_coalesced_bytes}
    ):
        register_pipeline()  # arrays capture the pipeline at open time...
        try:
            ds = xr.open_zarr(store, chunks=chunks, **open_zarr_kwargs)
        finally:
            use_default_pipeline()  # ...so reset immediately
    return cast("xr.Dataset", ds)
