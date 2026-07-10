"""``open_coalesced`` — open an Icechunk session as an xarray Dataset that reads
through the coalescing pipeline, with a bulk-friendly default so the fast path
fires without the caller knowing anything about coalescing or chunking.

Why this exists: reading an Icechunk array through xarray the *default* way makes
the array dask-backed at its native chunk grid, so a read is split into one task
per chunk. For arrays with many small chunks (e.g. Met Office HDF5 at (1,128,128)
-> 6000 chunks) that per-chunk dask/zarr orchestration dominates the wall time
(~25 s for a full read where the actual download+decode is ~2 s), and the
coalescing pipeline never sees more than one chunk at a time so it can't coalesce.

Opening so a single read spans many chunks fixes both: the coalescing pipeline
gets the whole request at once (one bulk ``get_many_chunks``) and there is no
per-chunk task overhead. ``chunks=None`` (the default here) reads eagerly in one
shot; pass ``chunks={"time": N}`` for lazy/out-of-core reads that still batch many
chunks per dask task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from coalescing_zarr.config import (
    DEFAULT_MAX_COALESCED_BYTES,
    DEFAULT_MAX_GAP,
    register_pipeline,
    use_default_pipeline,
)
from coalescing_zarr.icechunk_store import CoalescingIcechunkStore

if TYPE_CHECKING:
    import icechunk
    import xarray as xr

__all__ = ["open_coalesced"]


def open_coalesced(
    session: icechunk.Session,
    *,
    chunks: Any = None,
    max_gap: int = DEFAULT_MAX_GAP,
    max_coalesced_bytes: int | None = DEFAULT_MAX_COALESCED_BYTES,
    **open_zarr_kwargs: Any,
) -> xr.Dataset:
    """Open an Icechunk ``session`` as an xarray ``Dataset`` via the coalescing store.

    Reads go through :class:`~coalescing_zarr.icechunk_store.CoalescingIcechunkStore`
    and the :class:`~coalescing_zarr.pipeline.CoalescingCodecPipeline`, so a single
    array read is fetched as one bulk, range-coalesced ``get_many_chunks`` call and
    decoded on arrival. Ordinary xarray then works unchanged
    (``ds[var].sel(...).compute()``).

    Parameters
    ----------
    session
        An open ``icechunk.Session`` (e.g. ``repo.readonly_session("main")``) whose
        store exposes the native ``get_many_chunks``.
    chunks
        Passed to :func:`xarray.open_zarr`. **Default ``None``** reads eagerly (numpy
        backed) so each read is one bulk request — the fast path. Use
        ``chunks={"time": N}`` (or any block larger than the native chunk grid) to
        stay lazy/dask-backed while still batching many chunks per task. Avoid
        ``chunks={}`` / the xarray default: that is one dask task per native chunk,
        which reintroduces the per-chunk overhead and prevents coalescing.
    max_gap, max_coalesced_bytes
        Coalescing policy (see ``CoalescingIcechunkStore``): ``max_gap`` is the most
        unwanted bytes tolerated between two chunks before merging their range GETs;
        ``max_coalesced_bytes`` caps one merged request.
    **open_zarr_kwargs
        Forwarded to :func:`xarray.open_zarr` (``group``, ``mask_and_scale``, ...).
        ``zarr_format=3`` and ``consolidated=False`` are set unless overridden.

    Returns
    -------
    xarray.Dataset
        Backed by the coalescing store. The arrays capture the coalescing pipeline
        at open time, so the global pipeline is reset before returning (other stores
        opened later are unaffected).
    """
    import xarray as xr

    store = CoalescingIcechunkStore.from_session(
        session, max_gap=max_gap, max_coalesced_bytes=max_coalesced_bytes
    )
    open_zarr_kwargs.setdefault("zarr_format", 3)
    open_zarr_kwargs.setdefault("consolidated", False)
    register_pipeline()  # arrays capture the pipeline at open time...
    try:
        ds = xr.open_zarr(store, chunks=chunks, **open_zarr_kwargs)
        return cast("xr.Dataset", ds)
    finally:
        use_default_pipeline()  # ...so reset immediately; other backends unaffected
