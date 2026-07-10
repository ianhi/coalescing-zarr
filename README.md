# coalescing-zarr

Store-level range coalescing for virtual Zarr stores.

Reading a region of a virtual array fetches **one chunk per request** — correct,
but latency-bound when many small chunks live in the same backing file. This
package coalesces those fetches into a few larger range requests, trading a
little over-read for far fewer round-trips, while preserving zarr's
fetch↔decode overlap: bytes stream back in completion order and each chunk is
decoded the moment it arrives.

On overhead-bound reads (e.g. Met Office HDF5 with thousands of tiny chunks) this
is commonly an ~8× speedup over a stock read, from one line.

## Quickstart

```python
import icechunk
from coalescing_zarr import open_coalesced

repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")

# A normal xarray Dataset — but reads go through the coalescing pipeline.
ds = open_coalesced(session)
region = ds["temperature"].sel(time="2024-01-01").compute()
```

`open_coalesced` returns an ordinary `xarray.Dataset`; everything downstream
(`.sel`, `.compute`, plotting) is unchanged. It defaults to eager reads
(`chunks=None`) so each array read is one bulk, range-coalesced request. Pass
`chunks={"time": N}` to stay lazy/dask-backed while still batching many chunks
per task. Avoid the plain xarray default (`chunks={}`) — that is one dask task
per native chunk, which reintroduces the per-chunk overhead and prevents
coalescing.

### Reading across arrays in one call

`read_region` issues a single bulk coalesced read spanning several arrays that
share a grid (e.g. GOES bands), then decodes and assembles each:

```python
from coalescing_zarr import read_region

out = read_region(session, ["grp/CMI_C01", "grp/CMI_C02"], (slice(0, 512), slice(0, 512)))
# {"grp/CMI_C01": ndarray, "grp/CMI_C02": ndarray}
```

### Tuning

Both entry points take the same two knobs:

- **`max_gap`** (default 256 KiB) — the most unwanted bytes tolerated *between*
  two chunks before their range GETs are merged. `0` merges only strictly
  adjacent chunks (zero over-read, more round-trips); larger values trade
  over-read for fewer round-trips.
- **`max_coalesced_bytes`** (default unbounded) — hard cap on a single merged
  request, so one pathological run can't produce an enormous GET.

## Requirements

The fast path uses **Icechunk's native `get_many_chunks`**, which is **not in a
released icechunk yet**. If it's missing, `open_coalesced` / `read_region` raise
a `NotImplementedError` that says so.

- **Contributors** get the required build automatically: `uv sync` in this repo
  builds the forked icechunk pinned in `[tool.uv.sources]`. Nothing else to do.
- **Installing into your own environment:** install the forked icechunk build
  first, then this package. Because PyPI forbids git/URL dependencies, this
  requirement cannot be expressed in the published wheel's metadata — it is
  documented here instead. (A pre-built release of the fork is the intended path;
  until then, build the fork branch `ian/more-specific-vritual` of
  `earth-mover/icechunk`.)

This will collapse to a plain `pip install` against a released icechunk once the
feature ships upstream.

## Develop

```sh
uv sync            # builds the forked icechunk automatically
uv run prek install
uv run pytest
```

See [`design.md`](./design.md) for why this needs both a bulk store method and a
custom codec pipeline, and how the streaming/decode-overlap works.
