# coalescing-zarr

Store-level range coalescing for **virtualized** Zarr data.

The use case is virtualization: an Icechunk repo whose chunks are virtual
references into original archive files (HDF5 / NetCDF / TIFF), where one array is
often thousands of small chunks packed into a single backing object. zarr's read
path fetches **one chunk per request** — correct, but latency-bound when the
chunks are small and co-located.

This package moves the fetch decision down to the store, where the layout is
known. It works at three levels:

- **Codec pipeline (this package).** Installs a zarr `CodecPipeline` that hands
  the store the *whole* set of chunk keys for a read at once — calling
  `get_many_chunks` on the store — instead of one `get` per chunk. It then
  decodes each chunk the moment its bytes arrive.
- **Store (Icechunk).** Seeing the whole batch, the store can optimize how it
  fetches from storage: chunks that sit near each other in the same backing
  object are coalesced into a few large range requests — the optimization zarr's
  one-chunk-at-a-time path structurally can't express.
- **Storage.** Ends up serving a handful of large range GETs instead of thousands
  of tiny ones, trading a little over-read for far fewer round-trips.

Bytes stream back in completion order, so fetch↔decode overlap is preserved. On
overhead-bound reads (e.g. Met Office HDF5 with thousands of tiny chunks) this is
commonly an ~8× speedup over a stock read, from one line.

## Quickstart

```python
import icechunk
from coalescing_zarr import open_zarr_coalesced

repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")

# A normal xarray Dataset — but reads go through the coalescing pipeline.
ds = open_zarr_coalesced(session)
region = ds["temperature"].sel(time="2024-01-01").compute()
```

`open_zarr_coalesced` returns an ordinary `xarray.Dataset` over the session's
own store (no wrapper) — everything downstream (`.sel`, `.compute`, plotting) is
unchanged. It defaults to eager reads (`chunks=None`) so each array read is one
bulk, range-coalesced request. Pass `chunks={"time": N}` to stay lazy/dask-backed
while still batching many chunks per task. Avoid the plain xarray default
(`chunks={}`) — that is one dask task per native chunk, which reintroduces the
per-chunk overhead and prevents coalescing.

### Why a dedicated opener?

`open_zarr_coalesced` is `xarray.open_zarr` with two things arranged for you: the
coalescing codec pipeline is active for the open, and **chunking is set so a read
spans many chunks**. That second part is the reason it exists — a plain
`xarray.open_zarr(session.store)` makes the array dask-backed at its native chunk
grid, i.e. one task per chunk, so the pipeline only ever sees a single chunk per
read and can't coalesce (and the per-chunk overhead this targets returns).
Coalescing across the native grid without an explicit coarse `chunks=` is a known
limitation and future work; for now, open through this function.

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

## Installation

This package needs **Icechunk's native `get_many_chunks`**, which is **not in a
released icechunk yet** — so you install a forked icechunk build alongside it.
(If it's missing at runtime, `open_zarr_coalesced` / `read_region` raise a
`NotImplementedError` that says so.)

**Into your own environment** — install the forked icechunk wheel first, then
this package from GitHub. Pre-built `cp312-abi3` wheels (Python 3.12+, no Rust)
are on the fork release for macOS (x86_64/arm64), Linux glibc + musl
(x86_64/arm64), and Windows. Pick your platform — e.g. Linux x86_64 (glibc):

```sh
# 1. forked icechunk (find your platform's wheel with:
#    gh release view fork-coalescing-wip --repo ianhi/icechunk)
pip install --force-reinstall --no-deps \
  https://github.com/ianhi/icechunk/releases/download/fork-coalescing-wip/icechunk-2.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 2. this package (not on PyPI — install from GitHub)
pip install "coalescing-zarr @ git+https://github.com/ianhi/coalescing-zarr"
```

**Working in this repo** — just `uv sync`. It installs the right fork wheel for
your platform automatically (pinned in `[tool.uv.sources]`); no Rust, nothing
else to do.

**No wheel for your platform** (e.g. musl Linux): build the fork from source
(needs a Rust toolchain), then install this package:

```sh
pip install "git+https://github.com/earth-mover/icechunk@ian/more-specific-vritual#subdirectory=icechunk-python"
pip install "coalescing-zarr @ git+https://github.com/ianhi/coalescing-zarr"
```

Why the separate icechunk install: PyPI forbids git/URL dependencies, so the
forked-icechunk requirement can't live in this package's metadata. It collapses
to a single `pip install` once the feature ships in a released icechunk.

**On Coiled** — icechunk is a compiled extension, so its **Linux x86_64** build
must come from a base software environment; package sync then layers your local
pure-Python code (coalescing-zarr and any edits) on top. Build the base once with
just the icechunk wheel:

```python
import coiled

coiled.create_software_environment(
    name="icechunk-coalescing",
    pip=["https://github.com/ianhi/icechunk/releases/download/fork-coalescing-wip/icechunk-2.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"],
)
```

Then launch with `--sync` against that base — package sync adds coalescing-zarr
and leaves icechunk alone (the base already has the matching 2.1.0), so your
local edits show up without rebuilding the env:

```sh
coiled notebook start --sync --software icechunk-coalescing --vm-type c7a.medium
# clusters: coiled.Cluster(software="icechunk-coalescing", ...) with package sync
```

Prefer not to manage a base env? Bake everything into one senv (add
`"coalescing-zarr @ git+https://github.com/ianhi/coalescing-zarr"` to the `pip`
list above) and launch with `--software icechunk-coalescing` and no `--sync` —
simpler, but you rebuild on every code change.

> **Don't use _bare_ package sync from a non-Linux machine.** icechunk's compiled
> `.so` is platform-specific, so sync alone ships your local (e.g. macOS arm64)
> build to the Linux worker and it fails to import (`invalid ELF header`).
> Providing the Linux icechunk via the base software environment above is what
> makes `--sync` safe. (Bare package sync *from* a matching Linux x86_64 machine
> is fine.)

## Develop

```sh
uv sync            # builds the forked icechunk automatically
uv run prek install
uv run pytest
```

See [`design.md`](./design.md) for why this needs both a bulk store method and a
custom codec pipeline, and how the streaming/decode-overlap works.
