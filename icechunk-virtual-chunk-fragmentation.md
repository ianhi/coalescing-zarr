# Investigate: HTTP virtual-chunk reads fragment under `RepositoryConfig.default()`

A handoff for an icechunk agent. Discovered while benchmarking store-level range
coalescing for virtual Zarr stores (the `coalescing` project).

> **Status: reproduced standalone** (not just inside the benchmark harness).
> A 16-chunk array of 32 KiB chunks reads in **80 GETs** under
> `RepositoryConfig.default()` (5 range requests per chunk) vs **16 GETs** (one
> per chunk) when `ideal_concurrent_request_size` is set explicitly to its own
> documented 12 MB default. Repro at the bottom; run it directly.

## Summary

When reading **virtual** chunks from an HTTP virtual-chunk container, a single
chunk read is split into several smaller HTTP range GETs. With a 32 KiB chunk,
each chunk becomes **~5 range requests for the same total bytes**. Explicitly
setting

```python
config.storage = icechunk.StorageSettings(
    concurrency=icechunk.StorageConcurrencySettings(
        ideal_concurrent_request_size=12 * 1024 * 1024,  # the documented default
    )
)
```

collapses it to **exactly one GET per chunk**. So `RepositoryConfig.default()`
— which leaves `config.storage` unset (`None`) — does **not** behave like the
documented 12 MB default for virtual-chunk-container reads.

This matters: extra round-trips dominate wall-clock under latency, so the
fragmentation makes virtual reads through Icechunk look much slower than they
need to. Much of an apparent "Icechunk is slow on virtual reads" gap turned out
to be this config artifact, not anything inherent.

## Environment

- icechunk 2.1.0
- virtualizarr 2.7.0, zarr 3.2.1, obstore 0.10.0
- snailmail 0.4.1 (the measurement server, see below)
- Python 3.12+

## Observation

32 KiB chunks (array `(128, 1024)` int32, chunk `(8, 1024)` → 16 chunks),
reading the whole array, counting server-side GETs:

| config | GETs (16 chunks) | bytes |
|---|---|---|
| `RepositoryConfig.default()` (storage concurrency unset) | **80** (5×/chunk) | 524288 (unchanged) |
| explicit `ideal_concurrent_request_size = 12 MB` (the documented default) | **16** (1×/chunk) | 524288 |
| explicit `= 64 MiB` | 16 | 524288 |

(Verified standalone with the repro below, icechunk 2.1.0.)

Same total bytes in every row — only the **request count** differs.

The documented defaults (from icechunk's type stubs,
`StorageConcurrencySettings`):

- `ideal_concurrent_request_size` = **12,582,912 (12 MB)**
- `max_concurrent_requests_for_object` = **18**

Since 32 KiB ≪ 12 MB, nothing should split under the documented default — and
indeed it doesn't when set explicitly. Only the **unset** path fragments.

> Note: this only bites when a chunk is larger than the effective request size.
> For tiny chunks (e.g. NDPI JPEG tiles, ~0.5–7 KiB) the unset and 12 MB cases
> both give one GET per tile, so the issue is invisible there.

## Questions to answer

1. **Is this a bug?** When `config.storage` / `StorageConcurrencySettings` is
   unset, are the documented defaults (`ideal_concurrent_request_size = 12 MB`,
   `max_concurrent_requests_for_object = 18`) applied to the **virtual chunk
   container's** object store, or only to the repo's primary storage? It looks
   like virtual-chunk-container reads fall back to a different (much smaller)
   effective request size when unset.
2. Is the fragmentation-when-unset **intended** (a strategy to parallelize large
   reads), or an oversight where defaults aren't propagated to
   `VirtualChunkContainer` stores?
3. If unintended, what's the cheapest fix — propagate `StorageConcurrencySettings`
   defaults to virtual-chunk-container stores in `RepositoryConfig.default()`?

## About snailmail (the measurement tool)

A tiny local benchmarking server: an HTTP range server (or S3 object store) that
injects log-normal latency + a bandwidth cap and exposes server-side request
counters. It is purely the GET counter here; nothing else depends on it.

- Install: `pip install snailmail` (v0.4.1).
- API used below:
  - `snailmail.HTTPRangeServer.from_file(path, latency=snailmail.Fixed(0.0), bandwidth_mbs=None).start()`
  - `.url(name)` → URL for a served file
  - `.reset_counts()`
  - `.stats()` → `{"n_gets", "total_bytes", "max_in_flight"}`
  - `.stop()`

## Minimal reproduction

```python
# pip install icechunk virtualizarr zarr numpy snailmail obstore
import tempfile, numpy as np, zarr, icechunk, snailmail
from pathlib import Path
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore
from virtualizarr.manifests import (
    ChunkManifest, ManifestArray, ManifestGroup, ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata

tmp = Path(tempfile.mkdtemp())
CHUNK = (8, 1024)           # 8*1024*4 = 32768 B (32 KiB) per chunk, int32
GRID = (16, 1)              # 16 chunks
shape = (GRID[0] * CHUNK[0], GRID[1] * CHUNK[1])

# 1. backing blob + virtual manifest (contiguous chunks)
data = np.arange(shape[0] * shape[1], dtype="<i4").reshape(shape)
offsets = np.zeros(GRID, np.uint64); lengths = np.zeros(GRID, np.uint64)
blob = bytearray()
for r in range(GRID[0]):
    b = data[r * CHUNK[0]:(r + 1) * CHUNK[0], :].astype("<i4").tobytes()
    offsets[r, 0] = len(blob); lengths[r, 0] = len(b); blob += b
(tmp / "blob.bin").write_bytes(bytes(blob))

# 2. serve over snailmail (no latency; we only count GETs)
srv = snailmail.HTTPRangeServer.from_file(
    str(tmp / "blob.bin"), latency=snailmail.Fixed(0.0), bandwidth_mbs=None
).start()
blob_url = srv.url("blob.bin"); base = blob_url.rsplit("/", 1)[0] + "/"

# 3. virtualizarr manifest -> icechunk repo whose chunks point at the served blob
paths = np.full(GRID, blob_url, dtype=np.dtypes.StringDType())
manifest = ChunkManifest.from_arrays(
    paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
)
meta = create_v3_array_metadata(
    shape=shape, data_type=np.dtype("<i4"), chunk_shape=CHUNK,
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    dimension_names=("y", "x"),
)
group = ManifestGroup(arrays={"data": ManifestArray(metadata=meta, chunkmanifest=manifest)})
registry = ObjectStoreRegistry()
registry.register(base, HTTPStore.from_url(base, client_options={"allow_http": True}))


def make_repo(repo_dir, ideal):
    cfg = icechunk.RepositoryConfig.default()
    cfg.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(url_prefix=base, store=icechunk.http_store())
    )
    if ideal is not None:
        cfg.storage = icechunk.StorageSettings(
            concurrency=icechunk.StorageConcurrencySettings(
                ideal_concurrent_request_size=ideal
            )
        )
    storage = icechunk.Storage.new_local_filesystem(str(repo_dir))
    auth = {base: icechunk.Credentials.HttpAccess()}
    if repo_dir.exists() and any(repo_dir.iterdir()):
        return icechunk.Repository.open(storage=storage, config=cfg, authorize_virtual_chunk_access=auth)
    repo = icechunk.Repository.create(storage=storage, config=cfg, authorize_virtual_chunk_access=auth)
    s = repo.writable_session("main")
    ManifestStore(group=group, registry=registry).to_virtual_dataset(
        loadable_variables=[]
    ).vz.to_icechunk(s.store)
    s.commit("virtual")
    return repo


def count(repo, label):
    srv.reset_counts()
    out = np.asarray(zarr.open_group(repo.readonly_session("main").store, mode="r")["data"][:])
    assert np.array_equal(out, data)
    st = srv.stats()
    print(f"{label:38} GETs={st['n_gets']:4d}  bytes={st['total_bytes']}")


repo_dir = tmp / "repo"
count(make_repo(repo_dir, None),         "RepositoryConfig.default() (unset)")          # observed 80
count(make_repo(repo_dir, 12 * 1024 * 1024), "explicit ideal=12 MB (documented default)")  # observed 16
count(make_repo(repo_dir, 64 * 1024 * 1024), "explicit ideal=64 MiB")                       # observed 16
srv.stop()
```

**Expected if there were no bug:** all three rows equal (16 GETs), since
32 KiB ≪ 12 MB so nothing should split.
**Observed (verified standalone):** the unset row fragments to 80 (5×/chunk);
the explicit rows are 16 (1×/chunk).

Please confirm standalone, then determine whether unset config should inherit
the documented 12 MB default for virtual-chunk-container object stores.
