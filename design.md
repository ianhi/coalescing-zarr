# Design

## The problem

Reading a region of a virtual Zarr array fetches **one chunk per request**. That
is egress-optimal but latency-bound: when many small chunks live in the same
backing file at nearby offsets (NDPI slides, Met Office HDF5, GOES bands), the
per-request round-trips dominate the wall time. Sometimes it is faster to
download a whole file than to fetch its chunks individually.

Coalescing trades a little over-read for far fewer round-trips: fetch nearby
chunks in one larger range request. The caller chooses where on that trade-off
to sit (`max_gap`), rather than being locked into the egress optimum — which
matters less on, e.g., AWS Open Data where egress is free anyway.

## Two pieces, both required

zarr's v3 read path never calls a bulk-get on the store — `BatchedCodecPipeline`
fetches one chunk per `getter.get()`. So a bulk store method alone changes
nothing; zarr will never call it. Coalescing therefore needs two pieces:

1. **A bulk store method** — `get_many_chunks(keys)` receives the whole set of
   requested chunks at once, so it can plan coalesced byte-range spans and fetch
   them together.
2. **A codec pipeline that calls it** — `CoalescingCodecPipeline` overrides
   `read()`, the hook that sees the *entire* batch before zarr splits it into
   size-1 batches, and dispatches to `get_many_chunks`. This is mandatory glue,
   not an optional optimization.

Decode is **not** reimplemented. Each fetched chunk is replayed through zarr's
stock `read_batch` via a tiny `CachedGetter`, so the decode/assembly path
(codec chain, slicing, fill values) is byte-for-byte identical to a normal read.
We only change *how* bytes are fetched and *when* each decode starts.

## Streaming is the point

The bulk method yields `(key, bytes)` in **completion order**, and the pipeline
kicks off each chunk's decode the instant its bytes land. This preserves the
fetch↔decode overlap that zarr's default path already has. A naive coalescing
pipeline that returns one `dict[key, bytes]` destroys it — it fetches
everything, then decodes everything, two serial phases — and lets the slowest
span stall decode of chunks whose bytes arrived long ago.

So `max_gap` is really **two knobs at once**: round-trips ↔ over-read, and
round-trips ↔ pipelineability. Merging everything into one span minimizes
round-trips but kills the overlap. The sweet spot is "few spans," not "one
span." The metric that matters is **wall-clock to last decode under latency ×
bandwidth**, not GET count.

## Seams built to evolve

- **`plan_spans`** (`manifest_store/planning.py`) is a pure, I/O-free function:
  `resolved_chunks + knobs -> list[Span]`. Group by file, sort by offset,
  gap-merge. Trivially unit-testable and swappable for a smarter cost model.
  Cheap early-outs fall out of the structure for free — with <2 chunks, or one
  chunk per file (the time-series worst case), there is nothing to sort or merge.
- **Fetch / decode** are separate stages (`manifest_store/fetch.py`,
  `pipeline.py`) so the
  overlap strategy can change without touching the planner.

## Two store backends behind one pipeline

The pipeline supports two ways to satisfy a bulk read, chosen by store type. The
seam is deliberately in the pipeline, not in a store the caller must construct —
so a plain `session.store` reads through coalescing with no wrapper.

- **Native Icechunk (the headline).** Icechunk's *native* `get_many_chunks`
  resolves + coalesces + fetches through Icechunk's own client, across virtual
  and native chunks. `icechunk_native.py` is the thin translation (chunk keys ⇄
  `(array_path, coords)`, native `(index, bytes)` ⇄ `(key, buffer)`); the
  pipeline detects a native store and calls it directly. `open_zarr_coalesced()`
  arranges the pipeline + chunking for an xarray `Dataset`; `read_region()` does
  a single cross-array bulk read.
- **`CoalescingManifestStore`** is the pure-Python reference path: it resolves
  keys against a VirtualiZarr `ManifestStore` and fetches spans through obstore,
  exposing a `(keys, prototype) → (key, buffer)` `get_many_chunks`. Useful for
  testing the planner and pipeline against byte-exact synthetic data without a
  custom icechunk. It is the model the native path stands in for.

## Scope and direction

The MVP coalesces *within a single array read* (a codec pipeline only ever sees
one array's selection). Cross-variable coalescing — bands interleaved in one
file, fetched once instead of per-variable — is left on the table; `read_region`
is a first step toward the eventual "get many chunks for many arrays" query,
where an engine declares everything it wants and the store solves retrieval. The
long-term home for this logic is Icechunk itself, in Rust, over the in-memory
manifest; the Python store here is the faithful stand-in for that port.
