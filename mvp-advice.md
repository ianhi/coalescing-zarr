# MVP advice & lessons learned (companion to `design.md`)

Notes for the agent building the coalescing MVP. Written after building a working
end-to-end prototype of exactly this idea in the sibling repo
`../virtual` (`ndpi-virtualizarr`): a coalescing codec pipeline, a
`get_many_chunks` store method, a custom dispatch pipeline, a snailmail-based
benchmark harness, plus a git-history dig into zarr's bulk-get APIs. **Reuse that
work — most of the algorithm already exists and is validated.** Paths below are
absolute so you can open them directly.

---

## 1. One load-bearing correction to `design.md`

> design.md §Framing: *"the only change it requires of Zarr Python is a
> `get_many_chunks` API (which already exists), called from within Zarr when
> indexing into a large portion of an array."*

**`get_many_chunks` does not exist, and no bulk API is called by the array read
path.** This was checked against the zarr `main` checkout at
`/Users/ian/Documents/dev/zarr-python`. The real state:

- zarr **v2** had `Store.getitems(keys)` and the read path called it (PR #606).
  The **v3 rewrite (PR #1584)** removed it. The new `BatchedCodecPipeline`
  fetches **one chunk per `getter.get()`**, fanned out with `concurrent_map`
  bounded by `async.concurrency` (default 10) — `codec_pipeline.py:384`.
- `Store.get_partial_values(prototype, key_ranges)` exists as an `@abstractmethod`
  (`abc/store.py:220`) — multi-key/multi-range — **but the read path never calls
  it.** It was added for the sharding spec and is otherwise vestigial. In
  VirtualiZarr's `ManifestStore` it is literally `raise NotImplementedError`
  (`virtualizarr/manifests/store.py:216`).
- `Store._get_many` (private, `abc/store.py:401`) and single-key `Store.get_ranges`
  (`abc/store.py:412`, sharding-only, PR #3925) exist; neither is on the array
  read path.

**Consequence for the MVP:** a store method *alone changes nothing* — zarr will
never call it. You **must** also ship a custom codec pipeline that calls your bulk
API and is registered in place of the default. The MVP plan's bullet about "a
custom codec pipeline, registered in place of the default" is therefore not an
alternative to the store method — it is **mandatory glue**. The "already exists"
framing is the one thing that will mislead an implementer; everything else in
`design.md` holds.

This is also the historically-missing piece: every bulk API zarr ever had
(`getitems`, `get_partial_values`, `_get_many`) either died or went unused
*because the codec pipeline was never taught to call it*. Your pipeline is that
teaching.

---

## 2. Reuse, don't reinvent — the algorithm is already built & tested

In `/Users/ian/Documents/dev/virtual/src/ndpi_virtualizarr/pipeline.py`:

- `resolve_chunk_entry(manifest_store, key, index)` — maps a zarr chunk key →
  `(store, path, offset, length)` by reading the manifest. This is your
  "derive the effective shard index" step, already written for VirtualiZarr.
- `plan_coalesced_spans(resolved, max_gap)` — group by file, sort by offset,
  merge neighbours within `max_gap`. **This is the whole coalescing core.**
  Properties already verified by tests: `max_gap=0` ⇒ zero over-read; large gap
  ⇒ ~1 span; pixels byte-identical across gaps.
- `CoalescingCodecPipeline.read()` — overrides `read` (the right hook: it receives
  the *entire* `batch_info` before zarr splits it into size-1 batches), resolves,
  merges, fetches spans concurrently under a semaphore, replays prefetched bytes
  through `super().read()` via a `CachedGetter` so decode is byte-exact.

In `/Users/ian/Documents/dev/virtual/examples/codec-pipeline/4_coalescing_on_ndpi.ipynb`:

- A `get_many_chunks` store method on a `ManifestStore` subclass + a dumb
  `StoreCoalescingPipeline` that dispatches to it. **This is your MVP's shape.**
  Two caveats baked into that prototype, fix them in the MVP:
  1. it returns a `dict[key, bytes]` (a barrier — see §3);
  2. it fetches spans **serially** (`for span: await …`). The production
     `CoalescingCodecPipeline` fans spans out with `asyncio.gather` + a semaphore
     bounded by `async.concurrency` — **do that**, so coalescing never reduces
     fetch concurrency below the per-chunk baseline.

Tests to lift: `/Users/ian/Documents/dev/virtual/tests/test_coalescing_pipeline.py`
(fidelity vs plain read, GET-count collapse, gap knob, zero-over-read at gap=0).

---

## 3. Elevate the "compute over time" open question — it is first-order

`design.md` lists this last under Open questions; it is actually the central
performance tension, and the measurements make it concrete.

**zarr's default read path already streams.** `batch_size=1` (`config.py:110`):
the selection is split into **one-chunk batches** run concurrently (10 in flight),
and decode is offloaded to a thread — `await asyncio.to_thread(codec.decode, …)`
(`numcodecs/_codecs.py:176`), which for imagecodecs releases the GIL and so
parallelizes across cores. So out of the box, **a chunk's decode starts the moment
its bytes land, overlapping the fetches still in flight.**

A coalescing pipeline that overrides `read()` and returns one big `dict`
**destroys that overlap**: it does all fetching, *then* all decoding — two serial
phases — and with one coalesced span the over-read sits on the critical path to the
*first* decode. Worst cases observed/derived:

1. **Lost fetch↔decode overlap.** Cores idle during the fetch phase; network idle
   during the decode phase. Cost grows with selection size × latency.
2. **One giant span gates first decode** (head-of-line); the 35× over-read of a
   scattered square (see §4) is transferred before any tile decodes.
3. **Slowest span stalls everything** — a `dict` return must await *all* spans, so
   one tail-latency GET blocks decode of tiles whose span finished long ago.

**Design implication — make the bulk API streaming.** `get_many_chunks` should
return an **async iterator yielding `(key, bytes)` in completion order**, and the
pipeline should launch a decode the instant each key arrives. Then span-1 tiles
decode while span-2 is still in flight. (This is exactly the completion-order
return d-v-b argued for on `get_ranges`.) The Rust/Icechunk version gets this
nearly for free: Icechunk already fetches concurrently internally; expose results
as they land.

### Measured, not hypothetical — the decode-cost crossover

`/Users/ian/Documents/dev/virtual/benchmarks/decode_cost_bench.py` injects a
**tunable per-tile decode cost** (a `time.sleep` *inside* the `to_thread` decode,
so it occupies a worker and overlaps I/O like real GIL-releasing decode) and runs
the `subcube` ROI over snailmail (lognormal latency + bandwidth pipe), reporting
total wall **and time-to-first-decode**. Representative numbers (subcube, bw 200 MB/s,
concurrency 16; `coalesced` = current barrier pipeline, `per_tile` = stock streaming):

| latency | decode/tile | per_tile wall | coalesced wall | per_tile 1st-decode | coalesced 1st-decode |
|---|---|---|---|---|---|
| **mode 0** (warm/low) | 0 ms | 115 | **37** | 6 | 19 |
| | 5 ms | **184** | 192 | 12 | 24 |
| | 20 ms | **652** | 660 | 29 | 39 |
| **mode 45 ms** (cloud) | 0 ms | 1685 | **73** | 25 | 52 |
| | 20 ms | 2219 | **777** | 56 | 146 |

(all ms). Three things to take to the MVP:

1. **Coalescing's win is real and large under latency** — at mode 45 ms it beats
   per-chunk by ~3–20× regardless of decode. This confirms the project's premise;
   the cloud case it targets is exactly where it shines.
2. **But the barrier *always* delays first-pixel** — `coalesced` starts decoding
   2–3× later than `per_tile` in every row (it can't decode until its fetch lands).
   That matters for interactive/progressive viewers even when total wall is fine.
3. **At low latency + nontrivial decode the barrier flips to a loss** — by
   decode ≳ 5 ms/tile at mode 0, `per_tile` (streaming) matches then beats
   `coalesced` on *total* wall, because streaming hides its fetch under decode while
   the barrier serializes fetch + over-read. The `pathological_column` ROI (20.8 MB
   coalesced vs 11.5 MB per-tile — heavy over-read) loses by more.

**Conclusion:** the barrier leaves measurable wall-clock on the table whenever
decode isn't free. A **streaming `get_many_chunks`** keeps coalescing's round-trip
win *and* recovers the overlap — it should be the MVP's target shape, not a later
optimization. Use this script (extend its `STRATEGIES` with a streaming-coalesced
variant) as the harness that proves it.

**`max_gap` is therefore two knobs, not one:** round-trips ↔ over-read **and**
round-trips ↔ pipelineability. Merging everything to one span maximizes the first
and kills the second. The sweet spot is "few spans," not "one span" — keep enough
spans in flight to keep cores fed. Your cost model (design §How it works, step 3)
should optimize wall-clock-to-last-decode, not GET count.

---

## 4. Validated numbers & ready-made best/worst-case ROIs

Real measurements on `CMU-1.tiff` (Hamamatsu NDPI), level 1, chunks `(8, 512, 3)`,
default `max_gap = 256 KiB`:

| ROI | tiles | normal GETs | gap=0 | gap=256 KiB | note |
|---|---|---|---|---|---|
| full-width band `[0:256, :]` | 800 | 800 | **1 GET, 1.00×** | 1 GET, 1.00× | **degenerate** — one contiguous run; collapses even at gap=0 |
| square subcube `[2000:3024, 2000:3024]` | 384 | 384 | 128 GETs, 1.00× | **1 GET, 35× over-read** | representative — 128 runs of 3 tiles, ~15 KB gaps |

Use these as your artificial-case anchors (design §Benchmarking asks for exactly
this). **Lesson: do not headline the full-width band** — it's a contiguous blob
that any reader collapses trivially and shows no over-read. The *square* is the
honest case where the merge-vs-split decision actually bites. (The benchmark's
`subcube`/`pathological_column`/`well_aligned` patterns already encode this.)

Also: **locally, slide reads are decode-bound and fast** — the ~40 s figure in
design.md is a cloud/latency artifact. Coalescing only wins when round-trips
dominate, so **benchmark under simulated latency, never locally.**

---

## 5. Benchmark harness already exists — reuse it

`/Users/ian/Documents/dev/virtual/benchmarks/` has a snailmail-based harness with:
local 127.0.0.1 HTTP range server with **log-normal per-request latency** and a
**shared-pipe bandwidth cap** (so over-read is charged), server-side GET/byte/
in-flight counters, and the three access patterns. This is the "robust, un-gameable
suite" design.md wants — extend it rather than rebuild. Snailmail is at 0.4.0
(per-request classify/breakdown; use `icechunk.s3_storage(...)`). Notes on the
ecosystem are in `/Users/ian/Documents/dev/virtual/notes/coalescing-ecosystem-state.md`.

Charge over-read against bandwidth in the cost model: at 35× over-read, on a
throttled pipe the wasted bytes are the bottleneck, not the round-trips.

---

## 6. Cost model & early-outs (design §Open questions)

The dangerous case is **overhead with no benefit** (time series: one chunk per
file). Concrete, cheap early-outs, in order:

1. `< 2` chunks → skip coalescing entirely.
2. Group resolved chunks by `(store, file path)`. If every chunk is in a distinct
   file → nothing to merge; return the per-chunk plan immediately. (This is the
   time-series fast path.)
3. Only within a file with ≥2 chunks do you sort + gap-merge.

Watch the per-request resolve cost: the Python prototype's `resolve_chunk_entry`
does a `urlparse` + prefix-strip **per key** — fine for a prototype, but it is the
overhead design.md worries about. In Rust over the in-memory manifest it's cheap;
in the Python MVP, precompute/caching the path resolution per array amortizes it.
Measure "resolve+plan time" separately from fetch time so you can prove the
"as-fast-or-only-slightly-slower in the worst case" goal.

**API shape recommendation** (mirrors `get_ranges`/`_get_many` so it survives the
maintainers' "non-standard interface" objection): an **optional** method with a
default that just fans out over `get`, explicit `max_gap` / `max_coalesced_bytes`
knobs, and defined **partial-failure semantics** (`get_ranges` uses a
`BaseExceptionGroup` and cancels pending fetches on first failure — copy that).

---

## 7. Scope notes

- **Single array only for the MVP.** A codec pipeline only ever sees one array's
  selection, so the cross-variable GOES optimization (design §Limits) **cannot**
  live at the pipeline layer at all — it must be store-/engine-level. That's a
  strong second argument for the Icechunk-native end state, but explicitly
  out of scope for the wrapping-store MVP. Don't try to force it in.
- **Don't double-coalesce.** zarr `main` already coalesces *single-key* inner-shard
  reads via `get_ranges`, and obstore has its own per-file range grouping. The
  virtual *cross-key* case is the gap you fill. Make sure only one coalescing layer
  is active on a given read (the `../virtual` repo keeps a legacy registry-level
  `CoalescingStore` and the pipeline mutually exclusive for this reason).
- **VirtualiZarr path == Icechunk path.** As design.md notes, a `ManifestStore`
  built from an Icechunk store (via the VirtualiZarr Icechunk parser) is analogous
  to Icechunk itself and uses obstore to fetch — so prototyping `get_many_chunks`
  on a `ManifestStore` subclass is a faithful stand-in for the eventual Rust
  implementation. Start there.

---

## 8. Lessons learned (process)

- **Reads are decode-bound locally; coalescing is a round-trip optimization.**
  Always benchmark under latency/bandwidth simulation. A local "it's 1 GET now!"
  proves correctness, not speed.
- **Pick a representative ROI early.** A contiguous full-width band collapses to 1
  GET trivially and hides the over-read tradeoff; the square subcube exposes it.
  Validate the ROI's chunk-offset layout (gaps histogram) before drawing
  conclusions — I burned a cycle headlining the degenerate band.
- **Verify the API exists before designing around it.** The single biggest risk in
  `design.md` was the assumption that zarr calls a bulk get; it doesn't. Read
  `codec_pipeline.py:read/read_batch` and `abc/store.py` first.
- **Fewer GETs is not the objective; wall-clock is.** Over-read and lost fetch/
  decode overlap can make "1 GET" slower than "128 GETs" on the wrong pipe. The
  benchmark must measure time-to-last-decode under realistic latency×bandwidth, and
  the headline metric should be that, not the GET count.

---

## 9. Suggested MVP build order

1. Subclass `ManifestStore`; implement `get_many_chunks(keys)` reusing
   `resolve_chunk_entry` + `plan_coalesced_spans`. Fan spans out concurrently
   (semaphore = `async.concurrency`). **Return an async iterator in completion
   order**, not a dict.
2. Write a custom `CodecPipeline` overriding `read()` that dispatches to
   `get_many_chunks` when the store supports it (else `super().read()`), and
   **decodes each key as it arrives** (kick off `to_thread` decode on arrival).
   Register it via `codec_pipeline.path`.
3. Correctness gate: byte-identical to the plain read on band, square, column, and
   a time-series-like ROI (port `tests/test_coalescing_pipeline.py`).
4. Stand up the snailmail bench; measure **time-to-last-decode** and
   **time-to-first-decode** for {plain, coalesced} × {gap sweep} × {latency,
   bandwidth} on the square + a real GOES query. Add the no-benefit time-series
   case to prove the early-outs cost ~nothing.
5. Only then consider the Rust/Icechunk-native port; the Python numbers tell you
   whether the algorithm earns it.
