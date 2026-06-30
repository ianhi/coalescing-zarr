# Coalescing for virtual stores

## Motivation

When querying virtual stores with Icechunk, we are forced into the egress-cost
optimum: each chunk is fetched as its own request. This is not always the
desired trade-off. By coalescing requests at the store level, we can trade
higher egress costs for lower latency. The user should be able to choose where
on this trade-off they sit, rather than being locked into the egress-optimal
behavior (which matters less, for example, on AWS Open Data where egress is
free).

Current performance bottlenecks motivating this work:

- **NDPI virtualization is unusable.** Reading a small ROI takes ~40 seconds —
  slower than downloading the entire file and reading it locally. NDPI is the
  most extreme example of the small-chunk problem, and also the simplest to fix.
- **GOESS map queries.** A bigger, more impactful expansion of the same problem,
  with real users today.
- **TIFF-file interactions with object stores are pathological.** The usual
  workflow downloads a TIFF and opens it with `tifffile`, which calls `seek`
  constantly. Each `seek` resets the object store cache, so its built-in
  (~1 MB) buffer never helps; the same bytes get fetched repeatedly and often
  go unused. This is worse than virtualizing the file, since at least
  Icechunk's cache retains fetched chunks.

The same conclusion — that it is sometimes faster to download a whole file than
to fetch individual virtual chunks — has been reached independently (e.g. UK
Met Office data on Microsoft Planetary Computer, where very small chunks were
the cause).

## Framing: where does sharding live?

Today, dealing with sharding is a codec-level operation that happens in Python,
and this is assumed to be a good idea. We would prefer sharding to be handled by
the store, with the store exposing a bulk API. A bulk "get many chunks at once"
call gives the store the chance to recognize a fast path for fulfilling a
request for many similar chunks.

The shard index has to live somewhere. There are essentially three options:

1. **In the data itself** — what Zarr does today. Inflexible.
2. **The store derives it** from information it already has, such as the
   manifest. This is what we are proposing.
3. **Promote it to the format / data model** — e.g. Icechunk serializes a shard
   index into the manifest format. This would make sharding a first-class
   citizen but requires changing the Zarr spec; it is the longest route.

Option 2 is the most feasible in the short term, but it requires two pieces, not
one:

1. A **bulk store method** (`get_many_chunks`) that receives the whole set of
   requested chunks and can coalesce them.
2. A **custom codec pipeline that actually calls it.**

> **Correction (verified against zarr `main`):** there is no bulk get on the v3
> array read path, and `get_many_chunks` does not exist. zarr v2 had
> `Store.getitems(keys)` and the read path called it (PR #606); the v3 rewrite
> (PR #1584) removed it. The v3 `BatchedCodecPipeline` fetches **one chunk per
> `getter.get()`**, fanned out with `concurrent_map` bounded by
> `async.concurrency` (default 10). `Store.get_partial_values`, `Store._get_many`,
> and single-key `Store.get_ranges` exist but are sharding-only and **never
> called on the array read path** (in VirtualiZarr's `ManifestStore`,
> `get_partial_values` is `raise NotImplementedError`).
>
> Consequence: **a store method alone changes nothing — zarr will never call
> it.** The custom codec pipeline is therefore not an alternative to the store
> method; it is **mandatory glue.** Historically every bulk API zarr had
> (`getitems`, `get_partial_values`, `_get_many`) died or went unused precisely
> because the codec pipeline was never taught to call it. Our pipeline is that
> teaching.

## How it works

When Icechunk receives a "get all these chunks" task, it knows the chunks are
virtual. It can then:

1. Scan the manifest and observe that most of the requested chunks live in the
   same file / object.
2. Observe that many of those chunks are adjacent or near each other (within a
   configurable tolerance).
3. Consult a cost model to decide whether it is cheaper to fetch them in one
   large request rather than many small ones.
4. Only then issue the requests.

This is only possible because the responsibility for coalescing is pushed into
the store implementation — which is what is missing today.

To do step 3, Icechunk must derive an **effective shard index** from the
manifest. This should be cheap, because the manifest is already loaded into
memory and the work happens in Rust — but it must be benchmarked (see Open
questions).

Step 4 must **stream**: results are yielded in completion order as each
coalesced span lands, so the pipeline can start decoding while later spans are
still in flight (see §Batched decode and pipelining).

## Batched decode and pipelining

This is the central performance tension, not a footnote.

zarr's **default** read path already overlaps fetch and decode. The selection is
split into one-chunk batches (`batch_size=1`), up to `async.concurrency` run at
once, and each chunk's decode is offloaded to a thread
(`await asyncio.to_thread(codec.decode, …)`) — which for imagecodecs releases
the GIL and parallelizes across cores. So a chunk's decode starts the moment its
bytes land, overlapping the fetches still in flight.

A naive coalescing pipeline that overrides `read()` and returns one big
`dict[key, bytes]` **destroys that overlap**: it fetches everything, *then*
decodes everything — two serial phases. Three failure modes follow:

1. **Lost fetch↔decode overlap.** Cores idle during the fetch phase; the network
   idles during the decode phase. Cost grows with selection size × latency.
2. **One giant span gates the first decode** (head-of-line blocking): the entire
   over-read is transferred before any chunk decodes.
3. **Slowest span stalls everything.** A `dict` return must await all spans, so a
   single tail-latency GET blocks decode of chunks whose span finished long ago.

**Design implication:** the bulk API yields `(key, bytes)` in **completion
order**, and the pipeline kicks off a `to_thread` decode the instant each key
arrives. Span-1 chunks then decode while span-2 is still in flight. The Icechunk
end state gets this nearly for free — it already fetches concurrently
internally; it just needs to expose results as they land.

This reframes `max_gap` as **two knobs, not one**: round-trips ↔ over-read
**and** round-trips ↔ pipelineability. Merging everything into one span
maximizes the first and kills the second. The sweet spot is "few spans," not
"one span" — keep enough spans in flight to keep cores fed. **The cost model
optimizes wall-clock-to-last-decode, not GET count.**

## Decisions

- **Store-level range coalescing.** Range coalescing lives in the store behind a
  bulk `get_many_chunks` method, rather than modifying the Zarr spec or relying
  on codec-level handling.
- **Custom codec pipeline is mandatory.** Since zarr's read path never calls a
  bulk get, the MVP must ship a codec pipeline that calls `get_many_chunks` and
  is registered in place of the default.
- **MVP as a wrapping store prototype.** Begin with a wrapping store in Python
  to test the logic before migrating to Rust. This avoids forking Zarr Python.
- **Bulk API is streaming, completion-order.** `get_many_chunks` returns an
  async iterator yielding `(key, bytes)` as fetches land, so the pipeline can
  decode each chunk the moment it arrives (see §How it works and §Batched
  decode).
- **Single array only for the MVP.** A codec pipeline only ever sees one array's
  selection, so the cross-variable optimization (§Limits) cannot live here — it
  is explicitly out of scope.
- **Primary benchmark use cases: NDPI and GOESS map queries.**

## Data sources

- **NDPI** is available locally in `../virtual`.
- **GOESS** is available via Arraylake.
- For the initial MVP we work with NDPI, plus our artificial best-case /
  worst-case examples.

## MVP plan

This is a **fresh implementation.** A prior end-to-end prototype exists in
`../virtual` (see §Prior art) and is worth reading for lessons, but we are not
lifting its code — we want a clean design built to be improved (the coalescing
algorithm and the batched-decode handling are the two pieces most likely to
evolve, so both are isolated behind seams from the start).

Two components:

1. **`CoalescingManifestStore`** — wraps a base `ManifestStore` built from an
   Icechunk store via the VirtualiZarr Icechunk parser (the array chunk refs
   method). The base store already has full manifest information and fetches via
   obstore, making it analogous to the eventual Icechunk-native path. The
   wrapper adds `get_many_chunks(keys) -> AsyncIterator[(key, bytes)]`:
   - resolve each key → `(file, offset, length)` from the manifest;
   - **plan** coalesced spans (the algorithm — see seams below);
   - fetch spans concurrently (semaphore bounded by `async.concurrency`, so
     coalescing never drops below the per-chunk fetch concurrency);
   - slice each span back into per-key bytes and **yield them in completion
     order**.
2. **`CoalescingCodecPipeline`** — a `CodecPipeline` that overrides `read()`
   (the hook that sees the *entire* `batch_info` before zarr splits it into
   size-1 batches), dispatches to `get_many_chunks` when the store supports it
   (else falls back to `super().read()`), and **decodes each key as it arrives**
   via `to_thread`. Registered via the `codec_pipeline.path` config.

**Seams to design carefully (these will be tuned/replaced):**

- **Coalescing algorithm.** Keep planning a pure function:
  `plan(resolved_chunks, knobs) -> list[Span]`, with knobs `max_gap` /
  `max_coalesced_bytes`. No I/O inside; trivially unit-testable and swappable for
  a smarter cost model later. Cheap early-outs live here (see below).
- **Decode scheduling.** Keep "what to fetch" (planning), "fetch" (span I/O), and
  "decode" (per-key, on arrival) as three separable stages so the
  fetch↔decode overlap strategy can change without touching the algorithm.

**Cost-model early-outs** (cheap, ordered — the time-series no-benefit path must
cost ~nothing):

1. `< 2` chunks → skip coalescing.
2. Group resolved chunks by `(store, file)`; if every chunk is in a distinct
   file, return the per-chunk plan immediately (time-series fast path).
3. Only within a file with ≥2 chunks do we sort by offset and gap-merge.

Measure resolve+plan time separately from fetch time, to prove the
"as-fast-or-only-slightly-slower in the worst case" goal. (The prototype's
per-key `urlparse`+prefix-strip is exactly the overhead to watch; precompute /
cache path resolution per array.)

Doing the prototype in Python is useful pressure: if the algorithm is faster in
all cases in Python, it will certainly be faster in Rust. In the worst case, the
Python algorithm runs as slow as it can but must still give a benefit — the goal
is to be as fast or only slightly slower in the worst case while getting large
wins elsewhere.

## Benchmarking

- Construct **artificial best-case and worst-case virtual stores**, not just
  real data, so we have well-defined test cases that show where the algorithm
  falls on the spectrum. Also include real data.
- Define a benchmark suite robust enough that it cannot be gamed — the only way
  to make the numbers go down is by actually making the algorithm better.
- Use **Snail Mail** for controlled network simulation. Unlike a single fixed
  latency store, Snail Mail:
  - models latency as a distribution (log-normal), capturing the long tail of
    slow gets where the real cost lives;
  - is an HTTP server / local S3 object store rather than a Zarr store, so you
    can virtualize against it;
  - can cap bandwidth (e.g. simulate a 10 Mbit/s connection), giving a second
    axis — "is downloading the whole file still faster on bad Wi-Fi?";
  - lets you run Icechunk over it too, so manifest-fetch latency is simulated
    and the full system can be benchmarked under perfectly controlled conditions
    that cloud testing cannot provide.
- Use Max's chunk-layout visualization tool to inspect layout within files
  (e.g. GOESS, which turned out to be much less orderly than expected).

**Headline metric: time-to-last-decode (and time-to-first-decode) under
simulated latency × bandwidth — not GET count.** Fewer GETs is not the
objective; over-read and lost fetch↔decode overlap can make "1 GET" slower than
"128 GETs" on a throttled pipe. Charge over-read against the bandwidth cap so the
cost model feels it.

**Pick honest ROIs.** Measured on `CMU-1.tiff` (NDPI), level 1, chunks
`(8, 512, 3)`, `max_gap = 256 KiB`:

| ROI | tiles | normal GETs | gap=0 | gap=256 KiB |
|---|---|---|---|---|
| full-width band `[0:256, :]` | 800 | 800 | 1 GET, 1.00× | 1 GET, 1.00× |
| square subcube `[2000:3024, 2000:3024]` | 384 | 384 | 128 GETs, 1.00× | 1 GET, **35× over-read** |

The full-width band is **degenerate** — one contiguous run that collapses to a
single GET even at `gap=0`, with no over-read; do not headline it. The **square
subcube** is the honest case where the merge-vs-split decision actually bites.
Include a time-series-like ROI (one chunk per file) to prove the early-outs cost
~nothing. Validate an ROI's chunk-offset gap histogram before drawing
conclusions. Reads are decode-bound locally, so **never benchmark locally** —
the ~40 s NDPI figure is a latency artifact.

## Limits and future directions

### Cross-variable optimization is left on the table

Zarr stores different variables in separate files, so a single
`get_many_chunks` call for one array cannot optimize across variables that
happen to share a file. The GOESS RGB map query fetches three bands from the same
file but treats them as completely separate, fetching the same file three times.
In the worst case the bands are interleaved as a single unit, so a fetch pulls
in data for variables it then discards. We can rescue some of these
optimizations but not all of them. Caching layers (e.g. obstore) do not fix this
out of the box — obspec deliberately leaves caching separate.

### Tighter compute/storage coupling

As compute engines like Zacks get more powerful, the Zarr API becomes more
limiting; full power is only reached when storage and compute understand each
other directly (potentially bypassing the Zarr API). The ideal long-term shape
is a **"get many chunks for many arrays"** query: an engine declares all the
chunks it will want across all variables, and the Icechunk coalescing engine
solves the retrieval strategy. A well-written coalescing engine would then
immediately handle cases like the GOESS three-band query optimally.

## Prior art (for lessons, not for lifting)

A working end-to-end prototype of this idea lives in `../virtual`
(`ndpi-virtualizarr`): a coalescing codec pipeline, a `get_many_chunks` store
method, a snailmail benchmark harness, and tests. We build fresh, but the
lessons it validated shape this design:

- `plan_coalesced_spans` (group by file → sort by offset → merge within
  `max_gap`) is the coalescing core; its properties are test-verified
  (`max_gap=0` ⇒ zero over-read; large gap ⇒ ~1 span; bytes identical across
  gaps). Our `plan()` seam mirrors this.
- The prototype's two known flaws to **avoid**: it returns a `dict` (a barrier —
  kills pipelining, §Batched decode) and fetches spans **serially**. We stream
  in completion order and fan spans out concurrently.
- The snailmail harness (log-normal per-request latency, shared-pipe bandwidth
  cap, server-side GET/byte/in-flight counters, `subcube` /
  `pathological_column` / `well_aligned` patterns) is the shape of our bench.
- **Don't double-coalesce.** zarr `main` already coalesces *single-key*
  inner-shard reads via `get_ranges`, and obstore does per-file range grouping.
  Our gap is the *cross-key* virtual case — ensure only one coalescing layer is
  active on a given read.

## Open questions

### Cost of the coalescing logic

- What is the cost of scanning the manifest to identify groupable chunks?
- What is the cost of grouping requests at the store level?
- Crucially, deriving the effective shard index from the manifest adds overhead
  to every request. The worst case is overhead with no benefit (e.g. the GOESS
  time series query): the code must get to "I cannot combine these" as fast as
  possible, and every branch matters because it can add latency for nothing.
  There are fast early-outs — e.g. noticing that every requested chunk is in a
  different manifest — though some such heuristics may be wrong (different
  manifests could in theory cover the same files).

### Impact on workloads where coalescing does not help

- Could coalescing harm other cases?
- Time series is the worst case: one chunk per file, so there is nothing to
  coalesce.
- Map tiles may be a better case, depending on how chunks are laid out in the
  HDF5 file.
- In the worst case, a single GOESS band could be spread widely across the file,
  so grouping yields little benefit.

(The bytes-to-array-compute-over-time trade-off, previously listed here, is now
treated as first-order in §Batched decode and pipelining.)
