# coalescing-zarr

Store-level range coalescing for virtual Zarr stores.

When reading a region of a virtual array, many small chunks often live in the
same backing file at nearby byte offsets. zarr's read path fetches one chunk per
request, which is egress-optimal but latency-bound. This package coalesces those
fetches into a few larger range requests, trading some over-read for far fewer
round-trips — while preserving zarr's fetch↔decode overlap by streaming bytes
back in completion order and decoding each chunk as it arrives.

See [`design.md`](./design.md) for the full design.

## Components

- `CoalescingManifestStore` — wraps a VirtualiZarr `ManifestStore` (e.g. built
  from an Icechunk repo) and adds a bulk, streaming `get_many_chunks`.
- `CoalescingCodecPipeline` — a zarr `CodecPipeline` that calls `get_many_chunks`
  and decodes each chunk on arrival. Registered via the `codec_pipeline.path`
  config (zarr never calls a bulk get on its own, so this glue is required).
- `plan_spans` — the pure coalescing algorithm: group by file, gap-merge by
  offset. Isolated so the cost model can evolve independently.

## Develop

```sh
uv sync
uv run prek install
uv run pytest
```
