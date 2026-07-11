"""Integration tests: the coalescing read path through real zarr.

The correctness gate is that a coalesced read is **byte-identical** to a plain
read, across layouts. The remaining tests assert the coalescing *behaviour* the
design promises — collapse of adjacent runs, the gap knob trading round-trips
for over-read, and the time-series early-out doing no merging — using the
store's own over-read/span counters.
"""

from __future__ import annotations

import numpy as np
import zarr
from conftest import SyntheticStore

import coalescing_zarr
import coalescing_zarr.manifest_store
from coalescing_zarr.config import PIPELINE_PATH


def _read_coalesced(
    synth: SyntheticStore,
    *,
    max_gap: int,
    max_coalesced_bytes: int | None = None,
) -> np.ndarray:
    """Read the whole array through the coalescing pipeline with given knobs."""
    synth.store.max_gap = max_gap
    synth.store.max_coalesced_bytes = max_coalesced_bytes
    synth.store.stats.reset()
    with zarr.config.set({"codec_pipeline.path": PIPELINE_PATH}):
        arr = zarr.open_group(synth.store, mode="r")[synth.array_name]
        return arr[:]


def _plain(synth: SyntheticStore) -> np.ndarray:
    # Default pipeline -> per-chunk store.get, no coalescing.
    synth.store.stats.reset()
    with zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    ):
        arr = zarr.open_group(synth.store, mode="r")[synth.array_name]
        return arr[:]


def test_import_surface() -> None:
    assert coalescing_zarr.open_zarr_coalesced is not None
    assert coalescing_zarr.manifest_store.CoalescingManifestStore is not None


def test_byte_identical_adjacent(adjacent_store: SyntheticStore) -> None:
    plain = _plain(adjacent_store)
    got = _read_coalesced(adjacent_store, max_gap=256 * 1024)
    np.testing.assert_array_equal(got, adjacent_store.expected)
    np.testing.assert_array_equal(got, plain)


def test_byte_identical_gapped(gapped_store: SyntheticStore) -> None:
    plain = _plain(gapped_store)
    got = _read_coalesced(gapped_store, max_gap=256 * 1024)
    np.testing.assert_array_equal(got, plain)


def test_byte_identical_time_series(time_series_store: SyntheticStore) -> None:
    plain = _plain(time_series_store)
    got = _read_coalesced(time_series_store, max_gap=256 * 1024)
    np.testing.assert_array_equal(got, plain)


def test_adjacent_collapses_to_one_span(adjacent_store: SyntheticStore) -> None:
    _read_coalesced(adjacent_store, max_gap=0)
    # Contiguous run -> single coalesced request even with zero gap tolerance.
    assert adjacent_store.store.stats.spans == 1
    assert adjacent_store.store.stats.over_read_bytes == 0


def test_gap_zero_zero_over_read(gapped_store: SyntheticStore) -> None:
    _read_coalesced(gapped_store, max_gap=0)
    # 4 chunks separated by gaps -> 4 spans, nothing wasted.
    assert gapped_store.store.stats.spans == 4
    assert gapped_store.store.stats.over_read_bytes == 0


def test_large_gap_trades_round_trips_for_over_read(
    gapped_store: SyntheticStore,
) -> None:
    _read_coalesced(gapped_store, max_gap=4096)
    # The 1 KiB gaps are now bridged: one request, but bytes are over-read.
    assert gapped_store.store.stats.spans == 1
    assert gapped_store.store.stats.over_read_bytes == 3 * 1024


def test_time_series_no_merge(time_series_store: SyntheticStore) -> None:
    _read_coalesced(time_series_store, max_gap=256 * 1024)
    # One chunk per file: no merging possible, no over-read (the no-benefit case).
    assert time_series_store.store.stats.spans == 4
    assert time_series_store.store.stats.over_read_bytes == 0


def test_inlined_chunk_not_treated_as_missing(inlined_store: SyntheticStore) -> None:
    # An inlined chunk is real data; coalescing must not turn it into the fill
    # value. Coalesced read must equal both ground truth and the plain read.
    plain = _plain(inlined_store)
    got = _read_coalesced(inlined_store, max_gap=256 * 1024)
    np.testing.assert_array_equal(got, inlined_store.expected)
    np.testing.assert_array_equal(got, plain)
