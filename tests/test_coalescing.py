"""Integration tests: the coalescing read path through real zarr.

The correctness gate is that a coalesced read is **byte-identical** to a plain
read, across layouts. The remaining tests assert the coalescing *behaviour* the
design promises — collapse of adjacent runs, the gap knob trading round-trips
for over-read, and the time-series early-out doing no merging — using the
store's own over-read/span counters.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import zarr
from conftest import SyntheticStore

import coalescing_zarr
from coalescing_zarr.config import PIPELINE_PATH, settings


@contextmanager
def coalescing(max_gap: int, max_coalesced_bytes: int | None = None) -> Iterator[None]:
    """Activate the coalescing pipeline with given knobs, restoring afterwards."""
    prev_gap = settings.max_gap
    prev_cap = settings.max_coalesced_bytes
    settings.max_gap = max_gap
    settings.max_coalesced_bytes = max_coalesced_bytes
    try:
        with zarr.config.set({"codec_pipeline.path": PIPELINE_PATH}):
            yield
    finally:
        settings.max_gap = prev_gap
        settings.max_coalesced_bytes = prev_cap


def _read(synth: SyntheticStore) -> np.ndarray:
    synth.store.stats.reset()
    arr = zarr.open_group(synth.store, mode="r")[synth.array_name]
    return arr[:]


def _plain(synth: SyntheticStore) -> np.ndarray:
    # Default pipeline -> per-chunk store.get, no coalescing.
    with zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    ):
        return _read(synth)


def test_import_surface() -> None:
    assert coalescing_zarr.CoalescingManifestStore is not None


def test_byte_identical_adjacent(adjacent_store: SyntheticStore) -> None:
    plain = _plain(adjacent_store)
    with coalescing(max_gap=256 * 1024):
        got = _read(adjacent_store)
    np.testing.assert_array_equal(got, adjacent_store.expected)
    np.testing.assert_array_equal(got, plain)


def test_byte_identical_gapped(gapped_store: SyntheticStore) -> None:
    plain = _plain(gapped_store)
    with coalescing(max_gap=256 * 1024):
        got = _read(gapped_store)
    np.testing.assert_array_equal(got, plain)


def test_byte_identical_time_series(time_series_store: SyntheticStore) -> None:
    plain = _plain(time_series_store)
    with coalescing(max_gap=256 * 1024):
        got = _read(time_series_store)
    np.testing.assert_array_equal(got, plain)


def test_adjacent_collapses_to_one_span(adjacent_store: SyntheticStore) -> None:
    with coalescing(max_gap=0):
        _read(adjacent_store)
    # Contiguous run -> single coalesced request even with zero gap tolerance.
    assert adjacent_store.store.stats.spans == 1
    assert adjacent_store.store.stats.over_read_bytes == 0


def test_gap_zero_zero_over_read(gapped_store: SyntheticStore) -> None:
    with coalescing(max_gap=0):
        _read(gapped_store)
    # 4 chunks separated by gaps -> 4 spans, nothing wasted.
    assert gapped_store.store.stats.spans == 4
    assert gapped_store.store.stats.over_read_bytes == 0


def test_large_gap_trades_round_trips_for_over_read(
    gapped_store: SyntheticStore,
) -> None:
    with coalescing(max_gap=4096):
        _read(gapped_store)
    # The 1 KiB gaps are now bridged: one request, but bytes are over-read.
    assert gapped_store.store.stats.spans == 1
    assert gapped_store.store.stats.over_read_bytes == 3 * 1024


def test_time_series_no_merge(time_series_store: SyntheticStore) -> None:
    with coalescing(max_gap=256 * 1024):
        _read(time_series_store)
    # One chunk per file: no merging possible, no over-read (the no-benefit case).
    assert time_series_store.store.stats.spans == 4
    assert time_series_store.store.stats.over_read_bytes == 0
