"""Unit tests for the pure coalescing planner."""

from __future__ import annotations

from coalescing_zarr.planning import ResolvedChunk, plan_spans


class _FakeStore:
    """Stand-in object store; only its identity matters for grouping."""

    def __init__(self, name: str) -> None:
        self.name = name


def _chunk(
    store: _FakeStore, path: str, offset: int, length: int, key: str = "k"
) -> ResolvedChunk:
    return ResolvedChunk(key=key, store=store, path=path, offset=offset, length=length)


def test_empty() -> None:
    assert plan_spans([]) == []


def test_single_chunk_is_one_span() -> None:
    s = _FakeStore("a")
    spans = plan_spans([_chunk(s, "f", 0, 10)])
    assert len(spans) == 1
    assert spans[0].over_read == 0


def test_adjacent_chunks_merge_at_gap_zero() -> None:
    s = _FakeStore("a")
    chunks = [_chunk(s, "f", 0, 10), _chunk(s, "f", 10, 10), _chunk(s, "f", 20, 10)]
    spans = plan_spans(chunks, max_gap=0)
    assert len(spans) == 1
    assert spans[0].start == 0
    assert spans[0].end == 30
    assert spans[0].over_read == 0


def test_gap_zero_does_not_merge_across_gaps() -> None:
    s = _FakeStore("a")
    # 5-byte gap between each 10-byte chunk.
    chunks = [_chunk(s, "f", 0, 10), _chunk(s, "f", 15, 10), _chunk(s, "f", 30, 10)]
    spans = plan_spans(chunks, max_gap=0)
    assert len(spans) == 3
    assert sum(sp.over_read for sp in spans) == 0


def test_large_gap_merges_with_over_read() -> None:
    s = _FakeStore("a")
    chunks = [_chunk(s, "f", 0, 10), _chunk(s, "f", 15, 10), _chunk(s, "f", 30, 10)]
    spans = plan_spans(chunks, max_gap=16)
    assert len(spans) == 1
    # Bytes 0..40 fetched, 30 useful -> 10 bytes over-read (the two 5-byte gaps).
    assert spans[0].nbytes == 40
    assert spans[0].over_read == 10


def test_distinct_files_never_merge() -> None:
    s = _FakeStore("a")
    chunks = [_chunk(s, "f0", 0, 10), _chunk(s, "f1", 0, 10), _chunk(s, "f2", 0, 10)]
    spans = plan_spans(chunks, max_gap=1_000_000)
    assert len(spans) == 3
    assert all(sp.over_read == 0 for sp in spans)


def test_distinct_stores_never_merge() -> None:
    a, b = _FakeStore("a"), _FakeStore("b")
    chunks = [_chunk(a, "f", 0, 10), _chunk(b, "f", 10, 10)]
    spans = plan_spans(chunks, max_gap=1_000_000)
    assert len(spans) == 2


def test_max_coalesced_bytes_caps_span() -> None:
    s = _FakeStore("a")
    chunks = [_chunk(s, "f", i * 10, 10) for i in range(10)]  # contiguous 0..100
    spans = plan_spans(chunks, max_gap=0, max_coalesced_bytes=35)
    # No span may exceed 35 bytes; every chunk still placed exactly once.
    assert all(sp.nbytes <= 35 for sp in spans)
    assert sum(len(sp.members) for sp in spans) == 10


def test_every_chunk_appears_exactly_once() -> None:
    s = _FakeStore("a")
    chunks = [_chunk(s, "f", i * 12, 10, key=f"k{i}") for i in range(5)]
    spans = plan_spans(chunks, max_gap=4)
    seen = [m.key for sp in spans for m in sp.members]
    assert sorted(seen) == [f"k{i}" for i in range(5)]
