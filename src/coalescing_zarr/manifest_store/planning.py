"""The coalescing algorithm — a pure, I/O-free planner.

This module is deliberately isolated from any store or network code so it can be
unit-tested in isolation and swapped for a smarter cost model later. The only
job here is: given a set of chunks already resolved to byte ranges in backing
files, decide which ranges to fetch together.

The current strategy is the simplest thing that works: group by file, sort by
offset, and merge neighbours whose gap is within ``max_gap`` (subject to a
``max_coalesced_bytes`` cap). It does *not* yet weigh round-trips against
over-read against pipelineability — that is the future cost model. Keeping
``plan_spans`` pure is what makes replacing it cheap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obstore.store import ObjectStore


@dataclass(frozen=True)
class ResolvedChunk:
    """A single requested chunk, resolved to a byte range in a backing file.

    This is the output of "derive the effective shard index" — produced by the
    store from the manifest, before any bytes are fetched.
    """

    key: str
    """The Zarr chunk key (e.g. ``"band/c/0/3/0"``), used to map bytes back."""
    store: ObjectStore
    """The object store holding the backing file."""
    path: str
    """Path of the backing file *within* ``store``."""
    offset: int
    """Byte offset of this chunk within the file."""
    length: int
    """Byte length of this chunk."""

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass
class Span:
    """A contiguous byte range to fetch in one request, covering >=1 chunk.

    ``members`` are the chunks served by this span, in offset order. ``start`` /
    ``end`` (exclusive) bound the bytes actually fetched; the difference between
    their total length and the summed chunk lengths is the over-read.
    """

    store: ObjectStore
    path: str
    start: int
    end: int
    members: list[ResolvedChunk] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def useful_bytes(self) -> int:
        return sum(m.length for m in self.members)

    @property
    def over_read(self) -> int:
        return self.nbytes - self.useful_bytes


def _group_key(chunk: ResolvedChunk) -> tuple[int, str]:
    # Same backing file == same (store instance, path). ``id`` is sufficient
    # because the registry hands back cached store instances per prefix.
    return (id(chunk.store), chunk.path)


def plan_spans(
    resolved: list[ResolvedChunk],
    *,
    max_gap: int = 0,
    max_coalesced_bytes: int | None = None,
) -> list[Span]:
    """Group chunks into coalesced byte-range spans.

    Parameters
    ----------
    resolved
        Chunks resolved to ``(store, path, offset, length)``.
    max_gap
        Maximum number of *unwanted* bytes tolerated between two chunks before
        they are merged into one request. ``0`` merges only strictly adjacent
        chunks (zero over-read); larger values trade over-read for fewer
        round-trips.
    max_coalesced_bytes
        Optional hard cap on a single span's size, so one pathological run
        cannot produce an enormous request. ``None`` means no cap.

    Returns
    -------
    A list of spans. Each chunk appears in exactly one span. Chunks in distinct
    files never merge.

    Notes
    -----
    Cheap early-outs fall out of the structure for free: with ``< 2`` chunks, or
    when every chunk lives in a distinct file, every group has size 1 and we
    skip the sort/merge entirely. This is the time-series "no benefit" fast
    path — it costs only the grouping pass, no sorting.
    """
    if not resolved:
        return []

    groups: dict[tuple[int, str], list[ResolvedChunk]] = defaultdict(list)
    for chunk in resolved:
        groups[_group_key(chunk)].append(chunk)

    spans: list[Span] = []
    for group in groups.values():
        if len(group) == 1:
            # Singleton file: no merge possible, no sort needed.
            c = group[0]
            spans.append(Span(c.store, c.path, c.offset, c.end, [c]))
            continue

        group.sort(key=lambda c: c.offset)
        current: Span | None = None
        for c in group:
            if current is None:
                current = Span(c.store, c.path, c.offset, c.end, [c])
                continue
            gap = c.offset - current.end
            new_end = max(current.end, c.end)
            new_size = new_end - current.start
            within_gap = gap <= max_gap
            within_cap = max_coalesced_bytes is None or new_size <= max_coalesced_bytes
            if within_gap and within_cap:
                current.members.append(c)
                current.end = new_end
            else:
                spans.append(current)
                current = Span(c.store, c.path, c.offset, c.end, [c])
        assert current is not None
        spans.append(current)

    return spans
