"""Initial sweep + dual-axis bar plot for coalescing vs stock reads.

Runs a small sweep through ``perf_harness`` and plots, per access pattern and
case:

  * left axis  — time to get the array ready, a STACKED bar of
                 (store-build `wrap_ms`) + (fetch/read `wall_ms`);
  * right axis — total bytes downloaded (MB).

Left-axis bars are blue (two shades for the stack), the right-axis bar is
orange, and each axis's labels/ticks are coloured to match its bars.

    uv run python benchmarks/sweep_plot.py            # synthetic source
    uv run python benchmarks/sweep_plot.py ndpi       # real NDPI tile geometry
"""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path

import icechunk
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from coalescing_zarr import CoalescingManifestStore
from coalescing_zarr.config import PIPELINE_PATH
from perf_harness import (
    STOCK_PIPELINE,
    Harness,
    NdpiSource,
    SyntheticSource,
    to_json,
)

CASES = [
    "manifest",
    "icechunk",
    "icechunk-bigreq",
    "coalesced",
    "coalesced-cap",
    "coalesced-wide",
]
# Short x-axis labels (full names are long for a 6-case row).
CASE_LABELS = {
    "manifest": "manifest",
    "icechunk": "ic\n(default)",
    "icechunk-bigreq": "ic\n(12MB+)",
    "coalesced": "coal.\n(16KB)",
    "coalesced-cap": "coal.\nwide\ncap1M",
    "coalesced-wide": "coal.\nwide",
}

# Realistic small gap: NDPI tiles within a row are contiguous (gap 0) but
# consecutive rows are ~41-56 KB apart. 16 KB merges the contiguous within-row
# runs with zero over-read, without chaining whole rows into one giant span.
COAL_GAP = 16 * 1024
# A large gap that bridges everything -> a single GET with maximal over-read,
# kept to show the over-merge extreme on the download axis.
WIDE_GAP = 64 * 1024 * 1024
# Bound a single coalesced request's size. Pairs with WIDE_GAP to show that
# capping the span breaks the one giant GET into several bounded ones (parallel,
# streamable) without changing total over-read.
COAL_CAP = 1024 * 1024
# icechunk's per-object request-split target. Default fragments a large chunk
# read into ~ideal-sized sub-requests; setting it well above the chunk size
# gives one GET per chunk (the "bigreq" case).
BIGREQ = 256 * 1024 * 1024


def tuned_icechunk_store(h: Harness, ideal_request_size: int):
    """Reopen the harness's Icechunk repo with a custom request-split size.

    Reuses the repo the harness already wrote (``h.icechunk_store()``), just
    opening a read store with a different ``ideal_concurrent_request_size`` so we
    can compare icechunk concurrency settings without rewriting the repo.
    """
    h.icechunk_store()  # ensure the repo has been written once
    repo_dir = Path(h.icechunk_dir) if h.icechunk_dir else h._dir / "repo"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            url_prefix=h._base_url, store=icechunk.http_store()
        )
    )
    config.storage = icechunk.StorageSettings(
        concurrency=icechunk.StorageConcurrencySettings(
            ideal_concurrent_request_size=ideal_request_size,
            max_concurrent_requests_for_object=16,
        )
    )
    storage = icechunk.Storage.new_local_filesystem(str(repo_dir))
    auth = {h._base_url: icechunk.Credentials.HttpAccess()}
    repo = icechunk.Repository.open(
        storage=storage, config=config, authorize_virtual_chunk_access=auth
    )
    return repo.readonly_session("main").store


# Colours: left (time) axis = blues, right (download) axis = orange.
C_FETCH = "#4c78a8"  # time-to-ready (wall_ms)
C_DOWNLOAD = "#e1812c"  # total download (MB)

# Realistic per-chunk decode cost (ms), injected by the harness's decode wrapper
# so reads aren't treated as zero-cost-to-decode. Tune to taste.
DECODE_MS = 1.0


def _fmt_time(ms: float) -> str:
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def _fmt_mb(mb: float) -> str:
    return f"{mb:.0f}MB" if mb >= 10 else f"{mb:.1f}MB"


def _coal(h: Harness, **kw: object):
    """A store_wrapper building a CoalescingManifestStore with given knobs."""
    return lambda _base: CoalescingManifestStore(h.group, registry=h.registry, **kw)


def _cases(h: Harness) -> list:
    """The shared case list: stock baselines + coalescing variants."""
    return [
        ("manifest", h.plain_store(), None, STOCK_PIPELINE),
        ("icechunk", h.icechunk_store(), None, STOCK_PIPELINE),
        ("icechunk-bigreq", tuned_icechunk_store(h, BIGREQ), None, STOCK_PIPELINE),
        ("coalesced", h.plain_store(), _coal(h, max_gap=COAL_GAP), PIPELINE_PATH),
        (
            "coalesced-cap",
            h.plain_store(),
            _coal(h, max_gap=WIDE_GAP, max_coalesced_bytes=COAL_CAP),
            PIPELINE_PATH,
        ),
        ("coalesced-wide", h.plain_store(), _coal(h, max_gap=WIDE_GAP), PIPELINE_PATH),
    ]


def run_sweep(source_name: str) -> tuple[list, list[str], str]:
    source = NdpiSource(jpeg=False) if source_name == "ndpi" else SyntheticSource()
    with Harness(source=source, mode_ms=45, bandwidth_mbs=200, concurrency=16) as h:
        patterns = list(h.patterns)
        results = h.compare(
            cases=_cases(h),
            decode_ms=(DECODE_MS,),
            patterns=patterns,
        )
    to_json(results, out_json := Path(__file__).parent / f"sweep_{source_name}.json")
    print(f"wrote {out_json}")
    return results, patterns, type(source).__name__


def plot(results: list, patterns: list[str], source_label: str, out: Path) -> None:
    # One decode level per run, so index directly by (pattern, case).
    by = {(r.pattern, r.name): r for r in results}

    ncols = 2 if len(patterns) >= 4 else len(patterns)
    nrows = ceil(len(patterns) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 3.75 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    flat = axes.flat
    x = range(len(CASES))
    width = 0.36

    for ax_t, pattern in zip(flat, patterns, strict=False):
        ax_d = ax_t.twinx()
        # time to ready = store build (wrap, ~0) + fetch/read (wall).
        ready = [by[(pattern, c)].wrap_ms + by[(pattern, c)].wall_ms for c in CASES]
        mb = [by[(pattern, c)].mb for c in CASES]

        left = [i - width / 2 for i in x]
        right = [i + width / 2 for i in x]

        # Time spans ~100x (coalesced vs stock) -> log so small bars stay
        # visible. Download is a narrow range (a few to tens of MB) and reads
        # better linear, where over-read magnitude is obvious.
        ax_t.set_yscale("log")

        b_time = ax_t.bar(left, ready, width, color=C_FETCH)
        b_dl = ax_d.bar(right, mb, width, color=C_DOWNLOAD)

        # Headroom so the bar-top labels don't clip.
        ax_t.set_ylim(min(ready) * 0.45, max(ready) * 3.5)
        ax_d.set_ylim(0, max(mb) * 1.22)
        ax_t.bar_label(
            b_time, labels=[_fmt_time(v) for v in ready],
            padding=2, fontsize=6.5, color=C_FETCH,
        )
        ax_d.bar_label(
            b_dl, labels=[_fmt_mb(v) for v in mb],
            padding=2, fontsize=6.5, color=C_DOWNLOAD,
        )

        ax_t.set_title(pattern, fontsize=11, fontweight="bold")
        ax_t.set_xticks(list(x))
        ax_t.set_xticklabels(
            [CASE_LABELS[c] for c in CASES],
            fontsize=8,
        )
        ax_t.set_ylabel("time to ready (ms, log)", color=C_FETCH)
        ax_t.tick_params(axis="y", labelcolor=C_FETCH)
        ax_d.set_ylabel("download (MB)", color=C_DOWNLOAD)
        ax_d.tick_params(axis="y", labelcolor=C_DOWNLOAD)
        ax_t.set_axisbelow(True)
        ax_t.grid(axis="y", alpha=0.25)

    # Hide any unused axes (e.g. 3 patterns in a 2x2 grid).
    for ax in list(flat)[len(patterns) :]:
        ax.set_visible(False)

    handles = [
        mpatches.Patch(color=C_FETCH, label="time to ready (ms, left log)"),
        mpatches.Patch(color=C_DOWNLOAD, label="download (MB, right linear)"),
    ]
    # Legend along the bottom so it never collides with the title. Short
    # (1-row) figures need it pushed further down to clear the x-tick labels.
    legend_y = -0.04 if nrows >= 2 else -0.18
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, legend_y),
    )
    fig.suptitle(
        f"Coalescing vs stock ({source_label}): "
        "time-to-ready (log) + download (linear)\n"
        "Harness mode=45ms latency, 200 MB/s, concurrency=16, "
        f"decode={DECODE_MS}ms/chunk",
        fontsize=11,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


DECODE_SWEEP = (0.0, 1.0, 5.0, 20.0)
DECODE_PATTERN = "subcube"  # the representative ROI


def run_decode_sweep(source_name: str) -> tuple[list, str]:
    """Sweep per-chunk decode cost for one pattern, all cases.

    Shows the streaming-vs-barrier tension: a single-GET coalesced read
    (``coalesced-wide``) cannot overlap decode with fetch (all bytes land at
    once), so its wall climbs with decode cost; streaming readers (stock,
    small-gap / capped coalescing) overlap decode with in-flight fetches.
    """
    source = NdpiSource(jpeg=False) if source_name == "ndpi" else SyntheticSource()
    with Harness(source=source, mode_ms=45, bandwidth_mbs=200, concurrency=16) as h:
        results = h.compare(
            cases=_cases(h), decode_ms=DECODE_SWEEP, patterns=[DECODE_PATTERN]
        )
    out = Path(__file__).parent / f"decode_{source_name}.json"
    to_json(results, out)
    print(f"wrote {out}")
    return results, type(source).__name__


def plot_decode(results: list, source_label: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for case in CASES:
        rows = sorted(
            (r for r in results if r.name == case), key=lambda r: r.decode_ms
        )
        ax.plot(
            [r.decode_ms for r in rows],
            [r.wall_ms for r in rows],
            marker="o",
            label=CASE_LABELS[case].replace("\n", " "),
        )
    ax.set_yscale("log")
    ax.set_xlabel("per-chunk decode cost (ms)")
    ax.set_ylabel("time to ready (ms, log)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="case")
    fig.suptitle(
        f"Decode-cost sweep ({source_label}, {DECODE_PATTERN}): "
        "streaming vs single-GET barrier\n"
        "Harness mode=45ms latency, 200 MB/s, concurrency=16",
        fontsize=11,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    mode = sys.argv[2] if len(sys.argv) > 2 else "patterns"
    base = Path(__file__).parent
    if mode == "decode":
        results, label = run_decode_sweep(name)
        plot_decode(results, label, base / f"decode_{name}.png")
    else:
        results, patterns, label = run_sweep(name)
        plot(results, patterns, label, base / f"sweep_{name}.png")
