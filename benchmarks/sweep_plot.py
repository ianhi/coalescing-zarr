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
    "coalesced-wide",
]
# Short x-axis labels (full names are long for a 5-case row).
CASE_LABELS = {
    "manifest": "manifest",
    "icechunk": "ic\n(default)",
    "icechunk-bigreq": "ic\n(12MB+)",
    "coalesced": "coal.",
    "coalesced-wide": "coal.\nwide",
}

# A large gap so the "wide" case bridges gaps other cases leave split — the
# point being to make the over-read (download) trade visible on gappy patterns.
WIDE_GAP = 64 * 1024 * 1024
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
C_FETCH = "#4c78a8"  # fetch/read time (wall_ms)
C_WRAP = "#1b3a5c"  # store-build time (wrap_ms), darker shade of the same hue
C_DOWNLOAD = "#e1812c"  # total download (MB)


def run_sweep(source_name: str) -> tuple[list, list[str], str]:
    source = NdpiSource(jpeg=False) if source_name == "ndpi" else SyntheticSource()
    with Harness(source=source, mode_ms=45, bandwidth_mbs=200, concurrency=16) as h:
        patterns = list(h.patterns)
        results = h.compare(
            cases=[
                ("manifest", h.plain_store(), None, STOCK_PIPELINE),
                ("icechunk", h.icechunk_store(), None, STOCK_PIPELINE),
                (
                    "icechunk-bigreq",
                    tuned_icechunk_store(h, BIGREQ),
                    None,
                    STOCK_PIPELINE,
                ),
                (
                    "coalesced",
                    h.plain_store(),
                    h.wrap_with(CoalescingManifestStore),
                    PIPELINE_PATH,
                ),
                (
                    "coalesced-wide",
                    h.plain_store(),
                    lambda _base: CoalescingManifestStore(
                        h.group, registry=h.registry, max_gap=WIDE_GAP
                    ),
                    PIPELINE_PATH,
                ),
            ],
            decode_ms=(0,),
            patterns=patterns,
        )
    to_json(results, out_json := Path(__file__).parent / f"sweep_{source_name}.json")
    print(f"wrote {out_json}")
    return results, patterns, type(source).__name__


def plot(results: list, patterns: list[str], source_label: str, out: Path) -> None:
    by = {(r.pattern, r.name): r for r in results if r.decode_ms == 0}

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
        wrap = [by[(pattern, c)].wrap_ms for c in CASES]
        fetch = [by[(pattern, c)].wall_ms for c in CASES]
        mb = [by[(pattern, c)].mb for c in CASES]

        left = [i - width / 2 for i in x]
        right = [i + width / 2 for i in x]

        # Time spans ~100x (coalesced vs stock) -> log so small bars stay
        # visible. Download is a narrow range (a few to tens of MB) and reads
        # better linear, where over-read magnitude is obvious.
        ax_t.set_yscale("log")

        # Left axis: time-to-ready, stacked store-build (wrap) on top of
        # fetch/read (wall). wrap is ~0 today (store construction is O(1)), so
        # the wrap segment is an invisible sliver here, but the breakdown stays
        # for when manifest-scan / build cost grows (e.g. the Rust port).
        ax_t.bar(left, fetch, width, color=C_FETCH, label="fetch/read")
        ax_t.bar(left, wrap, width, bottom=fetch, color=C_WRAP, label="store build")
        # Right axis: total download.
        ax_d.bar(right, mb, width, color=C_DOWNLOAD, label="download")

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
        mpatches.Patch(color=C_FETCH, label="fetch / read (wall, left log)"),
        mpatches.Patch(color=C_WRAP, label="store build (wrap; ~0 here)"),
        mpatches.Patch(color=C_DOWNLOAD, label="download (MB, right linear)"),
    ]
    # Legend along the bottom so it never collides with the title. Short
    # (1-row) figures need it pushed further down to clear the x-tick labels.
    legend_y = -0.04 if nrows >= 2 else -0.18
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, legend_y),
    )
    fig.suptitle(
        f"Coalescing vs stock ({source_label}): "
        "time-to-ready (stacked, log) + download (linear)\n"
        "Harness mode=45ms latency, 200 MB/s, concurrency=16, decode=0",
        fontsize=11,
    )
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    results, patterns, label = run_sweep(name)
    plot(results, patterns, label, Path(__file__).parent / f"sweep_{name}.png")
