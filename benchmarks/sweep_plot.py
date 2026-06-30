"""Initial sweep + dual-axis bar plot for coalescing vs stock reads.

Runs a small sweep through ``perf_harness`` and plots, per access pattern and
case:

  * left axis  — time to get the array ready, a STACKED bar of
                 (store-build `wrap_ms`) + (fetch/read `wall_ms`);
  * right axis — total bytes downloaded (MB).

Left-axis bars are blue (two shades for the stack), the right-axis bar is
orange, and each axis's labels/ticks are coloured to match its bars.

    uv run python benchmarks/sweep_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from coalescing_zarr import CoalescingManifestStore
from coalescing_zarr.config import PIPELINE_PATH
from perf_harness import STOCK_PIPELINE, Harness

PATTERNS = ["band", "subcube", "column", "timeseries"]
CASES = ["icechunk", "manifest", "coalesced", "coalesced-wide"]

# A large gap so the "wide" case bridges gaps other cases leave split — the
# point being to make the over-read (download) trade visible on gappy patterns.
WIDE_GAP = 8 * 1024 * 1024

# Colours: left (time) axis = blues, right (download) axis = orange.
C_FETCH = "#4c78a8"   # fetch/read time (wall_ms)
C_WRAP = "#1b3a5c"    # store-build time (wrap_ms), darker shade of the same hue
C_DOWNLOAD = "#e1812c"  # total download (MB)


def run_sweep() -> list:
    with Harness(mode_ms=45, bandwidth_mbs=200, concurrency=16) as h:
        return h.compare(
            cases=[
                ("icechunk", h.icechunk_store(), None, STOCK_PIPELINE),
                ("manifest", h.plain_store(), None, STOCK_PIPELINE),
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
            patterns=PATTERNS,
        )


def plot(results: list, out: Path) -> None:
    # Index results by (pattern, case) for decode_ms == 0.
    by = {(r.pattern, r.name): r for r in results if r.decode_ms == 0}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    x = range(len(CASES))
    width = 0.36

    for ax_t, pattern in zip(axes.flat, PATTERNS, strict=True):
        ax_d = ax_t.twinx()
        wrap = [by[(pattern, c)].wrap_ms for c in CASES]
        fetch = [by[(pattern, c)].wall_ms for c in CASES]
        mb = [by[(pattern, c)].mb for c in CASES]

        left = [i - width / 2 for i in x]
        right = [i + width / 2 for i in x]

        # Left axis: stacked time bar (store-build on top of fetch).
        ax_t.bar(left, fetch, width, color=C_FETCH, label="fetch/read")
        ax_t.bar(left, wrap, width, bottom=fetch, color=C_WRAP, label="store build")
        # Right axis: total download.
        ax_d.bar(right, mb, width, color=C_DOWNLOAD, label="download")

        ax_t.set_title(pattern, fontsize=11, fontweight="bold")
        ax_t.set_xticks(list(x))
        ax_t.set_xticklabels(
            [c.replace("coalesced", "coal.") for c in CASES],
            fontsize=8, rotation=15, ha="right",
        )
        ax_t.set_ylabel("time to ready (ms)", color=C_FETCH)
        ax_t.tick_params(axis="y", labelcolor=C_FETCH)
        ax_d.set_ylabel("download (MB)", color=C_DOWNLOAD)
        ax_d.tick_params(axis="y", labelcolor=C_DOWNLOAD)
        ax_t.set_axisbelow(True)
        ax_t.grid(axis="y", alpha=0.25)

    handles = [
        mpatches.Patch(color=C_FETCH, label="fetch / read (wall)"),
        mpatches.Patch(color=C_WRAP, label="store build (wrap)"),
        mpatches.Patch(color=C_DOWNLOAD, label="download (MB, right axis)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        "Coalescing vs stock: time-to-ready (stacked) and download, by pattern\n"
        "Harness mode=45ms latency, 200 MB/s, concurrency=16, decode=0",
        fontsize=11, y=1.10,
    )
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    results = run_sweep()
    plot(results, Path(__file__).parent / "sweep.png")
