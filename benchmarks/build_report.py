"""Build a self-contained ``report.html`` from the committed sweep results.

Reads the ``sweep_*.json`` / ``decode_*.json`` produced by ``sweep_plot.py`` and
the ``*.png`` plots, and assembles a single local HTML page (prose + tables +
embedded plots) you can open in a browser. Regenerate after re-running sweeps:

    uv run python benchmarks/build_report.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).parent
CASE_ORDER = [
    "manifest",
    "icechunk",
    "icechunk-bigreq",
    "coalesced",
    "coalesced-cap",
    "coalesced-wide",
]
CASE_DESC = {
    "manifest": "stock VirtualiZarr ManifestStore, default pipeline (1 GET/chunk)",
    "icechunk": "persisted Icechunk repo, RepositoryConfig.default()",
    "icechunk-bigreq": "Icechunk with ideal_concurrent_request_size raised (no split)",
    "coalesced": "our coalescing, max_gap=16 KB (merge contiguous runs)",
    "coalesced-cap": "max_gap=wide + 1 MiB span cap (bounded parallel GETs)",
    "coalesced-wide": "max_gap=64 MiB (merge everything -> one GET)",
}


def _img(path: Path) -> str:
    if not path.exists():
        return f"<p><em>missing: {path.name}</em></p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{path.stem}" src="data:image/png;base64,{b64}">'


def _load(name: str) -> list[dict] | None:
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def _fmt_time(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def pattern_table(results: list[dict]) -> str:
    """One table per pattern: case x (GETs, MB, over-read, time, first-decode)."""
    patterns = list(dict.fromkeys(r["pattern"] for r in results))
    out = []
    for pat in patterns:
        rows = {r["name"]: r for r in results if r["pattern"] == pat}
        useful = rows["manifest"]["mb"]  # manifest reads exactly the useful bytes
        out.append(f"<h4>{pat}</h4><table><thead><tr>"
                   "<th>case</th><th>GETs</th><th>download</th><th>over-read</th>"
                   "<th>time to ready</th><th>1st decode</th></tr></thead><tbody>")
        for c in CASE_ORDER:
            if c not in rows:
                continue
            r = rows[c]
            over = r["mb"] / useful if useful else 1.0
            cls = ' class="win"' if c.startswith("coal") and r["wall_ms"] < rows[
                "manifest"]["wall_ms"] else ""
            out.append(
                f"<tr{cls}><td>{c}</td><td>{r['gets']:,}</td>"
                f"<td>{r['mb']:.2f} MB</td><td>{over:.1f}×</td>"
                f"<td>{_fmt_time(r['wall_ms'])}</td>"
                f"<td>{_fmt_time(r['first_ms'])}</td></tr>"
            )
        out.append("</tbody></table>")
    return "\n".join(out)


def decode_pivot(results: list[dict]) -> str:
    """Pivot: case x decode_ms -> time to ready."""
    decodes = sorted({r["decode_ms"] for r in results})
    by = {(r["name"], r["decode_ms"]): r for r in results}
    head = "".join(f"<th>{d:g} ms</th>" for d in decodes)
    out = [f"<table><thead><tr><th>case</th>{head}</tr></thead><tbody>"]
    for c in CASE_ORDER:
        cells = "".join(
            f"<td>{_fmt_time(by[(c, d)]['wall_ms'])}</td>"
            for d in decodes if (c, d) in by
        )
        if cells:
            out.append(f"<tr><td>{c}</td>{cells}</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


CSS = """
body{font:16px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;
margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.9rem}
h2{margin-top:2.4rem;border-bottom:2px solid #eee;padding-bottom:.3rem}
h4{margin:.8rem 0 .3rem;color:#444}
img{max-width:100%;border:1px solid #eee;border-radius:6px;margin:.6rem 0}
table{border-collapse:collapse;margin:.4rem 0 1rem;font-size:.86rem;width:100%}
th,td{border:1px solid #e3e3e3;padding:.3rem .55rem;text-align:right}
th:first-child,td:first-child{text-align:left;font-family:ui-monospace,monospace}
thead th{background:#f6f8fa} tr.win td{background:#eef7ee}
code{background:#f3f3f3;padding:.1rem .3rem;border-radius:4px;font-size:.9em}
.cases dt{font-family:ui-monospace,monospace;font-weight:600;margin-top:.4rem}
.note{background:#fff8e6;border-left:4px solid #e1b000;padding:.6rem 1rem;
border-radius:4px}
"""


def build() -> Path:
    syn = _load("sweep_synthetic.json")
    ndpi = _load("sweep_ndpi.json")
    dsyn = _load("decode_synthetic.json")
    dndpi = _load("decode_ndpi.json")

    cases_dl = "".join(
        f"<dt>{c}</dt><dd>{CASE_DESC[c]}</dd>" for c in CASE_ORDER
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Coalescing for virtual Zarr stores — benchmark report</title>
<style>{CSS}</style></head><body>

<h1>Coalescing for virtual Zarr stores</h1>
<p>Benchmark report for store-level range coalescing on virtual Zarr stores.
Reading a region of a virtual array touches many small chunks that often live in
the same backing file at nearby byte offsets. zarr fetches one chunk per request
(egress-optimal, latency-bound). <strong>Coalescing</strong> merges those into a
few larger range requests — trading some over-read for far fewer round-trips —
while streaming bytes back in completion order so decode overlaps fetch.</p>

<h2>Method</h2>
<p>The <code>perf_harness</code> rig serves a backing blob over
<strong>snailmail</strong> (a local HTTP range server with log-normal latency and
a bandwidth cap, plus server-side GET/byte counters), virtualizes it, and times
reads through any (store, pipeline). All runs: <code>mode=45 ms</code> latency,
200 MB/s, concurrency 16.</p>
<p>Two sources: a <strong>synthetic</strong> grid of 32 KiB chunks, and
<strong>NDPI</strong> — the <em>real tile geometry</em> of a whole-slide image
(119k tiny JPEG tiles, <code>jpeg=False</code> so bytes are a zero blob and the
focus is fetch behavior). Access patterns: <code>band</code> (contiguous strip),
<code>subcube</code> (square ROI), <code>column</code> (scattered down the file),
<code>timeseries</code> (isolated chunks).</p>
<dl class="cases">{cases_dl}</dl>

<h2>Finding 1 — Icechunk fragments virtual reads under the default config</h2>
<p>Reading virtual chunks through Icechunk's HTTP virtual-chunk container,
<code>RepositoryConfig.default()</code> splits each chunk read into ~5 smaller
range GETs. Its own documented default for
<code>ideal_concurrent_request_size</code> is 12 MB — but the default config
leaves it unset, and unset fragments. Verified standalone (16 chunks of 32 KiB):</p>
<table><thead><tr><th>config</th><th>GETs</th></tr></thead><tbody>
<tr><td>RepositoryConfig.default() (unset)</td><td>80 (5×/chunk)</td></tr>
<tr class="win"><td>explicit ideal=12 MB (documented default)</td>
<td>16 (1×/chunk)</td></tr>
<tr class="win"><td>explicit ideal=64 MiB</td><td>16</td></tr>
</tbody></table>
<p class="note">This only bites large chunks (&gt; the request size). NDPI tiles
are tiny, so there icechunk ≈ manifest. Details in
<code>icechunk-virtual-chunk-fragmentation.md</code>.</p>

<h2>Finding 2 — Pattern sweep</h2>
<h3>NDPI (real tile geometry)</h3>
{_img(HERE / "sweep_ndpi.png")}
{pattern_table(ndpi) if ndpi else "<p><em>no NDPI results</em></p>"}
<p>On the real geometry, <code>manifest ≈ icechunk ≈ icechunk-bigreq</code>
(tiles are tiny, nothing to fragment). Coalescing is a big win where data is
contiguous (<code>band</code>, <code>subcube</code>) with little/no over-read,
and a genuine over-read-vs-latency tradeoff where it is scattered
(<code>column</code>).</p>

<h3>Synthetic (32 KiB chunks)</h3>
{_img(HERE / "sweep_synthetic.png")}
{pattern_table(syn) if syn else "<p><em>no synthetic results</em></p>"}
<p>With large chunks the icechunk-config effect is visible:
<code>icechunk</code> (unset) fragments while <code>icechunk-bigreq</code>
matches <code>manifest</code>.</p>

<h2>Why is the over-read so extreme?</h2>
<p>NDPI tiles within a tile-row are stored contiguously (gap 0), but consecutive
rows are ~41–56 KB apart. The old default <code>max_gap=256 KiB</code> bridges
those row gaps, so a whole <code>column</code> chains into <strong>one 184 MB GET
for 3.2 MB of useful tiles (57×)</strong>. The fix is two levers:
<code>max_gap</code> below the row gap (don't bridge — <code>coalesced</code>
16 KB merges only contiguous runs, ~zero over-read), and
<code>max_coalesced_bytes</code> (cap a span so one request can't swallow the
file — <code>coalesced-cap</code>).</p>

<h2>Finding 3 — Decode-cost crossover (streaming vs single-GET barrier)</h2>
<p>Coalescing to a <em>single</em> GET (<code>coalesced-wide</code>) can't overlap
decode with fetch — all bytes land at once. Many smaller GETs
(<code>coalesced</code>, <code>coalesced-cap</code>) stream, so decode hides under
in-flight fetches. Sweeping per-chunk decode cost on the <code>subcube</code> ROI:</p>
<h3>NDPI</h3>
{_img(HERE / "decode_ndpi.png")}
{decode_pivot(dndpi) if dndpi else "<p><em>no NDPI decode results yet</em></p>"}
<h3>Synthetic</h3>
{_img(HERE / "decode_synthetic.png")}
{decode_pivot(dsyn) if dsyn else "<p><em>no synthetic decode results</em></p>"}
<p>The single-GET barrier starts fastest at zero decode but climbs steeply and
crosses above the streaming small-gap case once decode isn't free — so the right
objective is <strong>wall-clock-to-last-decode, not GET count</strong>.</p>

<h2>Design implications</h2>
<ul>
<li>The cost model needs <strong>both</strong> a gap threshold and a span-size
cap; merging everything into one span maximizes over-read and kills pipelining.</li>
<li>Optimize for time-to-last-decode under latency×bandwidth, not request count.</li>
<li>Coalescing helps most exactly where stock virtual reads hurt most — many
small, near-contiguous chunks under latency (NDPI subcube: ~30× faster).</li>
<li>The Icechunk-native path gets streaming concurrency for free; the Python
semaphore here is a stand-in, not to be ported.</li>
</ul>
</body></html>"""

    out = HERE / "report.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    print(f"wrote {build()}")
