"""README figures from benchmarks/results/*.csv: latency per kernel version (mma.sync path, SM100 tcgen05 path), library
comparison, roofline, the SM120 gap.

  uv run --with matplotlib python benchmarks/plot.py      # writes docs/img/<name>-{light,dark}.svg
"""
import csv
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, IMG = ROOT / "benchmarks" / "results", ROOT / "docs" / "img"

# Achievable DRAM read BW (GB/s) and mma.sync BF16 / E4M3 peaks (TFLOPS), measured with roofline.py
GPUS = {"RTX PRO 6000": dict(csv="rtx_pro_6000.csv", bw=1559, mma=455, mma8=897),
        "B200": dict(csv="b200.csv", bw=7300, mma=558, mma8=None)}
FLOP_PER_TOKEN_HEAD = 2 * 576 + 2 * 512  # QK over 512 + 64 channels, PV over 512, q_len 1
KV_BYTES = {"nvfp4": 352, "nvfp4-fp8": 352, "nvfp4-fp4q": 352, "nvfp4-tc": 352, "cake": 352, "nvfp4+fa2": 352, "fp8": 576,
            "bf16": 1152}
# BF16 path v1-v5 on both GPUs; the FP8 path (opt-in, for compute-bound shapes) v6-v11 on RTX PRO 6000
VERSIONS = {"RTX PRO 6000": ["v1", "v2", "v3", "v4", "v5", "v6", "v9", "v11", "v11c"],
            "B200": ["v1", "v2", "v3", "v4", "v5"]}
TICKS = {"v1": "v1\nCUDA\ncores", "v2": "v2\nmma.sync", "v3": "v3\ncp.async", "v4": "v4\nsplit-KV",
         "v5": "v5\nfewer\ninstrs", "v6": "v6\nFP8\nMMA", "v9": "v9\nQK by\ntoken", "v11": "v11\n8-bit\nldmatrix",
         "v11c": "v11c\nswizzle"}
OURS = ("nvfp4", "nvfp4-fp8", "nvfp4-tc")
COLOR = {"nvfp4": 0, "cake": 1, "nvfp4-fp8": 2, "nvfp4-tc": 3}  # categorical slot per provider, the same in every chart
# SM100 tcgen05 (CuTe DSL) versions the current code still builds; v2-v2.2 are in the optimization log only
TC_VERSIONS = ["tc-v0", "tc-v1a", "tc-v1b", "tc-v2.3", "tc-v2.4", "tc-v3.1", "tc-v3.2"]
TC_TICKS = {"tc-v0": "v0\nFP16 tile\nin smem", "tc-v1a": "v1a'\nFP4 QK\n(mxf4nvf4)", "tc-v1b": "v1b\nV into\nTMEM",
            "tc-v2.3": "v2.3\nwarp\nroles", "tc-v2.4": "v2.4\nTMA, 8\nconvert warps",
            "tc-v3.1": "v3.1\nQMUL4 V,\nmax by vote", "tc-v3.2": "v3.2\n32 heads\nper CTA"}
TABLE_ONLY = ("nvfp4+fa2", "nvfp4-fp4q")  # in the README tables, not in the bar charts
CONFIGS = [(16, 64, 65536), (16, 1, 65536), (128, 64, 65536), (128, 1, 65536)]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", gray="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", gray="#52514e", series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}


def name(provider, gpu):
    if provider == "trtllm:fp8":
        return "XQA (FP8)" if gpu == "RTX PRO 6000" else "trtllm-gen (FP8)"
    return {"nvfp4": "nvfp4-mla (BF16 mode)", "nvfp4-fp8": "nvfp4-mla (FP8 mode)", "nvfp4-fp4q": "nvfp4-mla (FP4-QK mode)",
            "nvfp4-tc": "nvfp4-mla tcgen05 (SM100)",
            "cutile": "FlashInfer cuTile (BF16)", "cute-dsl-monolithic": "FlashInfer CuTe DSL (BF16)",
            "cake": "FlashInfer Cake (NVFP4)", "triton:fp8": "SGLang Triton (FP8)",
            "triton:bf16": "SGLang Triton (BF16)", "fa2": "FlashInfer fa2 (BF16)", "cutlass": "cutlass (BF16)",
            "trtllm:bf16": "trtllm-gen (BF16)", "nvfp4+fa2": "dequant + fa2", "b12x:fp8": "FlashInfer b12x (FP8)",
            "b12x:bf16": "FlashInfer b12x (BF16)"}[provider]


def kv_bytes(provider):
    return KV_BYTES.get(provider) or KV_BYTES["fp8" if provider.endswith("fp8") else "bf16"]


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        r.update(heads=int(r["heads"]), batch=int(r["batch"]), s_kv=int(r["s_kv"]), ms=float(r["ms"]))
        if math.isfinite(r["ms"]):
            rows.append(r)
    return rows


def cfg_label(h, b, s):
    return f"H={h}, batch {b}, {s // 1024}K tokens"


def fmt_ms(ms):
    return f"{ms * 1e3:.0f} µs" if ms < 1 else f"{ms:.2f} ms"


def style(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0)
    if grid_axis:
        ax.grid(True, axis=grid_axis, which="major", color=t["grid"], lw=0.6)
    ax.set_axisbelow(True)


def title(ax, t, text):
    ax.set_title(text, loc="left", color=t["ink"], fontsize=10, fontweight="bold")


def versions(t):
    rows = load(RES / "versions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, gridspec_kw={"width_ratios": [8.8, 5.8]})
    for ax, gpu in zip(axes, GPUS):
        style(ax, t)
        vs = VERSIONS[gpu]
        for i, (h, b, s) in enumerate(CONFIGS[:3]):
            ms = {r["version"]: r["ms"] for r in rows
                  if r["gpu"] == gpu and (r["heads"], r["batch"], r["s_kv"]) == (h, b, s)}
            n = len(vs) if h == 128 else 5  # at H=16 the BF16 path (v5) stays the faster one
            ys = [ms[v] for v in vs[:n]]
            ax.plot(range(n), ys, color=t["series"][i], lw=2, marker="o", ms=6, mec=t["surface"], mew=1.5,
                    label=cfg_label(h, b, s), zorder=3)
            ax.annotate(f"{fmt_ms(ys[-1])}  ({ys[0] / ys[-1]:.0f}×)", (n - 1, ys[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", color=t["ink2"], fontsize=8.5)
        ax.set_yscale("log")
        ax.set_xticks(range(len(vs)), [TICKS[v] for v in vs], fontsize=8)
        ax.set_xlim(-0.3, len(vs) + 0.4)
        if len(vs) > 5:
            ax.axvline(4.5, color=t["grid"], lw=1, zorder=1)
            ax.text(4.6, 0.97, "FP8 path", transform=ax.get_xaxis_transform(), color=t["muted"], fontsize=8, va="top")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:g}")
        title(ax, t, gpu)
    axes[0].set_ylabel("latency (ms, log scale)", color=t["ink2"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, labelcolor=t["ink2"], fontsize=9)
    fig.text(0.01, 0.01, "Lower is better. v1-v5: BF16 path; v6-v11c: the FP8 path (opt-in), shown for H=128. v1-v3 timed "
             "with CUDA events, later versions with CUPTI; L2 flushed.\nLabels: last latency (speedup over v1).",
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    return fig


def sm100(t):
    """B200, the tcgen05 kernel version by version, with Cake on the same shapes."""
    rows = load(RES / "versions.csv")
    cake = {(r["heads"], r["batch"], r["s_kv"]): r["ms"] for r in load(RES / "b200.csv") if r["provider"] == "cake"}
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    style(ax, t)
    n = len(TC_VERSIONS)
    for i, (h, b, s) in enumerate(CONFIGS[:3]):
        ms = {r["version"]: r["ms"] for r in rows
              if r["gpu"] == "B200" and (r["heads"], r["batch"], r["s_kv"]) == (h, b, s)}
        ys = [ms[v] for v in TC_VERSIONS]
        ax.plot(range(n), ys, color=t["series"][i], lw=2, marker="o", ms=6, mec=t["surface"], mew=1.5,
                label=cfg_label(h, b, s), zorder=3)
        ax.annotate(f"{fmt_ms(ys[-1])}  ({ys[0] / ys[-1]:.1f}×)", (n - 1, ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=t["ink2"], fontsize=8.5)
        c = cake[h, b, s]
        ax.plot([n + 0.9], [c], ls="none", marker="D", ms=6, mfc="none", mec=t["series"][i], mew=1.5, zorder=3)
        ax.annotate(fmt_ms(c), (n + 0.9, c), xytext=(8, 0), textcoords="offset points", va="center",
                    color=t["ink2"], fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(list(range(n)) + [n + 0.9], [TC_TICKS[v] for v in TC_VERSIONS] + ["FlashInfer\nCake\n(NVFP4 Q)"],
                  fontsize=8)
    ax.axvline(n + 0.2, color=t["grid"], lw=1, zorder=1)
    ax.set_xlim(-0.3, n + 1.9)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:g}")
    ax.set_ylabel("latency (ms, log scale)", color=t["ink2"])
    title(ax, t, "B200: the tcgen05 kernel (CuTe DSL)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=False, labelcolor=t["ink2"], fontsize=8.5)
    fig.text(0.01, 0.01, "Lower is better. One process, median of 3 alternating rounds (CUPTI), L2 flushed; page size 64. "
             "Labels: last latency (speedup over v0).\nv2-v2.2 are in the optimization log only: v2.3's load order "
             "replaced theirs, so the current code no longer builds them.", color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def comparison(t, gpu):
    rows = load(RES / GPUS[gpu]["csv"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for ax, (h, b, s) in zip(axes.flat, CONFIGS):
        style(ax, t, grid_axis=None)
        pts = sorted(((r["ms"], r["provider"]) for r in rows
                      if (r["heads"], r["batch"], r["s_kv"]) == (h, b, s) and r["provider"] not in TABLE_ONLY),
                     reverse=True)
        ys = range(len(pts))
        ax.barh(ys, [m for m, _ in pts], height=0.62,
                color=[t["series"][COLOR[p]] if p in COLOR else t["gray"] for _, p in pts])
        ax.set_yticks(ys, [name(p, gpu) for _, p in pts], fontsize=8.5)
        top = max(m for m, _ in pts)
        for y, (m, p) in zip(ys, pts):
            ax.text(m + top * 0.015, y, fmt_ms(m), va="center", color=t["ink" if p in OURS else "ink2"],
                    fontsize=8.5, fontweight="bold" if p in OURS else "normal")
        ax.set_xlim(0, top * 1.22)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        title(ax, t, cfg_label(h, b, s))
    fig.text(0.01, 0.01, f"{gpu}, q_len 1, page size 64. GPU kernel time per call (CUPTI), L2 flushed. "
             "Lower is better.", color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=2.0)
    return fig


# hand-placed roofline labels that would otherwise hit the roof or the v1 -> v5 line
LABEL_AT = {("RTX PRO 6000", "triton:fp8", 16): (-9, -2, "right", "top"),
            ("RTX PRO 6000", "trtllm:fp8", 128): (6, 4, "left", "bottom"),
            ("B200", "cake", 16): (-8, 0, "right", "center"),
            ("B200", "nvfp4", 16): (-8, 0, "right", "center"),
            ("B200", "nvfp4-tc", 16): (7, -2, "left", "top"),
            ("B200", "nvfp4-tc", 128): (7, -2, "left", "top"),
            ("RTX PRO 6000", "nvfp4", 128): (8, -6, "left", "top")}


def roofline(t):
    vrows = load(RES / "versions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (gpu, g) in zip(axes, GPUS.items()):
        style(ax, t, grid_axis="both")
        ax.set_xscale("log")
        ax.set_yscale("log")
        xs = [10 * 1.05 ** i for i in range(130)]
        ax.plot(xs, [min(g["bw"] * x / 1e3, g["mma"]) for x in xs], color=t["ink2"], lw=1.5, zorder=2)
        if g["mma8"]:
            ax.plot(xs, [min(g["bw"] * x / 1e3, g["mma8"]) for x in xs], color=t["muted"], lw=1.2, zorder=2)
        fp8 = f" / E4M3 {g['mma8']}" if g["mma8"] else ""
        ax.text(0.98, 0.03, f"roofs: DRAM {g['bw']} GB/s,\nmma.sync BF16 {g['mma']}{fp8} TFLOPS", transform=ax.transAxes,
                ha="right", va="bottom", color=t["ink2"], fontsize=8)
        tflops = lambda r: FLOP_PER_TOKEN_HEAD * r["heads"] * r["batch"] * r["s_kv"] / (r["ms"] * 1e-3) / 1e12
        intensity = lambda p, h: FLOP_PER_TOKEN_HEAD * h / kv_bytes(p)
        for h in (16, 128):
            vs = VERSIONS[gpu] if h == 128 else VERSIONS[gpu][:5]
            pts = sorted((vs.index(r["version"]), tflops(r)) for r in vrows if r["version"] in vs
                         and r["gpu"] == gpu and (r["heads"], r["batch"], r["s_kv"]) == (h, 64, 65536))
            x = intensity("nvfp4", h)
            ax.plot([x] * len(pts), [y for _, y in pts], color=t["series"][0], lw=2, marker="o", ms=6,
                    mec=t["surface"], mew=1.5, zorder=4)
            last = vs[len(pts) - 1]
            fp8_roof = last not in VERSIONS["B200"]  # v6 on: FP8 MMA
            roof = min(g["bw"] * x / 1e3, g["mma8"] if fp8_roof else g["mma"])
            dx, dy, ha, va = LABEL_AT.get((gpu, "nvfp4", h), (8, 0, "left", "center"))
            ax.annotate(f"H={h} {last}\n{100 * pts[-1][1] / roof:.0f}% of {'E4M3 ' if fp8_roof else ''}roof",
                        (x, pts[-1][1]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=t["ink"],
                        fontsize=8)
            ax.annotate("v1", (x, pts[0][1]), xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center", color=t["ink2"], fontsize=8)
        # only the rival per head count (fastest other library) and Cake; the bar charts carry the rest
        rows = [r for r in load(RES / g["csv"]) if (r["batch"], r["s_kv"]) == (64, 65536)
                and r["provider"] not in OURS + TABLE_ONLY]
        rivals = {(min((r for r in rows if r["heads"] == h), key=lambda r: r["ms"])["provider"], h) for h in (16, 128)}
        for r in rows:
            if (r["provider"], r["heads"]) not in rivals and r["provider"] != "cake":
                continue
            cake = r["provider"] == "cake"
            x, y = intensity(r["provider"], r["heads"]), tflops(r)
            ax.plot(x, y, ls="none", marker="D" if cake else "s", ms=6, color=t["series"][1] if cake else t["gray"],
                    mec=t["surface"], mew=1.5, zorder=3)
            dx, dy, ha, va = LABEL_AT.get((gpu, r["provider"], r["heads"]), (7, 0, "left", "center"))
            ax.annotate(f"{name(r['provider'], gpu)}\nH={r['heads']}", (x, y), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va, color=t["ink2"], fontsize=7.5,
                        bbox=dict(boxstyle="square,pad=0.1", fc=t["surface"], ec="none"))
        for r in load(RES / g["csv"]):  # SM100: this kernel's tcgen05 path, above the mma.sync roof it does not use
            if r["provider"] != "nvfp4-tc" or (r["batch"], r["s_kv"]) != (64, 65536):
                continue
            x, y = intensity("nvfp4-tc", r["heads"]), tflops(r)
            ax.plot(x, y, ls="none", marker="o", ms=6, color=t["series"][COLOR["nvfp4-tc"]], mec=t["surface"], mew=1.5,
                    zorder=4)
            dram = f", {100 * y / (g['bw'] * x / 1e3):.0f}% of\nDRAM roof" if r["heads"] == 16 else ""
            dx, dy, ha, va = LABEL_AT.get((gpu, "nvfp4-tc", r["heads"]), (7, 0, "left", "center"))
            ax.annotate(f"tcgen05 v3\nH={r['heads']}{dram}", (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, color=t["ink"], fontsize=8)
        ax.set_xlim(10, xs[-1])
        ax.set_ylim(1, 3000)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.set_xlabel("arithmetic intensity (flop per KV byte)", color=t["ink2"])
        title(ax, t, gpu)
    axes[0].set_ylabel("achieved TFLOPS", color=t["ink2"])
    fig.text(0.01, 0.01, "Batch 64, 64K tokens, q_len 1. Compute roofs are this kernel's mma.sync: BF16 for v1-v5, E4M3 for "
             "the FP8 path (v6-v11).\ntcgen05 kernels (trtllm-gen, Cake, this kernel's SM100 path) have higher roofs, not "
             "drawn.", color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def gap(t):
    """SM120, H=128 batch 64: what is left of the gap to XQA once the FP4 -> FP8 conversion costs nothing (E1)."""
    rows = load(RES / "sm120_gap.csv")
    ms = {(r["provider"], r["s_kv"]): r["ms"] for r in rows}
    ss = sorted({r["s_kv"] for r in rows})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [5, 6]})
    style(a1, t)
    a1.axhline(1, color=t["muted"], lw=1.2, zorder=2)
    a1.text(0, 1.01, "XQA (FP8) = 1", va="bottom", color=t["ink2"], fontsize=8.5)
    for p, color, ls in (("v11c FP8", t["series"][2], "-"), ("E1 (no conversion)", t["ink2"], "--")):
        ys = [ms[p, s] / ms["XQA FP8", s] for s in ss]
        a1.plot(range(len(ss)), ys, color=color, ls=ls, lw=2, marker="o", ms=6, mec=t["surface"], mew=1.5, label=p,
                zorder=3)
        a1.annotate(f"{p}\n{ys[-1]:.2f}×", (len(ss) - 1, ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=t["ink2"], fontsize=8.5)
    a1.set_xticks(range(len(ss)), [f"{s // 1024}K" for s in ss])
    a1.set_xlabel("KV tokens per request", color=t["ink2"])
    a1.set_xlim(-0.3, len(ss) + 0.6)
    a1.set_ylim(0.9, None)
    a1.set_ylabel("time / XQA time", color=t["ink2"])
    title(a1, t, "H=128, batch 64: relative to XQA")
    a1.legend(loc="upper right", frameon=False, labelcolor=t["ink2"], fontsize=8.5)

    style(a2, t, grid_axis=None)
    S = ss[-1]
    bars = [("v11c FP8", t["series"][2], None), ("v11c FP4 QK", t["series"][2], None),
            ("E1 (no conversion)", t["gray"], "///"), ("XQA FP8", t["gray"], None)][::-1]
    ys = range(len(bars))
    vals = [ms[p, S] for p, _, _ in bars]
    for patch, (_, _, h) in zip(a2.barh(ys, vals, height=0.62, color=[c for _, c, _ in bars], edgecolor=t["surface"],
                                        lw=0), bars):
        patch.set_hatch(h)  # E1 is not a real kernel: textured, so it does not read as a library bar
    a2.set_yticks(ys, [p for p, _, _ in bars], fontsize=8.5)
    for y, v, (p, _, _) in zip(ys, vals, bars):
        a2.text(v + max(vals) * 0.015, y, fmt_ms(v), va="center", color=t["ink" if p.startswith("v11c") else "ink2"],
                fontsize=8.5, fontweight="bold" if p.startswith("v11c") else "normal")
    a2.set_xlim(0, max(vals) * 1.2)
    a2.set_xticks([])
    a2.spines["bottom"].set_visible(False)
    title(a2, t, f"H=128, batch 64, {S // 1024}K tokens")
    fig.text(0.01, 0.01, "RTX PRO 6000, one process, median of 3 alternating rounds (CUPTI). E1 is a test build of the "
             "FP8 mode with the FP4 -> FP8 conversion replaced\nby two logic ops per word: wrong results, same control "
             "flow. Removing 40% of the instructions saves 12% at 64K; the rest of the gap is the kernel's serial chain.",
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def main():
    plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none", "font.size": 9, "svg.hashsalt": "nvfp4-mla",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    IMG.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        figs = {"versions": versions(t), "roofline": roofline(t), "sm120-gap": gap(t), "sm100-versions": sm100(t),
                **{"compare-" + g.lower().replace(" ", "-"): comparison(t, g) for g in GPUS}}
        for key, fig in figs.items():
            fig.savefig(IMG / f"{key}-{mode}.svg", facecolor=t["surface"], metadata={"Date": None})  # stable diffs
            plt.close(fig)
    print("wrote", sorted(p.name for p in IMG.glob("*.svg")))


if __name__ == "__main__":
    main()
