"""README figures from benchmarks/results/*.csv: latency per kernel version, library comparison, roofline.

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

# Achievable DRAM read BW (GB/s) and mma.sync BF16 m16n8k16 peak (TFLOPS), measured with roofline.py
GPUS = {"RTX PRO 6000": dict(csv="rtx_pro_6000.csv", bw=1559, mma=455),
        "B200": dict(csv="b200.csv", bw=7300, mma=558)}
FLOP_PER_TOKEN_HEAD = 2 * 576 + 2 * 512  # QK over 512 + 64 channels, PV over 512, q_len 1
KV_BYTES = {"nvfp4": 352, "cake": 352, "nvfp4+fa2": 352, "fp8": 576, "bf16": 1152}
VERSIONS = ["v1", "v2", "v3", "v4", "v5"]
VERSION_TICKS = ["v1\nCUDA cores", "v2\nmma.sync", "v3\ncp.async", "v4\nsplit-KV", "v5\nfewer instrs"]
CONFIGS = [(16, 64, 65536), (16, 1, 65536), (128, 64, 65536), (128, 1, 65536)]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", gray="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", gray="#52514e", series=["#3987e5", "#d95926", "#199e70"]),
}


def name(provider, gpu):
    if provider == "trtllm:fp8":
        return "XQA (FP8)" if gpu == "RTX PRO 6000" else "trtllm-gen (FP8)"
    return {"nvfp4": "nvfp4-mla (NVFP4)", "cake": "FlashInfer Cake (NVFP4)", "triton:fp8": "SGLang Triton (FP8)",
            "triton:bf16": "SGLang Triton (BF16)", "fa2": "FlashInfer fa2 (BF16)", "cutlass": "cutlass (BF16)",
            "trtllm:bf16": "trtllm-gen (BF16)", "nvfp4+fa2": "dequant + fa2"}[provider]


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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, gpu in zip(axes, GPUS):
        style(ax, t)
        for i, (h, b, s) in enumerate(CONFIGS[:3]):
            ms = {r["version"]: r["ms"] for r in rows
                  if r["gpu"] == gpu and (r["heads"], r["batch"], r["s_kv"]) == (h, b, s)}
            ys = [ms[v] for v in VERSIONS]
            ax.plot(range(5), ys, color=t["series"][i], lw=2, marker="o", ms=6, mec=t["surface"], mew=1.5,
                    label=cfg_label(h, b, s), zorder=3)
            ax.annotate(f"{fmt_ms(ys[-1])}  ({ys[0] / ys[-1]:.0f}×)", (4, ys[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", color=t["ink2"], fontsize=8.5)
        ax.set_yscale("log")
        ax.set_xticks(range(5), VERSION_TICKS, fontsize=8.5)
        ax.set_xlim(-0.3, 5.2)
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:g}")
        title(ax, t, gpu)
    axes[0].set_ylabel("latency (ms, log scale)", color=t["ink2"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, labelcolor=t["ink2"], fontsize=9)
    fig.text(0.01, 0.01, "Lower is better. v1–v3 timed with CUDA events, v4–v5 with CUPTI kernel time; "
             "L2 flushed before each call. Labels: v5 latency (speedup over v1).", color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    return fig


def comparison(t, gpu):
    rows = load(RES / GPUS[gpu]["csv"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for ax, (h, b, s) in zip(axes.flat, CONFIGS):
        style(ax, t, grid_axis=None)
        pts = sorted(((r["ms"], r["provider"]) for r in rows if (r["heads"], r["batch"], r["s_kv"]) == (h, b, s)),
                     reverse=True)
        ys = range(len(pts))
        ax.barh(ys, [m for m, _ in pts], height=0.62,
                color=[t["series"][0] if p == "nvfp4" else t["series"][1] if p == "cake" else t["gray"]
                       for _, p in pts])
        ax.set_yticks(ys, [name(p, gpu) for _, p in pts], fontsize=8.5)
        top = max(m for m, _ in pts)
        for y, (m, p) in zip(ys, pts):
            ax.text(m + top * 0.015, y, fmt_ms(m), va="center", color=t["ink" if p == "nvfp4" else "ink2"],
                    fontsize=8.5, fontweight="bold" if p == "nvfp4" else "normal")
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
            ("B200", "cake", 16): (7, -4, "left", "top"),
            ("B200", "nvfp4", 16): (-8, 0, "right", "center")}


def roofline(t):
    vrows = load(RES / "versions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (gpu, g) in zip(axes, GPUS.items()):
        style(ax, t, grid_axis="both")
        ax.set_xscale("log")
        ax.set_yscale("log")
        xs = [10 * 1.05 ** i for i in range(130)]
        ax.plot(xs, [min(g["bw"] * x / 1e3, g["mma"]) for x in xs], color=t["ink2"], lw=1.5, zorder=2)
        ax.set_title(f"roofs: DRAM {g['bw']} GB/s, mma.sync BF16 {g['mma']} TFLOPS", loc="right", color=t["ink2"],
                     fontsize=8)
        tflops = lambda r: FLOP_PER_TOKEN_HEAD * r["heads"] * r["batch"] * r["s_kv"] / (r["ms"] * 1e-3) / 1e12
        intensity = lambda p, h: FLOP_PER_TOKEN_HEAD * h / kv_bytes(p)
        for h in (16, 128):
            pts = sorted((VERSIONS.index(r["version"]), tflops(r)) for r in vrows
                         if r["gpu"] == gpu and (r["heads"], r["batch"], r["s_kv"]) == (h, 64, 65536))
            x = intensity("nvfp4", h)
            ax.plot([x] * len(pts), [y for _, y in pts], color=t["series"][0], lw=2, marker="o", ms=6,
                    mec=t["surface"], mew=1.5, zorder=4, label="nvfp4-mla, v1 → v5" if h == 16 else None)
            roof = min(g["bw"] * x / 1e3, g["mma"])
            dx, dy, ha, va = LABEL_AT.get((gpu, "nvfp4", h), (8, 0, "left", "center"))
            ax.annotate(f"H={h} v5\n{100 * pts[-1][1] / roof:.0f}% of roof", (x, pts[-1][1]), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va, color=t["ink"], fontsize=8)
            ax.annotate("v1", (x, pts[0][1]), xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center", color=t["ink2"], fontsize=8)
        # only the rival per head count (fastest other library) and Cake; the bar charts carry the rest
        rows = [r for r in load(RES / g["csv"]) if (r["batch"], r["s_kv"]) == (64, 65536)
                and r["provider"] not in ("nvfp4", "nvfp4+fa2")]
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
                        textcoords="offset points", ha=ha, va=va, color=t["ink2"], fontsize=7.5)
        ax.set_xlim(10, xs[-1])
        ax.set_ylim(1, 3000)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.set_xlabel("arithmetic intensity (flop per KV byte)", color=t["ink2"])
        title(ax, t, gpu)
    axes[0].set_ylabel("achieved TFLOPS", color=t["ink2"])
    fig.text(0.01, 0.01, "Batch 64, 64K tokens, q_len 1. The compute roof is mma.sync BF16, the instruction this "
             "kernel uses; FP8/FP4 kernels have higher compute roofs.", color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main():
    plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none", "font.size": 9,
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    IMG.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        figs = {"versions": versions(t), "roofline": roofline(t),
                **{"compare-" + g.lower().replace(" ", "-"): comparison(t, g) for g in GPUS}}
        for key, fig in figs.items():
            fig.savefig(IMG / f"{key}-{mode}.svg", facecolor=t["surface"])
            plt.close(fig)
    print("wrote", sorted(p.name for p in IMG.glob("*.svg")))


if __name__ == "__main__":
    main()
