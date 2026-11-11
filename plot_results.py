#!/usr/bin/env python3
"""
Plots for the README / LinkedIn, from the files evaluate.py writes.

    results/density_timeline.png   one crowd: peak density over time, with and without drones,
                                   escalation steps marked
    results/per_seed.png           every crowd: seconds at >= 4 people/m2, with and without drones

Usage:
    python plot_results.py                          # uses results/evaluation.json, seed 5
    python plot_results.py --label after_prediction --seed 5

Run evaluate.py with the same --label first. All densities are simulator ground truth.
"""

import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated two-series pair (reference palette, light-surface steps)
OFF_COL, ON_COL = "#eb6834", "#2a78d6"
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def timeline_plot(seed, path):
    runs = {}
    for tag in ("off", "on"):
        with open(os.path.join("results", "runs", f"seed{seed}_{tag}.json")) as f:
            runs[tag] = json.load(f)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    style(ax)
    for tag, col, name in (("off", OFF_COL, "Drones only watch"), ("on", ON_COL, "Drones intervene")):
        t = [s["t"] for s in runs[tag]["timeline"]]
        v = [s["gt_peak_density"] for s in runs[tag]["timeline"]]
        ax.plot(t, v, color=col, linewidth=2, label=name)
        ax.annotate(name, (t[-1], v[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT)                    # direct label
    for level, text in ((4, "movement restricted"), (5, "dangerous")):
        ax.axhline(level, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(1, level + 0.08, f"{level} people/m² - {text}", fontsize=8, color=TEXT_2)
    shown = set()
    for e in runs["on"]["events"]:
        for key in ("SPOTLIGHT", "ANNOUNCE", "SECURITY"):
            if e["text"].startswith(key + ":") and key not in shown:
                shown.add(key)
                ax.axvline(e["t"], color=ON_COL, linewidth=1, alpha=0.5)
                ax.text(e["t"] + 0.6, 6.55, key.lower(), fontsize=8, color=TEXT_2, rotation=90, va="top")
    ax.set_ylim(0, 6.8)
    ax.set_xlim(0, runs["on"]["duration_s"])
    ax.set_xlabel("time (s)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("peak density (people/m²)", color=TEXT_2, fontsize=9)
    ax.set_title(f"Same crowd (seed {seed}), with and without CrowdFlow", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def per_seed_plot(result, path):
    rows = result["runs"]
    seeds = sorted({r["seed"] for r in rows})
    val = {(r["seed"], r["interventions"]): r["s_ge4"] for r in rows}
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    style(ax)
    w = 0.38
    for k, (on, col, name) in enumerate(((False, OFF_COL, "Drones only watch"),
                                         (True, ON_COL, "Drones intervene"))):
        xs = [i + (k - 0.5) * (w + 0.02) for i in range(len(seeds))]
        ax.bar(xs, [val[(s, on)] for s in seeds], width=w, color=col, label=name)
    ax.set_ylim(0, 1.3 * max(val.values()))                 # headroom for the legend
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"crowd {s}" for s in seeds], fontsize=8)
    ax.set_ylabel("seconds at ≥ 4 people/m²", color=TEXT_2, fontsize=9)
    off, on = result["drones_observe_only"], result["drones_intervene"]
    ax.set_title(f"{len(seeds)} simulated crowds: time at high density "
                 f"(mean {off['s_ge4_mean']:.0f} s → {on['s_ge4_mean']:.0f} s)",
                 color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="evaluation", help="results/<label>.json from evaluate.py")
    ap.add_argument("--seed", type=int, default=5, help="crowd shown in the timeline plot")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join("results", f"{args.label}.json")) as f:
        result = json.load(f)
    timeline_plot(args.seed, os.path.join("results", "density_timeline.png"))
    per_seed_plot(result, os.path.join("results", "per_seed.png"))
    print("Wrote results/density_timeline.png and results/per_seed.png")


if __name__ == "__main__":
    main()
