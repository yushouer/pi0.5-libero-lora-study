#!/usr/bin/env python3
"""Regenerate project figures from CSV results. No ML dependencies required.
Usage (from repository root): python scripts/plot_results.py
Requires: matplotlib (optional: Noto Sans CJK font)
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Custom fonts are not bundled. On systems without Noto Sans CJK, the charts
# still render using DejaVu Sans because all plotted labels are in English.
font_names = {font.name for font in font_manager.fontManager.ttflist}
FONT = "Noto Sans CJK SC" if "Noto Sans CJK SC" in font_names else "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#D6DFEA",
    "axes.labelcolor": "#516074",
    "text.color": "#18243A",
    "xtick.color": "#617084",
    "ytick.color": "#617084",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
TEAL = "#007F83"
GOLD = "#E3A13A"
SLATE = "#697991"
RED = "#C85B54"
MUTED = "#E8EEF3"

with (ROOT / "results" / "checkpoint_metrics.csv").open(encoding="utf-8", newline="") as f:
    data = list(csv.DictReader(f))
with (ROOT / "results" / "failure_cases.csv").open(encoding="utf-8-sig", newline="") as f:
    failures = list(csv.DictReader(f))

steps = [int(d["checkpoint_step"]) for d in data]
succ = [int(d["successes"]) for d in data]
trials = [int(d["trials"]) for d in data]
percent = [100 * k / n for k, n in zip(succ, trials)]
loss = [float(d["validation_flow_matching_loss"]) for d in data]


def wilson_interval(k: int, n: int, z: float = 1.9599639845):
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return 100 * (mid-half), 100 * (mid+half)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E8EDF2", zorder=0)
    ax.set_axisbelow(True)


def draw_success(ax, compact=False):
    errs = [wilson_interval(k, n) for k, n in zip(succ, trials)]
    x = range(len(steps))
    ax.bar(x, percent, color=["#8FBAB8", "#4EA2A2", TEAL], width=.56, zorder=2)
    ax.errorbar(x, percent,
                yerr=[[p-e[0] for p,e in zip(percent,errs)],
                      [e[1]-p for p,e in zip(percent,errs)]],
                fmt="none", ecolor="#26364B", capsize=5, capthick=1.5,
                elinewidth=1.6, zorder=3)
    for i, (p,k,n) in enumerate(zip(percent,succ,trials)):
        upper = errs[i][1]
        ax.text(i, upper + 5, f"{p:.0f}%  ({k}/{n})",
                ha="center", va="bottom", color="#142A3B", fontsize=10, weight="bold")
    ax.set_xticks(list(x), [f"{s:,}" for s in steps])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100], ["0%","20%","40%","60%","80%","100%"])
    ax.set_xlabel("Checkpoint step")
    ax.set_title("Closed-loop task success", loc="left", pad=15)
    clean_axes(ax)
    if not compact:
        ax.text(0.0, -0.26,
                "Same book task; 50 rollouts/checkpoint; error bars: Wilson 95% CI.",
                fontsize=9, color=SLATE, transform=ax.transAxes)


def draw_loss(ax, compact=False):
    ax.plot(range(3), loss, color=TEAL, linewidth=2.5, marker="o", markersize=9)
    ax.fill_between(range(3), loss, [0.0375]*3, color=TEAL, alpha=0.075)
    for i, y in enumerate(loss):
        ax.annotate(f"{y:.5f}", (i, y), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=10,
                    weight="bold", color="#20324A")
    ax.set_xticks(range(3), [f"{s:,}" for s in steps])
    ax.set_ylim(0.0375, 0.0424)
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel("Mean Flow Matching loss")
    ax.set_xlabel("Checkpoint step")
    ax.set_title("Held-out validation loss", loc="left", pad=15)
    clean_axes(ax)
    if not compact:
        decrease = (loss[0] - loss[-1]) / loss[0] * 100
        ax.text(0.0, -0.26,
                f"6 held-out episodes / 1,222 frames.  Relative decrease: {decrease:.2f}%.  Note: zoomed y-axis.",
                fontsize=9, color=SLATE, transform=ax.transAxes)


def draw_failures(ax, compact=False):
    cats = Counter(row["category"] for row in failures)
    labels = ["Placement incomplete", "Grasp unsuccessful", "Contact destabilization"]
    keys = ["placement_incomplete", "grasp_failed", "contact_destabilization"]
    colors = [TEAL, GOLD, RED]
    values = [cats[k] for k in keys]
    yy=[2,1,0]
    ax.barh(yy, values, height=.6, color=colors, zorder=2)
    ax.set_yticks(yy, labels)
    ax.set_xlim(0, 15.5)
    for y, val in zip(yy, values):
        ax.text(val + .3, y, f"{val}/21  ({100*val/21:.1f}%)",
                va="center", ha="left", fontsize=10, weight="bold", color="#24364C")
    ax.set_xticks([0,5,10,15])
    ax.grid(axis="x", color="#E8EDF2")
    ax.set_title("Failure modes (video review)", loc="left", pad=15)
    ax.set_xlabel("Failed episodes")
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    if not compact:
        ax.text(0.0, -0.31,
                "Mutually exclusive visual labels; not confirmed root causes.",
                fontsize=9, color=SLATE, transform=ax.transAxes)


def draw_params(ax, compact=False):
    total = 3403421456
    vals = [49987584,416969488,2936464384]
    cats = ["LoRA", "Other trainable", "Frozen"]
    colors = [TEAL, GOLD, "#B7C5D4"]
    shares=[100*v/total for v in vals]
    left=0
    for v,c in zip(shares,colors):
        ax.barh([0], [v], left=left, color=c, height=.44, edgecolor="white",linewidth=1.3)
        left+=v
    ax.set_xlim(0,100)
    ax.set_ylim(-.7,.8)
    ax.set_yticks([])
    ax.set_xticks([0,25,50,75,100], ["0%","25%","50%","75%","100%"])
    ax.grid(axis="x",color="#E8EDF2")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("3.403B model parameter allocation", loc="left", pad=18)
    label_y=-.40
    for i,(name,v,pct,c) in enumerate(zip(cats,vals,shares,colors)):
        # axes-fraction positioning keeps labels readable even for 1.47% LoRA slice
        ax.text([.015,.34,.70][i], .06 if compact else -.05,
                f"{name}: {pct:.2f}%\n{v/1e6:,.2f}M",
                transform=ax.transAxes, color=c if i<2 else SLATE,
                fontsize=9, va="top", weight="bold")
    if not compact:
        ax.text(.0,-.32,"Trainable total: 466.96M (13.72%); this is not pure LoRA.",
                color=SLATE,fontsize=9,transform=ax.transAxes)


def save_single(name, fun, figsize=(8.5,5.4), bottom=.23, left=.16):
    fig,ax=plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=left, right=.94, top=.84, bottom=bottom)
    fun(ax)
    fig.savefig(FIG/name, dpi=200)
    plt.close(fig)


save_single("success_rate.png",draw_success,bottom=.27,left=.12)
save_single("validation_loss.png",draw_loss,bottom=.27,left=.16)
save_single("failure_modes.png",draw_failures,bottom=.29,left=.29)
save_single("parameter_allocation.png",draw_params,figsize=(9,3.9),bottom=.36,left=.08)

fig,axs=plt.subplots(2,2,figsize=(15,10))
fig.subplots_adjust(left=.13,right=.96,top=.86,bottom=.1,wspace=.34,hspace=.51)
fig.suptitle("π0.5 LoRA · LIBERO Book Task | Experimental Overview",
             fontsize=20,weight="bold",y=.974,x=.11,ha="left",color="#162238")
fig.text(.11,.927,"Single RTX 4090  ·  35 train episodes  ·  6 validation episodes  ·  6,000 training steps",
         fontsize=11,color=SLATE)
draw_success(axs[0,0],compact=True)
draw_loss(axs[0,1],compact=True)
draw_failures(axs[1,0],compact=True)
draw_params(axs[1,1],compact=True)
fig.text(.11,.028,"Source: local experiment logs, checkpoint validation and 50 recorded rollouts.  Failure labels are observational.",
         fontsize=9,color=SLATE)
fig.savefig(FIG/"experiment_overview.png",dpi=195)
plt.close(fig)
print('Figures written to:',FIG)
for p in FIG.iterdir():
    print(p.name,p.stat().st_size)
