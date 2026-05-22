"""
Plot results from the 40-grid dataset-variation grid search.
=============================================================

Usage
-----
  python plot_dataset_grid_search_40grid.py                  # looks in current dir
  python plot_dataset_grid_search_40grid.py --results_dir /path/to/csvs
  python plot_dataset_grid_search_40grid.py --save          # save PNGs instead of showing

Expected input files
---------------------
  grid_search_dataset_40grid_A.csv   – Family A (manual augmentation)
  grid_search_dataset_40grid_B.csv   – Family B (epsilon variation)
  grid_search_dataset_40grid_C.csv   – Family C (random-start policy coverage)

Each figure shows 4 sub-plots:
  1. Convergence rate (binary 0/1, can be averaged over repeated seeds)
  2. Convergence distance (how far from goal the final state is)
  3. Final Reward
  4. V-Optimality Gap  (E_{s~d_π*}[V*(s) - V^π(s)])
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "font.family":   "sans-serif",
    "font.size":     12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi":    120,
})

# ─────────────────────────────────────────────────────────────
# CLI  (also safe to %run from a Jupyter notebook)
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot 40-grid dataset grid-search results")
parser.add_argument("--results_dir", default=None,
                    help="Directory containing the CSV files (default: data/results/grids/ relative to FOGAS root)")
parser.add_argument("--save", action="store_true",
                    help="Save figures as PNGs instead of displaying")
parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figures")
# parse_known_args ignores unknown flags injected by Jupyter / ipykernel
args, _unknown = parser.parse_known_args()

# ── Paths ────────────────────────────────────────────────────
# FOGAS project root  (this file lives at <root>/experiments/fogas/scripts/)
_SCRIPT_DIR = Path(__file__).resolve().parent          # …/experiments/fogas/scripts
_FOGAS_ROOT = _SCRIPT_DIR.parent.parent                # …/FOGAS

# Where the CSVs live
if args.results_dir is not None:
    results_dir = Path(args.results_dir).resolve()
else:
    results_dir = _FOGAS_ROOT / "datasets" / "grids"

# Where PNGs are saved (separate from CSVs)
_output_dir = _FOGAS_ROOT / "experiments" / "fogas" / "plots"
_output_dir.mkdir(parents=True, exist_ok=True)

print(f"  CSV source : {results_dir}")
print(f"  PNG output : {_output_dir}")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

METRICS = [
    ("convergence",  "Convergence Rate",          "Rate (0 / 1)",          True),
    ("conv_dist",    "Convergence Distance",       "Steps to Goal (↓ better)", False),
    ("final_reward", "Final Reward",               "Reward (↑ better)",     True),
    ("v_gap",        "V-Optimality Gap",           "Gap (↓ better)",       False),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["o", "s", "^", "D"]


def load_csv(name):
    path = results_dir / name
    if not path.exists():
        print(f"  ⚠️  File not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {path.name} — {len(df)} rows, columns: {list(df.columns)}")
    return df


def make_fig(title, nrows=1, ncols=4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    return fig, axes.flatten()


def plot_line_with_optional_hue(
    ax, df, x_col, metric_col, hue_col=None, invert=False,
    xlabel=None, ylabel=None, title=None,
):
    """
    If hue_col is None: plot a single (x → metric) line.
    If hue_col is set:  plot one line per unique value in hue_col.
    """
    ax.set_title(title or metric_col)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or metric_col)
    ax.grid(True, alpha=0.3)

    if hue_col is None:
        grouped = df.groupby(x_col)[metric_col].agg(["mean", "sem"]).reset_index()
        ax.plot(grouped[x_col], grouped["mean"], marker="o", linewidth=2,
                color=COLORS[0], label="mean")
        ax.fill_between(
            grouped[x_col],
            grouped["mean"] - grouped["sem"],
            grouped["mean"] + grouped["sem"],
            alpha=0.25, color=COLORS[0]
        )
    else:
        hue_vals = sorted(df[hue_col].unique())
        for i, hv in enumerate(hue_vals):
            sub = df[df[hue_col] == hv].groupby(x_col)[metric_col].agg(["mean", "sem"]).reset_index()
            c = COLORS[i % len(COLORS)]
            m = MARKERS[i % len(MARKERS)]
            ax.plot(sub[x_col], sub["mean"], marker=m, linewidth=2, color=c,
                    label=f"{hue_col}={hv}")
            ax.fill_between(
                sub[x_col],
                sub["mean"] - sub["sem"],
                sub["mean"] + sub["sem"],
                alpha=0.18, color=c
            )
        ax.legend(loc="best")

    if invert:
        ax.invert_yaxis()
    # Nice x-axis formatting
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))


def save_or_show(fig, name):
    plt.tight_layout()
    if args.save:
        out = _output_dir / f"{name}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"  📸 Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# FAMILY A  –  Manual Augmentation
#   x-axis: coverage fraction (0.20 … 1.00)
#   hue:    n_uniform  (1, 10, 25, 50)
# ─────────────────────────────────────────────────────────────
print("\n─── Family A: Manual Augmentation ───")
df_A = load_csv("grid_search_dataset_40grid_A.csv")

if df_A is not None:
    fig, axes = make_fig(
        "Family A – Manual Augmentation\n"
        "x: Fraction of Unvisited (s,a) Pairs Covered  |  hue: Samples per Pair",
        nrows=2, ncols=2
    )
    for ax, (col, ylabel, _, invert) in zip(axes, METRICS):
        if col not in df_A.columns:
            ax.set_visible(False)
            continue
        plot_line_with_optional_hue(
            ax, df_A,
            x_col    = "coverage",
            metric_col = col,
            hue_col  = "n_uniform",
            invert   = invert,
            xlabel   = "Coverage Fraction",
            ylabel   = ylabel,
            title    = ylabel,
        )
    save_or_show(fig, "plot_A_coverage_vs_metrics")

    # Secondary plot: x-axis = n_uniform, hue = coverage
    fig2, axes2 = make_fig(
        "Family A – Manual Augmentation\n"
        "x: Samples per Pair  |  hue: Coverage Fraction",
        nrows=2, ncols=2
    )
    for ax, (col, ylabel, _, invert) in zip(axes2, METRICS):
        if col not in df_A.columns:
            ax.set_visible(False)
            continue
        plot_line_with_optional_hue(
            ax, df_A,
            x_col    = "n_uniform",
            metric_col = col,
            hue_col  = "coverage",
            invert   = invert,
            xlabel   = "Samples per Unvisited (s,a) Pair",
            ylabel   = ylabel,
            title    = ylabel,
        )
    save_or_show(fig2, "plot_A_nsamples_vs_metrics")

    # Convenience: total_aug_rows as x-axis
    fig3, axes3 = make_fig(
        "Family A – Manual Augmentation\n"
        "x: Total Augmentation Rows Added",
        nrows=2, ncols=2
    )
    for ax, (col, ylabel, _, invert) in zip(axes3, METRICS):
        if col not in df_A.columns:
            ax.set_visible(False)
            continue
        plot_line_with_optional_hue(
            ax, df_A,
            x_col    = "total_aug_rows",
            metric_col = col,
            hue_col  = None,
            invert   = invert,
            xlabel   = "Total Augmentation Rows",
            ylabel   = ylabel,
            title    = ylabel,
        )
    save_or_show(fig3, "plot_A_totalrows_vs_metrics")


# ─────────────────────────────────────────────────────────────
# FAMILY B  –  Epsilon Variation
#   x-axis: epsilon  (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
# ─────────────────────────────────────────────────────────────
print("\n─── Family B: Epsilon Variation ───")
df_B = load_csv("grid_search_dataset_40grid_B.csv")

if df_B is not None:
    # Primary plot: x=epsilon, hue=proportions config
    fig, axes = make_fig(
        "Family B – Epsilon Variation\n"
        "x: Epsilon (same for both policies)  |  hue: Proportion Config [opt/alt/rand]",
        nrows=2, ncols=2
    )
    hue_col_B = "proportions" if "proportions" in df_B.columns else None
    for ax, (col, ylabel, _, invert) in zip(axes, METRICS):
        if col not in df_B.columns:
            ax.set_visible(False)
            continue
        plot_line_with_optional_hue(
            ax, df_B,
            x_col      = "epsilon",
            metric_col = col,
            hue_col    = hue_col_B,
            invert     = invert,
            xlabel     = "Epsilon (exploration rate)",
            ylabel     = ylabel,
            title      = ylabel,
        )
    save_or_show(fig, "plot_B_epsilon_vs_metrics")

    # Secondary plot: x=proportions, hue=epsilon — shows impact of data mix
    if hue_col_B is not None:
        fig2, axes2 = make_fig(
            "Family B – Proportion Config Effect\n"
            "x: Proportion Config [opt/alt/rand]  |  hue: Epsilon",
            nrows=2, ncols=2
        )
        for ax, (col, ylabel, _, invert) in zip(axes2, METRICS):
            if col not in df_B.columns:
                ax.set_visible(False)
                continue
            hue_vals = sorted(df_B["epsilon"].unique())
            for i, hv in enumerate(hue_vals):
                sub = df_B[df_B["epsilon"] == hv].sort_values("proportions")
                ax.plot(sub["proportions"], sub[col],
                        marker=MARKERS[i % len(MARKERS)],
                        color=COLORS[i % len(COLORS)],
                        linewidth=2, label=f"ε={hv}")
            ax.set_title(ylabel)
            ax.set_xlabel("Proportion Config")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            if invert:
                ax.invert_yaxis()
        save_or_show(fig2, "plot_B_props_vs_metrics")


# ─────────────────────────────────────────────────────────────
# FAMILY C  –  Random-Start Policy Coverage
#   x-axis: p_rand  (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
# ─────────────────────────────────────────────────────────────
print("\n─── Family C: Random-Start Policy Coverage ───")
df_C = load_csv("grid_search_dataset_40grid_C.csv")

if df_C is not None:
    fig, axes = make_fig(
        "Family C – Random-Start Policy Coverage\n"
        "x: Fraction of Dataset from Uniform-Random-Start Policy  |  100 k total steps",
        nrows=2, ncols=2
    )
    for ax, (col, ylabel, _, invert) in zip(axes, METRICS):
        if col not in df_C.columns:
            ax.set_visible(False)
            continue
        plot_line_with_optional_hue(
            ax, df_C,
            x_col    = "p_rand",
            metric_col = col,
            hue_col  = None,
            invert   = invert,
            xlabel   = "Proportion of Random-Start Policy (p_rand)",
            ylabel   = ylabel,
            title    = ylabel,
        )
    save_or_show(fig, "plot_C_prand_vs_metrics")


# ─────────────────────────────────────────────────────────────
# Combined summary: one convergence overview per family
# ─────────────────────────────────────────────────────────────
print("\n─── Combined Convergence Overview ───")

available = []
if df_A is not None and "convergence" in df_A.columns:
    available.append(("A – Augmentation\n(x: coverage, hue: n_uniform)",
                       df_A, "coverage", "n_uniform"))
if df_B is not None and "convergence" in df_B.columns:
    available.append(("B – Epsilon Variation\n(x: epsilon)",
                       df_B, "epsilon", None))
if df_C is not None and "convergence" in df_C.columns:
    available.append(("C – Random-Start Coverage\n(x: p_rand)",
                       df_C, "p_rand", None))

if available:
    fig, axes = plt.subplots(1, len(available),
                             figsize=(6 * len(available), 5), squeeze=False)
    fig.suptitle("Convergence Rate – All Families", fontsize=15, fontweight="bold")
    for ax, (title, df, x_col, hue) in zip(axes.flatten(), available):
        plot_line_with_optional_hue(
            ax, df,
            x_col      = x_col,
            metric_col = "convergence",
            hue_col    = hue,
            invert     = False,
            xlabel     = x_col,
            ylabel     = "Convergence Rate",
            title      = title,
        )
        ax.set_ylim(-0.05, 1.10)
    save_or_show(fig, "plot_combined_convergence")

print("\n✅ Plotting complete.")
