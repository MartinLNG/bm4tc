#!/usr/bin/env python
"""
Plot alpha-curve figures for each analyzed alpha_curve sweep.

For each dataset produces alpha_curve.png with two subplots:
  Left  — Accuracy vs alpha: clean acc, robust acc (no purif), robust acc (with purif)
  Right — Loss vs alpha: cls NLL (left y-axis) and gen NLL (right y-axis, independent scale)

All lines show mean ± 1 std across seeds.

Usage
-----
    python analysis/visualize/alpha_curve_plots.py                          # all datasets
    python analysis/visualize/alpha_curve_plots.py --filter-dataset ecg     # substring match
    python analysis/visualize/alpha_curve_plots.py <sweep_dir>              # single sweep dir
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT         = Path(__file__).parent.parent.parent
OUTPUTS_ROOT = ROOT / "analysis" / "outputs" / "alpha_curve"

ALPHA_COL   = "config/trainer.generative.criterion.kwargs.alpha"
ROB_COL     = "eval/test/rob/0.2"       # 10% of legendre range (2.0)
PURIF_COL   = "eval/uq_purify_acc/0.2/0.2"
ACC_COL     = "eval/test/acc"
CLSLOSS_COL = "eval/test/clsloss"
GENLOSS_COL = "eval/test/genloss"

DPI     = 150
FIGSIZE = (11, 4.5)

_ALPHA_TICKS  = [0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.5, 0.8, 1.0]
_ALPHA_LABELS = ['0', '1e-5', '1e-4', '1e-3', '1e-2', '5e-2', '0.1', '0.5', '0.8', '1']
_SYMLOG_THRESH = 5e-6


def _apply_alpha_xaxis(ax):
    ax.set_xscale('symlog', linthresh=_SYMLOG_THRESH)
    ax.set_xlim(0, 1)
    ax.set_xticks(_ALPHA_TICKS)
    ax.set_xticklabels(_ALPHA_LABELS, rotation=45, ha='right', fontsize=7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def agg_by_alpha(df: pd.DataFrame, col: str):
    """Return (alpha_vals, means, stds) sorted by alpha, or None if col missing."""
    if col not in df.columns:
        return None
    g = (
        df.groupby(ALPHA_COL)[col]
        .agg(["mean", "std"])
        .sort_index()
    )
    return g.index.values, g["mean"].values, g["std"].fillna(0).values


def plot_line(ax, result, label, color, linestyle="-"):
    """Plot mean line + ±1 std band. result is (alphas, means, stds) or None."""
    if result is None:
        return
    alphas, means, stds = result
    ax.plot(alphas, means, color=color, linestyle=linestyle, linewidth=1.8, label=label)
    ax.fill_between(alphas, means - stds, means + stds, color=color, alpha=0.15)


def get_dataset_name(csv_path: Path) -> str:
    """Strip date suffix from parent dir name: circles_4k_1404 → circles_4k."""
    return re.sub(r"_\d{4}$", "", csv_path.parent.name)


# ---------------------------------------------------------------------------
# Per-dataset figure
# ---------------------------------------------------------------------------

def plot_alpha_curve(csv_path: Path):
    df = pd.read_csv(csv_path)
    output_dir = csv_path.parent
    dataset = get_dataset_name(csv_path)

    if ALPHA_COL not in df.columns:
        print(f"  Skipping {dataset}: '{ALPHA_COL}' column not found.")
        return

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.suptitle(dataset, fontsize=12)

    # ------------------------------------------------------------------
    # Left: accuracy
    # ------------------------------------------------------------------
    plot_line(ax_l, agg_by_alpha(df, ACC_COL),   "Clean acc",       "steelblue",  "-")
    plot_line(ax_l, agg_by_alpha(df, ROB_COL),   "Rob (no purif)",  "darkorange", "--")
    plot_line(ax_l, agg_by_alpha(df, PURIF_COL), "Rob (purif r=0.2)", "seagreen", "-.")

    ax_l.set_xlabel("α  (0 = cls,  1 = gen)")
    ax_l.set_ylabel("Accuracy")
    _apply_alpha_xaxis(ax_l)
    ax_l.set_ylim(0, 1)
    ax_l.legend(fontsize=8, loc="best")
    ax_l.grid(True, alpha=0.3)
    ax_l.set_title("Accuracy vs α  (ε = 0.2)")

    # ------------------------------------------------------------------
    # Right: losses (dual y-axes)
    # ------------------------------------------------------------------
    ax_r2 = ax_r.twinx()

    cls_result = agg_by_alpha(df, CLSLOSS_COL)
    gen_result = agg_by_alpha(df, GENLOSS_COL)

    if cls_result is None:
        print(f"  Warning [{dataset}]: '{CLSLOSS_COL}' missing — rerun analysis with COMPUTE_CLS_LOSS=True.")
    if gen_result is None:
        print(f"  Warning [{dataset}]: '{GENLOSS_COL}' missing — rerun analysis with COMPUTE_GEN_LOSS=True.")

    plot_line(ax_r,  cls_result, "Cls NLL", "darkred",    "-")
    plot_line(ax_r2, gen_result, "Gen NLL", "steelblue",  "-")

    ax_r.set_xlabel("α  (0 = cls,  1 = gen)")
    ax_r.set_ylabel("Cls NLL", color="darkred")
    ax_r2.set_ylabel("Gen NLL", color="steelblue")
    ax_r.tick_params(axis="y", labelcolor="darkred")
    ax_r2.tick_params(axis="y", labelcolor="steelblue")
    _apply_alpha_xaxis(ax_r)
    ax_r.grid(True, alpha=0.3)
    ax_r.set_title("NLL Loss vs α")

    # Combined legend for twin axes
    lines  = ax_r.get_lines()  + ax_r2.get_lines()
    labels = [l.get_label() for l in lines]
    if lines:
        ax_r.legend(lines, labels, fontsize=8, loc="best")

    fig.tight_layout()
    out = output_dir / "alpha_curve.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_csvs(root: Path):
    return sorted(root.rglob("evaluation_data.csv"))


def get_dataset_base(p: Path) -> str:
    return re.sub(r"_\d{4}$", "", p.parent.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot alpha-curve figures for all analyzed alpha_curve sweeps."
    )
    parser.add_argument("sweep_dir", nargs="?", default=None,
                        help="Path to a single sweep dir (optional; default: all).")
    parser.add_argument("--filter-dataset", metavar="DS",
                        help="Only process CSVs whose dataset name contains DS.")
    args = parser.parse_args()

    if args.sweep_dir:
        root = Path(args.sweep_dir)
        if not root.is_absolute():
            root = ROOT / root
        csvs = sorted(root.rglob("evaluation_data.csv"))
        if not csvs:
            # Maybe they passed the dir directly
            candidate = root / "evaluation_data.csv"
            if candidate.exists():
                csvs = [candidate]
    else:
        if not OUTPUTS_ROOT.exists():
            print(f"No alpha_curve outputs found under {OUTPUTS_ROOT.relative_to(ROOT)}.")
            sys.exit(1)
        csvs = find_csvs(OUTPUTS_ROOT)

    if args.filter_dataset:
        csvs = [c for c in csvs if args.filter_dataset in get_dataset_base(c)]

    if not csvs:
        print("No evaluation_data.csv files found.")
        sys.exit(1)

    print(f"Plotting {len(csvs)} dataset(s):\n")
    for csv_path in csvs:
        print(f"[{get_dataset_base(csv_path)}]")
        plot_alpha_curve(csv_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
