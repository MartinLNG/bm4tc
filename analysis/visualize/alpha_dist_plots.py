#!/usr/bin/env python
"""
Plot how p(c|x) and p(x) evolve across alpha for each alpha_curve sweep.

For each dataset produces three figures saved alongside evaluation_data.csv:
  alpha_dist_combined.png  — 2 rows (p(c|x) top, p(x) bottom) × N alpha columns
  alpha_dist_pcx.png       — 1 row: p(c|x) vs alpha
  alpha_dist_px.png        — 1 row: p(x) vs alpha

For each alpha value the best-accuracy run (by eval/test/acc) is used.

Usage
-----
    python analysis/visualize/alpha_dist_plots.py                       # all datasets
    python analysis/visualize/alpha_dist_plots.py <sweep_dir>           # single sweep dir
    python analysis/visualize/alpha_dist_plots.py --max 5               # cap at 5 columns
    python analysis/visualize/alpha_dist_plots.py --no-data             # no data overlay
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from analysis.utils import load_run_config, find_model_checkpoint
from src.models import BornMachine  # must precede src.data to avoid circular import
from src.data.handler import DataHandler
from analysis.visualize.distributions import (
    make_grid,
    compute_conditional_probs,
    compute_joint_probs,
    _overlay_data,
    _save_fig,
    _infer_regime,
    DECISION_BOUNDARY_CMAP,
)

OUTPUTS_ROOT = PROJECT_ROOT / "analysis" / "outputs" / "alpha_curve"
ALPHA_COL    = "config/trainer.generative.criterion.kwargs.alpha"
ACC_COL      = "eval/test/acc"
DPI          = 150
PANEL_SIZE   = 3.0  # inches per panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample_alphas(sorted_alphas: list, max_count: int) -> list:
    """Return ≤ max_count alpha values, always including the 0 and 1 endpoints."""
    eps = 1e-6
    endpoints = [a for a in sorted_alphas if abs(a) < eps or abs(a - 1.0) < eps]
    middle    = [a for a in sorted_alphas if abs(a) >= eps and abs(a - 1.0) >= eps]

    n_middle = max_count - len(endpoints)
    if n_middle <= 0 or len(middle) <= n_middle:
        return sorted(endpoints + middle)

    indices = np.round(np.linspace(0, len(middle) - 1, n_middle)).astype(int)
    return sorted(endpoints + [middle[i] for i in sorted(set(indices))])


def _best_run_per_alpha(df: pd.DataFrame) -> dict:
    """Return alpha → run_path for the highest eval/test/acc run at each alpha."""
    result = {}
    for alpha, group in df.groupby(ALPHA_COL):
        if ACC_COL in group.columns and group[ACC_COL].notna().any():
            best_idx = group[ACC_COL].idxmax()
        else:
            best_idx = group.index[0]
        result[float(alpha)] = group.loc[best_idx, "run_path"]
    return result


def _load_bm(run_path: str, device: torch.device):
    run_dir = Path(run_path)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    cfg  = load_run_config(run_dir)
    ckpt = find_model_checkpoint(run_dir)
    bm   = BornMachine.load(str(ckpt))
    bm.to(device)
    regime     = _infer_regime(bm, ckpt)
    sync_after = "classification" if regime == "discriminative" else "generation"
    bm.sync_tensors(after=sync_after)
    return bm, cfg


# ---------------------------------------------------------------------------
# Per-axis rendering
# ---------------------------------------------------------------------------

def _render_decision_boundary(ax, conditional, grid_x1, grid_x2, input_range,
                              train_data=None, train_labels=None, num_classes=None,
                              eps=0.05):
    lo, hi  = input_range
    res     = grid_x1.shape[0]
    prob1   = conditional.numpy()[:, 1].reshape(res, res)
    display = prob1.copy().astype(float)
    display[(prob1 >= 0.5 - eps) & (prob1 <= 0.5 + eps)] = np.nan
    ax.set_facecolor("white")
    ax.pcolormesh(grid_x1, grid_x2, display, cmap=DECISION_BOUNDARY_CMAP,
                  shading="auto", vmin=0.0, vmax=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if train_data is not None and num_classes is not None:
        _overlay_data(ax, train_data, train_labels, num_classes)


def _render_px(ax, joint, grid_x1, grid_x2, input_range,
               train_data=None, train_labels=None, num_classes=None):
    lo, hi    = input_range
    res       = grid_x1.shape[0]
    joint_np  = joint.numpy()
    class_max = joint_np.max(axis=0)
    class_max = np.where(class_max == 0, 1.0, class_max)
    marginal  = (joint_np / class_max).sum(axis=1).reshape(res, res)
    vmin      = float(np.percentile(marginal, 2))
    vmax      = float(np.percentile(marginal, 98))
    ax.pcolormesh(grid_x1, grid_x2, marginal, cmap="Purples",
                  shading="auto", vmin=vmin, vmax=vmax)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if train_data is not None and num_classes is not None:
        _overlay_data(ax, train_data, train_labels, num_classes)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def _assemble_figures(panels, grid_x1, grid_x2, input_range, num_classes):
    """
    panels: list of (alpha, conditional, joint, train_data, train_labels, show_data)
    Returns (fig_combined, fig_pcx, fig_px).
    """
    n = len(panels)
    w = PANEL_SIZE * n
    h = PANEL_SIZE

    fig_comb, ax_comb = plt.subplots(2, n, figsize=(w, 2 * h), squeeze=False)
    fig_pcx,  ax_pcx  = plt.subplots(1, n, figsize=(w,     h), squeeze=False)
    fig_px,   ax_px   = plt.subplots(1, n, figsize=(w,     h), squeeze=False)

    for col, (alpha, conditional, joint, train_data, train_labels, show_data) in enumerate(panels):
        td = train_data   if show_data else None
        tl = train_labels if show_data else None
        nc = num_classes  if show_data else None

        _render_decision_boundary(ax_comb[0, col], conditional, grid_x1, grid_x2, input_range, td, tl, nc)
        _render_px(               ax_comb[1, col], joint,       grid_x1, grid_x2, input_range, td, tl, nc)
        _render_decision_boundary(ax_pcx[0, col],  conditional, grid_x1, grid_x2, input_range, td, tl, nc)
        _render_px(               ax_px[0, col],   joint,       grid_x1, grid_x2, input_range, td, tl, nc)

        title = f"α = {alpha:.2g}"
        ax_comb[0, col].set_title(title, fontsize=14)
        ax_pcx[0,  col].set_title(title, fontsize=14)
        ax_px[0,   col].set_title(title, fontsize=14)

    for fig in (fig_comb, fig_pcx, fig_px):
        fig.tight_layout()

    return fig_comb, fig_pcx, fig_px


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def plot_alpha_dists(csv_path: Path, max_alpha: int, resolution: int,
                     show_data: bool, device_str: str):
    df = pd.read_csv(csv_path)
    output_dir = csv_path.parent

    if ALPHA_COL not in df.columns:
        print(f"  Skipping: '{ALPHA_COL}' column not found.")
        return
    if "run_path" not in df.columns:
        print(f"  Skipping: 'run_path' column not found.")
        return

    all_alphas = sorted(float(a) for a in df[ALPHA_COL].dropna().unique())
    selected   = _subsample_alphas(all_alphas, max_alpha)
    best_runs  = _best_run_per_alpha(df)
    device     = torch.device(device_str)

    panels: list = []
    grid_x1 = grid_x2 = grid_points = None
    input_range = num_classes = None
    shared_train_data = shared_train_labels = None

    for alpha in selected:
        matched = min(best_runs.keys(), key=lambda k: abs(k - alpha))
        if abs(matched - alpha) > 1e-4:
            print(f"  Warning: no run found for alpha≈{alpha:.3g}, skipping.")
            continue

        run_path = best_runs[matched]
        try:
            bm, cfg = _load_bm(run_path, device)
        except Exception as exc:
            print(f"  Warning: could not load model for alpha={alpha:.3g}: {exc}")
            continue

        if grid_x1 is None:
            if bm.num_sites != 3:
                print(f"  Skipping: dataset has {bm.num_sites-1} input features "
                      f"(distribution plots are only supported for 2D toy data).")
                del bm
                return
            input_range                 = bm.input_range
            grid_x1, grid_x2, grid_points = make_grid(input_range, resolution)
            num_classes                 = bm.out_dim

        if show_data and shared_train_data is None:
            dh = DataHandler(cfg.dataset)
            dh.load()
            dh.split_and_rescale(bm)
            shared_train_data   = dh.data["train"]
            shared_train_labels = dh.labels["train"]

        print(f"    α={alpha:.2g}: computing p(c|x) and p(x)…")
        conditional = compute_conditional_probs(bm, grid_points, device)
        joint       = compute_joint_probs(bm, grid_points, device, normalize=True)
        panels.append((matched, conditional, joint,
                        shared_train_data, shared_train_labels, show_data))
        del bm

    if not panels:
        print("  No panels produced; skipping.")
        return

    fig_comb, fig_pcx, fig_px = _assemble_figures(
        panels, grid_x1, grid_x2, input_range, num_classes
    )

    _save_fig(fig_comb, output_dir / "alpha_dist_combined.png")
    _save_fig(fig_pcx,  output_dir / "alpha_dist_pcx.png")
    _save_fig(fig_px,   output_dir / "alpha_dist_px.png")
    plt.close("all")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _dataset_base(p: Path) -> str:
    return re.sub(r"_\d{4}$", "", p.parent.name)


def find_csvs(root: Path):
    return sorted(root.rglob("evaluation_data.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot p(c|x) and p(x) distributions across alpha for alpha_curve sweeps."
    )
    parser.add_argument("sweep_dir", nargs="?", default=None,
                        help="Path to a single sweep dir (default: all under analysis/outputs/alpha_curve/).")
    parser.add_argument("--max", type=int, default=8, metavar="N",
                        help="Maximum number of alpha columns (≥ 3; α=0 and α=1 always included). Default: 8.")
    parser.add_argument("--no-data", action="store_true",
                        help="Suppress training-data scatter overlay.")
    parser.add_argument("--resolution", type=int, default=150,
                        help="Grid resolution per axis (default: 150).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", metavar="DS",
                        help="Only process sweeps whose dataset name contains DS.")
    args = parser.parse_args()

    if args.max < 3:
        parser.error("--max must be ≥ 3 (α=0, at least one middle value, α=1).")

    if args.sweep_dir:
        root = Path(args.sweep_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        csvs = sorted(root.rglob("evaluation_data.csv"))
        if not csvs and (root / "evaluation_data.csv").exists():
            csvs = [root / "evaluation_data.csv"]
    else:
        if not OUTPUTS_ROOT.exists():
            print(f"No alpha_curve outputs found under {OUTPUTS_ROOT.relative_to(PROJECT_ROOT)}.")
            sys.exit(1)
        csvs = find_csvs(OUTPUTS_ROOT)

    if args.dataset:
        csvs = [c for c in csvs if args.dataset in _dataset_base(c)]

    if not csvs:
        print("No evaluation_data.csv files found.")
        sys.exit(1)

    print(f"Processing {len(csvs)} dataset(s):\n")
    for csv_path in csvs:
        print(f"[{_dataset_base(csv_path)}]")
        plot_alpha_dists(
            csv_path,
            max_alpha=args.max,
            resolution=args.resolution,
            show_data=not args.no_data,
            device_str=args.device,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
