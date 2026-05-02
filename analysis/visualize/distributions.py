# %% [markdown]
# # Visualize Learned Probability Distributions
#
# This notebook visualizes the learned p(c|x) and p(x,c) distributions
# of a trained BornMachine over the 2D input space.
#
# **What it shows:**
# - p(c=1|x): conditional probability heatmap
# - p(x): marginal probability heatmap
# - Optional training data overlay for verification
#
# **Usage:**
# - Set `RUN_DIR` to your Hydra output directory
# - Run cells interactively (VS Code) or as a script
# - CLI: python -m analysis.visualize.distributions --run <run_dir> --save-dir <dir>

# %% [markdown]
# ## Setup and Configuration

# %%
import sys
from pathlib import Path

# Handle both script and interactive execution
if "__file__" in dir():
    project_root = Path(__file__).parent.parent.parent
else:
    # Interactive/notebook mode - assume we're in analysis/visualize/
    project_root = Path.cwd().parent.parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import logging

_C0 = "#1F77B4"   # matplotlib default blue  (class 0)
_C1 = "#FF7F0E"   # matplotlib default orange (class 1)

DECISION_BOUNDARY_CMAP = LinearSegmentedColormap.from_list(
    "bw_orange", [_C0, "white", _C1]
)

def _cls_cmap(class_idx: int):
    """White → class-color sequential colormap."""
    palette = [_C0, _C1]
    return LinearSegmentedColormap.from_list(
        f"cls{class_idx}", ["white", palette[class_idx % 2]]
    )
from analysis.utils import load_run_config, find_model_checkpoint
from src.models import BornMachine  # must import before src.data to avoid circular import
from src.data.handler import DataHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filename stem → "discriminative" | "generative" (backward compat for old checkpoints)
_CHECKPOINT_REGIME = {
    "cls": "discriminative", "adv": "discriminative",
    "gen": "generative",     "gan": "generative",
}


def _infer_regime(bm: "BornMachine", checkpoint_path: Path) -> str:
    """Return 'discriminative' or 'generative', from checkpoint metadata or filename."""
    if bm._last_regime is not None:
        return bm._last_regime
    stem = checkpoint_path.stem
    regime = _CHECKPOINT_REGIME.get(stem)
    if regime is None:
        logger.warning(
            f"Cannot infer regime from checkpoint name '{stem}'; "
            f"defaulting to 'discriminative'."
        )
        regime = "discriminative"
    else:
        logger.info(f"Inferred regime '{regime}' from checkpoint name '{stem}'.")
    return regime

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION FOR YOUR EXPERIMENT
# =============================================================================

# Path to run directory (contains .hydra/config.yaml and models/)
RUN_DIR = "outputs/adv_seed_sweep_moons_4k_12Feb26/3"  # Change to your run directory

# Grid resolution for heatmaps (resolution x resolution points)
RESOLUTION = 150

# Whether to normalize p(x,c) by the partition function
NORMALIZE_JOINT = True

# Whether to overlay training data points on plots
SHOW_DATA = True

# Device for computation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directory for saved figures
SAVE_DIR = "analysis/outputs/distributions/"

# %% [markdown]
# ## Utility Functions

# %%
def make_grid(input_range, resolution):
    """Create a 2D meshgrid over [lo, hi]^2.

    Args:
        input_range: Tuple (lo, hi) defining the input space bounds.
        resolution: Number of points along each axis.

    Returns:
        grid_x1: (resolution, resolution) array of x1 coordinates.
        grid_x2: (resolution, resolution) array of x2 coordinates.
        grid_points: (resolution^2, 2) tensor of all grid points.
    """
    lo, hi = input_range
    x1 = np.linspace(lo, hi, resolution)
    x2 = np.linspace(lo, hi, resolution)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    grid_points = torch.tensor(
        np.column_stack([grid_x1.ravel(), grid_x2.ravel()]),
        dtype=torch.float32,
    )
    return grid_x1, grid_x2, grid_points


def compute_conditional_probs(bm, grid_points, device, batch_size=10000):
    """Compute p(c|x) over grid points using batched inference.

    Args:
        bm: BornMachine instance (already on device).
        grid_points: (N, 2) tensor of input points.
        device: Torch device.
        batch_size: Number of points per batch.

    Returns:
        (N, num_classes) tensor of conditional class probabilities.
    """
    all_probs = []
    n = grid_points.shape[0]
    for start in range(0, n, batch_size):
        batch = grid_points[start:start + batch_size].to(device)
        with torch.no_grad():
            probs = bm.class_probabilities(batch)
        all_probs.append(probs.cpu())
        bm.reset()
    return torch.cat(all_probs, dim=0)


def compute_joint_probs(bm: BornMachine, grid_points, device, normalize=True, batch_size=10000):
    """Compute p(x,c) = |psi(x,c)|^2 [/ Z] over grid points.

    For each class c, computes the unnormalized joint probability using the
    generator's sequential contraction. Optionally normalizes by the partition
    function.

    Args:
        bm: BornMachine instance (already on device).
        grid_points: (N, 2) tensor of input points.
        device: Torch device.
        normalize: If True, divide by exp(log_partition_function()).
        batch_size: Number of points per batch.

    Returns:
        (N, num_classes) tensor of (normalized) joint probabilities.
    """
    n = grid_points.shape[0]
    num_classes = bm.out_dim

    # Compute partition function once if normalizing
    if normalize:
        bm.generator.reset()
        with torch.no_grad():
            log_Z = bm.generator.log_partition_function()
        Z = torch.exp(log_Z).item()
        logger.info(f"Partition function Z = {Z:.6f} (log Z = {log_Z:.4f})")
    else:
        Z = 1.0

    all_probs = []
    for c in range(num_classes):
        class_probs = []
        for start in range(0, n, batch_size):
            batch = grid_points[start:start + batch_size].to(device)
            labels = torch.full((batch.shape[0],), c, dtype=torch.long, device=device)
            bm.generator.reset()
            with torch.no_grad():
                prob = bm.generator.unnormalized_prob(batch, labels)
            class_probs.append(prob.cpu())
        all_probs.append(torch.cat(class_probs, dim=0))
        logger.info(f"Computed joint probabilities for class {c}")

    # Stack: (N, num_classes)
    joint = torch.stack(all_probs, dim=1)
    if normalize:
        joint = joint / Z
    return joint


def _overlay_data(ax, data, labels, num_classes):
    """Scatter overlay using matplotlib default class colors (C0=blue, C1=orange)."""
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    for c in range(num_classes):
        mask = labels == c
        ax.scatter(data[mask, 0], data[mask, 1],
                   s=8, alpha=0.75, color=palette[c],
                   linewidths=0, marker=markers[c % len(markers)], zorder=5)


def _save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {path}")


def plot_decision_boundary(
    conditional, grid_x1, grid_x2,
    input_range=(0.0, 1.0),
    train_data=None, train_labels=None, num_classes=None,
    eps=0.05, save_path=None,
) -> plt.Figure:
    """Decision boundary: blue→white→orange diverging map, white band at 0.5 ± eps."""
    fig, ax = plt.subplots(figsize=(4, 4))
    lo, hi = input_range
    res = grid_x1.shape[0]
    prob1 = conditional.numpy()[:, 1].reshape(res, res)

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
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_class_conditional(
    conditional, grid_x1, grid_x2,
    input_range=(0.0, 1.0),
    train_data=None, train_labels=None, num_classes=None,
    class_idx=1, cmap=None, save_path=None,
) -> plt.Figure:
    """Square figure: p(c=class_idx|x) heatmap, white→class-color, no colorbar."""
    if cmap is None:
        cmap = _cls_cmap(class_idx)
    fig, ax = plt.subplots(figsize=(4, 4))
    lo, hi = input_range
    res = grid_x1.shape[0]
    prob = conditional.numpy()[:, class_idx].reshape(res, res)
    ax.pcolormesh(grid_x1, grid_x2, prob, cmap=cmap,
                  shading="auto", vmin=0.0, vmax=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if train_data is not None and num_classes is not None:
        _overlay_data(ax, train_data, train_labels, num_classes)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_joint_marginal(
    joint, grid_x1, grid_x2,
    input_range=(0.0, 1.0),
    train_data=None, train_labels=None, num_classes=None,
    cmap="Purples", save_path=None,
) -> plt.Figure:
    """Square figure: class-normalized joint Σ_c p(x,c)/max_x p(x,c), no colorbar.

    Each class is rescaled so its peak contributes equally, preventing a
    high-probability class from washing out the spatial structure of rarer classes.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    lo, hi = input_range
    res = grid_x1.shape[0]
    joint_np = joint.numpy()                               # (N, num_classes)
    class_max = joint_np.max(axis=0)                       # (num_classes,)
    class_max = np.where(class_max == 0, 1.0, class_max)  # guard div-by-zero
    marginal = (joint_np / class_max).sum(axis=1).reshape(res, res)
    vmin = float(np.percentile(marginal, 2))
    vmax = float(np.percentile(marginal, 98))
    pcm = ax.pcolormesh(grid_x1, grid_x2, marginal, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if train_data is not None and num_classes is not None:
        _overlay_data(ax, train_data, train_labels, num_classes)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def visualize_from_run_dir(
    run_dir,
    resolution=150,
    normalize_joint=True,
    show_data=True,
    device="cpu",
    save_dir=None,
    boundary_eps=0.05,
):
    """High-level convenience: load model + data and produce distribution plots.

    Args:
        run_dir: Path to the Hydra output directory.
        resolution: Grid resolution for heatmaps.
        normalize_joint: Whether to normalize p(x,c) by the partition function.
        show_data: Whether to overlay training data points.
        device: Torch device string.
        save_dir: Directory to save figures. If None, does not save.

    Returns:
        Tuple of (fig_cls, fig_jnt) Matplotlib Figure objects.
    """
    device = torch.device(device)
    run_dir = Path(run_dir)

    # Load config and model
    cfg = load_run_config(run_dir)
    checkpoint_path = find_model_checkpoint(run_dir)
    bm = BornMachine.load(str(checkpoint_path))
    bm.to(device)
    logger.info(f"Loaded model from {checkpoint_path}")

    # Ensure classifier and generator are in sync before computing joint probs.
    # For discriminatively trained models the classifier is canonical; for
    # generatively trained models the generator is canonical.
    regime = _infer_regime(bm, checkpoint_path)
    sync_after = "classification" if regime == "discriminative" else "generation"
    bm.sync_tensors(after=sync_after)
    logger.info(f"Synced tensors (regime='{regime}', after='{sync_after}')")

    # Load and prepare data
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bm)

    # Get training data for overlay
    train_data = datahandler.data["train"] if show_data else None
    train_labels = datahandler.labels["train"] if show_data else None
    num_classes = bm.out_dim if show_data else None

    # Build grid
    input_range = bm.input_range
    grid_x1, grid_x2, grid_points = make_grid(input_range, resolution)

    # Compute distributions
    logger.info("Computing conditional probabilities p(c|x)...")
    conditional = compute_conditional_probs(bm, grid_points, device)

    logger.info("Computing joint probabilities p(x,c)...")
    joint = compute_joint_probs(bm, grid_points, device, normalize=normalize_joint)

    # Determine save paths
    class_path = (Path(save_dir) / "best_class_dist.png") if save_dir else None
    joint_path = (Path(save_dir) / "best_joint.png") if save_dir else None
    db_path    = (Path(save_dir) / "decision_boundary.png") if save_dir else None

    fig_cls = plot_class_conditional(
        conditional, grid_x1, grid_x2,
        input_range=input_range,
        train_data=train_data, train_labels=train_labels, num_classes=num_classes,
        save_path=class_path,
    )
    fig_jnt = plot_joint_marginal(
        joint, grid_x1, grid_x2,
        input_range=input_range,
        save_path=joint_path,
    )
    fig_db = plot_decision_boundary(
        conditional, grid_x1, grid_x2,
        input_range=input_range,
        train_data=train_data, train_labels=train_labels, num_classes=num_classes,
        eps=boundary_eps, save_path=db_path,
    )

    return fig_cls, fig_jnt, fig_db


# %% [markdown]
# ## Load Model and Data

# %%
def load_model_and_data():
    """Load trained BornMachine and DataHandler from run directory.

    Returns:
        Tuple of (bornmachine, datahandler, device, cfg).
    """
    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    run_dir = Path(RUN_DIR)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    logger.info(f"Loading config from: {run_dir}")
    cfg = load_run_config(run_dir)

    checkpoint_path = find_model_checkpoint(run_dir)
    logger.info(f"Loading model from: {checkpoint_path}")
    bm = BornMachine.load(str(checkpoint_path))
    bm.to(device)

    regime = _infer_regime(bm, checkpoint_path)
    sync_after = "classification" if regime == "discriminative" else "generation"
    bm.sync_tensors(after=sync_after)
    logger.info(f"Synced tensors (regime='{regime}', after='{sync_after}')")

    # Reconstruct DataHandler
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bm)

    return bm, datahandler, device, cfg


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate distribution plots for a run directory.")
    parser.add_argument("--run", default=None, help="Path to Hydra run directory.")
    parser.add_argument("--save-dir", default=None, help="Directory to save output figures.")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--no-data", action="store_true")
    parser.add_argument("--device", default=DEVICE)
    cli_args = parser.parse_args()

    if cli_args.run is not None:
        fig_cls, fig_jnt, fig_db = visualize_from_run_dir(
            run_dir=cli_args.run,
            resolution=cli_args.resolution,
            normalize_joint=NORMALIZE_JOINT,
            show_data=not cli_args.no_data,
            device=cli_args.device,
            save_dir=cli_args.save_dir,
        )
        plt.show()
    else:
        print("=" * 60)
        print("Loading model and data...")
        print("=" * 60)

        bm, datahandler, device, cfg = load_model_and_data()

        print(f"\nDataset: {cfg.dataset.name}")
        print(f"Train samples: {len(datahandler.data['train'])}")
        print(f"Number of classes: {datahandler.num_cls}")
        print(f"Input range: {bm.input_range}")

        print("=" * 60)
        print("Computing distributions over input grid...")
        print("=" * 60)

        input_range = bm.input_range
        grid_x1, grid_x2, grid_points = make_grid(input_range, RESOLUTION)
        print(f"Grid: {RESOLUTION}x{RESOLUTION} = {grid_points.shape[0]} points")

        print("\nComputing p(c|x)...")
        conditional = compute_conditional_probs(bm, grid_points, device)
        print(f"  Shape: {conditional.shape}")

        print("\nComputing p(x,c)...")
        joint = compute_joint_probs(bm, grid_points, device, normalize=NORMALIZE_JOINT)
        print(f"  Shape: {joint.shape}")

        train_data = datahandler.data["train"] if SHOW_DATA else None
        train_labels = datahandler.labels["train"] if SHOW_DATA else None

        save_dir = Path(SAVE_DIR)
        if not save_dir.is_absolute():
            save_dir = project_root / save_dir

        num_classes = bm.out_dim if SHOW_DATA else None
        fig_cls = plot_class_conditional(
            conditional, grid_x1, grid_x2,
            input_range=input_range,
            train_data=train_data, train_labels=train_labels,
            num_classes=num_classes,
            save_path=save_dir / "best_class_dist.png",
        )
        fig_jnt = plot_joint_marginal(
            joint, grid_x1, grid_x2,
            input_range=input_range,
            save_path=save_dir / "best_joint.png",
        )
        fig_db = plot_decision_boundary(
            conditional, grid_x1, grid_x2,
            input_range=input_range,
            train_data=train_data, train_labels=train_labels,
            num_classes=num_classes,
            save_path=save_dir / "decision_boundary.png",
        )
        plt.show()

        print("\n" + "=" * 60)
        print("Distribution Visualization Complete")
        print("=" * 60)
        print(f"\nSaved to: {save_dir}")
