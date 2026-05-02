# %% [markdown]
# # Post-Hoc Seed Sweep Analysis
#
# This notebook **loads each saved model** from a seed sweep directory and
# **recomputes metrics post-hoc on the test set**, enabling:
# - Computing metrics that weren't computed during training
# - Ensuring data splits match each run's config (correct seeds)
# - Consistent evaluation settings across all runs (e.g., same attack strengths)
#
# **Scope:** Seed sweeps (and alpha_curve sweeps) where all runs share the same
# hyperparameters.  Best-model selection is not meaningful here — use a dedicated
# HPO analysis script for that.  All reported numbers are **test-set only**.
#
# **Sections:**
# 1. Configuration
# 2. Per-model evaluation
# 3. Statistics (correlation, Pareto, summary — data only; no intermediate image files)
# 4. Sanity check against W&B summary metrics
# 5. Learned distribution visualization for representative model (best_run_distributions.png, best_run_samples.png)
# 6. Summary export (sweep_analysis_summary.txt + evaluation_data.csv)

# %% [markdown]
# ## 1. Configuration

# %%
import sys
import argparse
from pathlib import Path

# Handle both script and interactive execution
if "__file__" in dir():
    project_root = Path(__file__).parent.parent
else:
    project_root = Path.cwd().parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

# Path to sweep directory (contains numbered sub-dirs with .hydra/config.yaml)
# Can be overridden from the command line:
#   python analysis/sweep_analysis.py outputs/seed_sweep/cls/fourier/d4D3/circles_4k_1802
SWEEP_DIR = "outputs/seed_sweep_uq_gen_d30D18fourier_moons_4k_1702"
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("sweep_dir", nargs="?", default=None)
_cli.add_argument("--no-viz", action="store_true", help="Skip best-run distribution plots.")
_cli.add_argument("--no-mia", action="store_true", help="Skip membership inference attack.")
_cli_args, _ = _cli.parse_known_args()
if _cli_args.sweep_dir is not None:
    SWEEP_DIR = _cli_args.sweep_dir

# Training regime: "pre", "gen", "adv", "gan".
# Auto-detected from SWEEP_DIR (which encodes the regime via the ${training_regime:} resolver).
# Override manually only if auto-detection gives the wrong result.
from analysis.utils.resolve import resolve_regime_from_path as _resolve_regime_from_path
REGIME = _resolve_regime_from_path(SWEEP_DIR)
if REGIME is None:
    print(
        f"WARNING: Could not auto-detect training regime from '{SWEEP_DIR}'.\n"
        "  Set REGIME manually to one of: 'pre', 'gen', 'adv', 'gan'."
    )
    REGIME = "pre"  # fallback — change if incorrect
else:
    print(f"Auto-detected training regime: '{REGIME}' (from sweep_dir)")

from analysis.utils.resolve import resolve_embedding_from_path as _resolve_embedding
from analysis.utils.resolve import embedding_range_size as _embedding_range_size
_EMBEDDING = _resolve_embedding(SWEEP_DIR)
if _EMBEDDING is None:
    print(f"WARNING: Could not detect embedding from '{SWEEP_DIR}'. Assuming Fourier (range size 1.0).")
_RANGE_SIZE = _embedding_range_size(_EMBEDDING)
print(f"Embedding: '{_EMBEDDING or 'unknown'}'  →  input range size: {_RANGE_SIZE}")

# --- METRIC TOGGLES ---
COMPUTE_ACC = True
COMPUTE_ROB = True
COMPUTE_MIA = False
COMPUTE_CLS_LOSS = True
COMPUTE_GEN_LOSS = True
COMPUTE_FID = False
COMPUTE_UQ = True  # Uncertainty quantification (detection + purification)
COMPUTE_GIBBS_PURIFICATION = True  # Gibbs-sampling purification (requires COMPUTE_UQ=True)
COMPUTE_JOINT_ATTACK = True  # Joint generative attack (JOINT_PGD) alongside standard PGD
COMPUTE_DISTRIBUTIONS = False  # Set False (or pass --no-viz) to skip best-run distribution plots

# --- EVASION CONFIG (single source of truth for all adversarial attacks) ---
# Applies to: robustness eval, UQ adversarial examples, adversarial MIA.
# Set to None to use each run's own evasion config.
# Strengths are expressed as FRACTIONS of the input range (max margin ~0.15 of range):
#   Fourier (range 1.0): same as absolute value
#   Legendre (range 2.0): multiply by 2.0
_STRENGTH_FRACTIONS = [0.05, 0.1, 0.15]
EVASION_CONFIG = {
    "method": "PGD",
    "norm": "inf",
    "num_steps": 40,
    "strengths": [s * _RANGE_SIZE for s in _STRENGTH_FRACTIONS],
}

# --- SAMPLING OVERRIDE ---
# Set to a dict to override sampling config, or None to use per-run config.
SAMPLING_OVERRIDE = None

# --- MIA SETTINGS ---
# Feature toggles: which confidence features to extract from p(c|x).
# Label-free features (max_prob, entropy, margin, modified_entropy) are always
# derived from the probability vector alone.  correct_prob and loss require a
# reference label — controlled by use_true_labels below.
MIA_FEATURES = {
    "max_prob": True,
    "entropy": True,
    "correct_prob": True,
    "loss": False,
    "margin": False,
    "modified_entropy": False,
    # True  = use ground-truth labels for correct_prob/loss (worst-case risk).
    # False = use predicted labels (argmax of probs) to avoid label leakage.
    "use_true_labels": True,
}

# --- MIA ADVERSARIAL SETTINGS ---
# Set MIA_ADV_STRENGTH to None to skip adversarial MIA entirely.
# Attack settings (method, norm, num_steps) are derived from EVASION_CONFIG.
MIA_ADV_STRENGTH = 0.10 * _RANGE_SIZE  # 10% of input range; already in EVASION_CONFIG["strengths"], deduped automatically.

# --- UQ SETTINGS (UQ-specific params only; attack settings from EVASION_CONFIG) ---
UQ_CONFIG = {
    "radii": [0.10 * _RANGE_SIZE],   # single radius: 10% of input range (was [0.15, 0.3])
    "percentiles": [1, 5, 10, 20],
}

# --- GIBBS PURIFICATION SETTINGS (only used when COMPUTE_GIBBS_PURIFICATION=True) ---
# Memory cost: (gibbs_batch_size × num_bins)² × 4 bytes.
#   bs=8,  bins=200 → ~10 MB density matrix
#   bs=32, bins=200 → ~160 MB density matrix
GIBBS_CONFIG = {
    "n_sweeps": [1, 3, 5],       # evaluate each sweep count independently
    "num_bins": 200,              # grid resolution over the input range
    "gibbs_batch_size": 8,        # samples processed per sequential() call
    "radius": 0.1,                # relative to input range (delta_abs = rel * (hi - lo)); None = unrestricted
}

# --- EVALUATION SETTINGS ---
# Evaluate on test set only — this script is for seed sweeps where all runs
# share the same hyperparameters, so best-model selection is not meaningful.
EVAL_SPLITS = ["test"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- STATISTICS SETTINGS ---
EFFECTIVE_N = None  # Override for stderr calc (None = use actual N)

# --- PLOT SETTINGS ---
FIGSIZE = (10, 6)
DPI = 100

# --- PARETO SETTINGS ---
# Robustness strength for Pareto frontier selection.
# Set to None to auto-select the weakest non-zero strength.
PARETO_ROB_STRENGTH = 0.10 * _RANGE_SIZE   # 10% of input range

# --- SANITY CHECK ---
# Map eval column -> W&B summary column for comparison.
# Set to None or {} to skip sanity check.
SANITY_CHECK_METRICS = {
    "eval/test/acc": "summary/adv/test/acc",
    "eval/valid/clsloss": "summary/adv/valid/clsloss",
}
SANITY_CHECK_TOL = 1e-4

# --- CONFIG KEYS TO EXTRACT ---
# Hydra config keys to include in the DataFrame alongside eval metrics.
CONFIG_KEYS = [
    "dataset.name",
    "tracking.seed",
    "dataset.gen_dow_kwargs.seed",
    "trainer.generative.criterion.kwargs.alpha",
]

# --- CLI overrides (applied after config block so they take effect) ---
if _cli_args.no_viz:
    COMPUTE_DISTRIBUTIONS = False
if _cli_args.no_mia:
    COMPUTE_MIA = False

# %% [markdown]
# ## 2. Per-Model Evaluation

# %%
# Add PARETO and MIA strengths to EVASION_CONFIG; sort. Build full UQ config.
if EVASION_CONFIG:
    _strengths = [float(s) for s in EVASION_CONFIG.get("strengths", [])]
    if COMPUTE_ROB and PARETO_ROB_STRENGTH is not None:
        if float(PARETO_ROB_STRENGTH) not in _strengths:
            _strengths.append(float(PARETO_ROB_STRENGTH))
    if COMPUTE_MIA and MIA_ADV_STRENGTH is not None:
        if float(MIA_ADV_STRENGTH) not in _strengths:
            _strengths.append(float(MIA_ADV_STRENGTH))
    EVASION_CONFIG["strengths"] = sorted(set(_strengths))
    print(f"Final attack strengths: {EVASION_CONFIG['strengths']}")
elif PARETO_ROB_STRENGTH is not None and COMPUTE_ROB:
    print(f"Note: EVASION_CONFIG is None; using per-run evasion configs. "
          f"Ensure each run includes eps={PARETO_ROB_STRENGTH}.")

_full_uq_config = None
if COMPUTE_UQ and UQ_CONFIG is not None and EVASION_CONFIG:
    _full_uq_config = {
        "norm":             EVASION_CONFIG.get("norm", "inf"),
        "num_steps":        EVASION_CONFIG.get("num_steps", 20),  # purification steps
        "attack_method":    EVASION_CONFIG.get("method", "PGD"),
        "attack_strengths": EVASION_CONFIG["strengths"],
        "attack_num_steps": EVASION_CONFIG.get("num_steps", 20),
        **UQ_CONFIG,  # radii, percentiles (may override num_steps if user adds it)
    }
    if COMPUTE_GIBBS_PURIFICATION:
        _full_uq_config["run_gibbs"] = True
        _full_uq_config["gibbs_n_sweeps"] = GIBBS_CONFIG["n_sweeps"]
        _full_uq_config["gibbs_num_bins"] = GIBBS_CONFIG["num_bins"]
        _full_uq_config["gibbs_batch_size"] = GIBBS_CONFIG["gibbs_batch_size"]
        _full_uq_config["gibbs_radius"] = GIBBS_CONFIG.get("radius")
elif COMPUTE_GIBBS_PURIFICATION and not COMPUTE_UQ:
    print("WARNING: COMPUTE_GIBBS_PURIFICATION=True requires COMPUTE_UQ=True; skipping Gibbs.")

_full_joint_uq_config = None
if COMPUTE_JOINT_ATTACK and COMPUTE_UQ and _full_uq_config is not None:
    _full_joint_uq_config = {**_full_uq_config, "attack_method": "JOINT_PGD"}

# %%
from analysis.utils import EvalConfig, evaluate_sweep

eval_cfg = EvalConfig(
    compute_acc=COMPUTE_ACC,
    compute_rob=COMPUTE_ROB,
    compute_mia=COMPUTE_MIA,
    compute_cls_loss=COMPUTE_CLS_LOSS,
    compute_gen_loss=COMPUTE_GEN_LOSS,
    compute_fid=COMPUTE_FID,
    compute_uq=COMPUTE_UQ,
    splits=EVAL_SPLITS,
    evasion_override=EVASION_CONFIG,
    sampling_override=SAMPLING_OVERRIDE,
    mia_features=MIA_FEATURES,
    mia_adversarial_strength=MIA_ADV_STRENGTH,
    mia_adversarial_num_steps=EVASION_CONFIG.get("num_steps", 20) if EVASION_CONFIG else 20,
    mia_adversarial_step_size=None,
    mia_adversarial_norm=EVASION_CONFIG.get("norm", "inf") if EVASION_CONFIG else "inf",
    uq_config=_full_uq_config if COMPUTE_UQ else None,
    joint_uq_config=_full_joint_uq_config,
    device=DEVICE,
)

sweep_path = project_root / SWEEP_DIR
print("=" * 60)
print(f"Evaluating sweep: {sweep_path}")
print(f"Device: {DEVICE}")
print("=" * 60)

df = evaluate_sweep(str(sweep_path), eval_cfg, config_keys=CONFIG_KEYS)

# %%
# Auto-extract max_epoch for cls_reg sweeps (sweeper param not in CONFIG_KEYS by default)
from analysis.utils.mia_utils import load_run_config as _load_run_config_for_me
if not df.empty and "config/max_epoch" not in df.columns:
    _max_epochs = []
    for _, _r in df.iterrows():
        try:
            _cfg_i = _load_run_config_for_me(Path(_r["run_path"]))
            _me = None
            for _k in ["adversarial", "generative"]:
                _t = getattr(_cfg_i.trainer, _k, None)
                if _t is not None:
                    _me = getattr(_t, "max_epoch", None)
                    break
            _max_epochs.append(_me)
        except Exception:
            _max_epochs.append(None)
    if any(e is not None for e in _max_epochs):
        df["config/max_epoch"] = _max_epochs
        print(f"Auto-extracted config/max_epoch values: {sorted(set(e for e in _max_epochs if e is not None))}")

# For cls_reg: evaluate pretrained model and prepend as max_epoch=0 baseline row
if not df.empty:
    try:
        from analysis.utils import evaluate_pretrained_model
        _first_run_dir = Path(df.iloc[0]["run_path"])
        _first_cfg = _load_run_config_for_me(_first_run_dir)
        _model_path_rel = getattr(_first_cfg, "model_path", None)
        if _model_path_rel is not None:
            _model_path_abs = str(project_root / _model_path_rel)
            print(f"\nDetected cls_reg pretrained model: {_model_path_rel}")
            print("Evaluating pretrained model (will be row max_epoch=0)...")
            _pretrained_metrics = evaluate_pretrained_model(_model_path_abs, _first_run_dir, eval_cfg)
            _pretrained_row = {
                "run_name": "pretrained",
                "run_path": _model_path_abs,
                "config/max_epoch": 0,
                **_pretrained_metrics,
            }
            df = pd.concat([pd.DataFrame([_pretrained_row]), df], ignore_index=True)
            print("Prepended pretrained baseline row (config/max_epoch=0) to evaluation data.")
    except Exception as _e:
        print(f"Warning: Could not evaluate pretrained model baseline: {_e}")

# %%
# Show DataFrame summary
if not df.empty:
    eval_cols = [c for c in df.columns if c.startswith("eval/")]
    print(f"\nEval columns: {eval_cols}")
    print(f"\n{df[['run_name'] + eval_cols].to_string()}")

# %%
# Mirror sweep path under analysis/outputs/:
#   outputs/seed_sweep/X/Y/Z  →  analysis/outputs/seed_sweep/X/Y/Z
_sp = Path(SWEEP_DIR)
if _sp.is_absolute():
    try:
        _sp = _sp.relative_to(project_root)
    except ValueError:
        pass
try:
    _rel = _sp.relative_to("outputs")
except ValueError:
    _rel = _sp
output_dir = project_root / "analysis" / "outputs" / _rel
output_dir.mkdir(parents=True, exist_ok=True)
sweep_name = str(_rel)  # human-readable label used in plot titles and summary
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## 3. Statistics & Visualization

# %%
# Resolve metric columns for the test split.
TEST_ACC = "eval/test/acc" if COMPUTE_ACC else None
TEST_ROB = [c for c in df.columns if c.startswith("eval/test/rob/")] if COMPUTE_ROB and not df.empty else []
TEST_CLS_LOSS = "eval/test/clsloss" if COMPUTE_CLS_LOSS else None

# MIA is split-agnostic (always uses train vs test internally)
MIA_COL = "eval/mia_accuracy" if COMPUTE_MIA else None

# Adversarial MIA: best worst-case threshold accuracy across all features
ADV_MIA_COL = None
ADV_MIA_FEATURE_COLS = []
if COMPUTE_MIA and MIA_ADV_STRENGTH is not None and not df.empty:
    if "eval/adv_mia_wc_best" in df.columns:
        ADV_MIA_COL = "eval/adv_mia_wc_best"
    ADV_MIA_FEATURE_COLS = [c for c in df.columns if c.startswith("eval/adv_mia_wc/")]

# Clean worst-case threshold (for apples-to-apples comparison with adversarial MIA)
MIA_WC_COL = None
MIA_WC_FEATURE_COLS = []
if COMPUTE_MIA and MIA_ADV_STRENGTH is not None and not df.empty:
    if "eval/mia_wc_best" in df.columns:
        MIA_WC_COL = "eval/mia_wc_best"
    MIA_WC_FEATURE_COLS = [c for c in df.columns if c.startswith("eval/mia_wc/")]

UQ_ADV_ACC_COLS = []
UQ_PURIFY_ACC_COLS = []
UQ_PURIFY_RECOVERY_COLS = []
UQ_DETECTION_COLS = []
GIBBS_PURIFY_ACC_COLS = []
UQ_CLEAN_PURIFY_ACC_COLS = []
GIBBS_CLEAN_PURIFY_ACC_COLS = []
if COMPUTE_UQ and not df.empty:
    UQ_ADV_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_adv_acc/"))
    UQ_PURIFY_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_purify_acc/"))
    UQ_PURIFY_RECOVERY_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_purify_recovery/"))
    UQ_DETECTION_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_detection/"))
    GIBBS_PURIFY_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/gibbs_purify_acc/"))
    UQ_CLEAN_PURIFY_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_clean_purify_acc/"))
    GIBBS_CLEAN_PURIFY_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/gibbs_clean_purify_acc/"))

JOINT_ADV_ACC_COLS = []
JOINT_PURIFY_ACC_COLS = []
JOINT_DETECTION_COLS = []
GIBBS_JOINT_PURIFY_ACC_COLS = []
if COMPUTE_JOINT_ATTACK and COMPUTE_UQ and not df.empty:
    JOINT_ADV_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_joint_adv_acc/"))
    JOINT_PURIFY_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_joint_purify_acc/"))
    JOINT_DETECTION_COLS = sorted(c for c in df.columns if c.startswith("eval/uq_joint_detection/"))
    GIBBS_JOINT_PURIFY_ACC_COLS = sorted(c for c in df.columns if c.startswith("eval/gibbs_joint_purify_acc/"))

print(f"TEST_ACC:     {TEST_ACC}")
print(f"TEST_ROB:     {TEST_ROB}")
print(f"MIA_COL:      {MIA_COL}")
print(f"ADV_MIA_COL:  {ADV_MIA_COL}")
print(f"MIA_WC_COL:   {MIA_WC_COL}")
if ADV_MIA_FEATURE_COLS:
    print(f"ADV_MIA per-feature: {ADV_MIA_FEATURE_COLS}")

# Resolve single robustness strength for Pareto frontier (test columns only)
PARETO_TEST_ROB_COL = None

if TEST_ROB:
    _strength_map = {}
    for col in TEST_ROB:
        try:
            s = float(col.split("/")[-1])
            if s > 0:
                _strength_map[s] = col
        except ValueError:
            continue

    if _strength_map:
        if PARETO_ROB_STRENGTH is not None:
            if PARETO_ROB_STRENGTH in _strength_map:
                _chosen = PARETO_ROB_STRENGTH
            else:
                print(f"WARNING: PARETO_ROB_STRENGTH={PARETO_ROB_STRENGTH} not in evaluated "
                      f"strengths {sorted(_strength_map.keys())}. Falling back to weakest.")
                _chosen = min(_strength_map.keys())
        else:
            _chosen = min(_strength_map.keys())

        PARETO_TEST_ROB_COL = _strength_map[_chosen]
        print(f"\nPareto robustness strength: eps={_chosen}")
        print(f"  PARETO_TEST_ROB_COL: {PARETO_TEST_ROB_COL}")

# %% [markdown]
# ### 3a. Representative Run (highest test accuracy, for visualization only)

# %%
from analysis.utils import get_best_run, load_run_config

best_run = None

if not df.empty and TEST_ACC and TEST_ACC in df.columns:
    best_run = get_best_run(df, TEST_ACC, minimize=False)
    if best_run is not None:
        print(f"\n=== Representative Run (highest test accuracy — informational only) ===")
        print(f"Run: {best_run.get('run_name', 'unknown')}")
        if TEST_ACC in best_run.index:
            print(f"  Test Clean Accuracy: {best_run[TEST_ACC]:.4f}")
        for rob_col in TEST_ROB:
            if rob_col in best_run.index:
                strength = rob_col.split("/")[-1]
                print(f"  Test Robust Accuracy (eps={strength}): {best_run[rob_col]:.4f}")
        if MIA_COL and MIA_COL in best_run.index:
            print(f"  MIA Accuracy: {best_run[MIA_COL]:.4f}")
        if COMPUTE_UQ and UQ_PURIFY_ACC_COLS:
            print(f"  --- UQ (Detection + Purification on test) ---")
            uq_clean = best_run.get("eval/uq_clean_accuracy")
            if uq_clean is not None and not np.isnan(uq_clean):
                print(f"  UQ Clean Acc: {uq_clean:.4f}")
            for adv_col in UQ_ADV_ACC_COLS:
                eps = adv_col.split("/")[-1]
                adv_val = best_run.get(adv_col, float("nan"))
                print(f"  Adv Acc (eps={eps}, no defense): {adv_val:.4f}")
                for col in [c for c in UQ_PURIFY_ACC_COLS if f"/{eps}/" in c]:
                    radius = col.split("/")[-1]
                    purify_val = best_run.get(col, float("nan"))
                    delta = purify_val - adv_val if not (np.isnan(purify_val) or np.isnan(adv_val)) else float("nan")
                    delta_str = f"  (Δ={delta:+.4f})" if not np.isnan(delta) else ""
                    print(f"  Purify Acc (eps={eps}, r={radius}): {purify_val:.4f}{delta_str}")
            if UQ_DETECTION_COLS:
                print(f"  Detection rates (fraction of adv examples detected as OOD):")
                for col in UQ_DETECTION_COLS:
                    parts = col.split("/")
                    pct_str, eps = parts[-2], parts[-1]
                    rate = best_run.get(col, float("nan"))
                    print(f"    tau={pct_str}, eps={eps}: {rate:.4f}")

# %% [markdown]
# ### 3d. Metric-Metric Correlation Heatmaps (test)

# %%
from analysis.utils import compute_metric_correlations

# Initialized here so Section 6 can reference it regardless of df being empty
corr_test = pd.DataFrame()

if not df.empty:
    test_metrics = [c for c in [TEST_ACC, TEST_CLS_LOSS, MIA_COL, MIA_WC_COL, ADV_MIA_COL] if c] + list(TEST_ROB)
    # Keep only columns present in df with at least 2 distinct values (drops constants → avoids NaN rows)
    test_metrics = [c for c in test_metrics if c in df.columns and df[c].nunique() > 1]

    if len(test_metrics) >= 2:
        corr_test = compute_metric_correlations(df, test_metrics)
        if not corr_test.empty:
            print("\nMetric-Metric Correlations (test):")
            print(corr_test.round(3).to_string())

# %% [markdown]
# ### 3f. Pareto Frontiers (test)

# %%
from analysis.utils import get_pareto_runs

# Compute acc-vs-strength band data (mean ± std ± stderr) for TXT export
band_data = {}  # {eps: {"mean": ..., "std": ..., "stderr": ...}}
if not df.empty and TEST_ACC:
    _band_cols = [(0.0, TEST_ACC)] + [(float(c.split("/")[-1]), c) for c in TEST_ROB]
    for _eps, _col in _band_cols:
        if _col in df.columns:
            _vals = df[_col].dropna()
            _n = len(_vals)
            band_data[_eps] = {
                "mean": _vals.mean(),
                "std": _vals.std(),
                "stderr": _vals.std() / (_n ** 0.5) if _n > 0 else float("nan"),
            }

if not df.empty and PARETO_TEST_ROB_COL:
    _pareto_strength = PARETO_TEST_ROB_COL.split("/")[-1]

    if TEST_ACC:
        pareto_df = get_pareto_runs(df, TEST_ACC, PARETO_TEST_ROB_COL, True, True)
        if not pareto_df.empty:
            print(f"\nPareto-optimal runs (test acc vs rob/{_pareto_strength}):")
            display_cols = ["run_name", TEST_ACC, PARETO_TEST_ROB_COL]
            display_cols = [c for c in display_cols if c in pareto_df.columns]
            print(pareto_df[display_cols].to_string(index=False))

# %% [markdown]
# ### 3g. Adversarial MIA Results

# %%
# Compute sweep-mean correct_prob arrays (best model's arrays accessible via best_run row)
if not df.empty and COMPUTE_MIA:
    TRAIN_CP_COL = "eval/mia_train_correct_probs"
    TEST_CP_COL = "eval/mia_test_correct_probs"
    if TRAIN_CP_COL in df.columns and TEST_CP_COL in df.columns:
        train_arrays = [np.array(x) for x in df[TRAIN_CP_COL].dropna() if x is not None]
        test_arrays = [np.array(x) for x in df[TEST_CP_COL].dropna() if x is not None]
        if train_arrays and len(set(a.shape for a in train_arrays)) == 1:
            mean_train = np.mean(train_arrays, axis=0).tolist()
            df["eval/mia_mean_train_correct_probs"] = [mean_train] * len(df)
        elif train_arrays:
            print("Warning: mia_train_correct_probs arrays have inhomogeneous shapes "
                  "(e.g. varying split sizes across seeds) — skipping sweep-mean.")
        if test_arrays and len(set(a.shape for a in test_arrays)) == 1:
            mean_test = np.mean(test_arrays, axis=0).tolist()
            df["eval/mia_mean_test_correct_probs"] = [mean_test] * len(df)
        elif test_arrays:
            print("Warning: mia_test_correct_probs arrays have inhomogeneous shapes "
                  "(e.g. varying split sizes across seeds) — skipping sweep-mean.")

if not df.empty and ADV_MIA_COL and ADV_MIA_FEATURE_COLS:
    print(f"\nAdversarial MIA Worst-Case Threshold (eps={MIA_ADV_STRENGTH}):")
    print("  Per-feature accuracy (oracle threshold, mean +/- std across runs):\n")
    if MIA_WC_COL and MIA_WC_COL in df.columns:
        vals_wc = df[MIA_WC_COL].dropna()
        print(f"    {'WC Threshold (clean)':25s}  {vals_wc.mean():.4f} +/- {vals_wc.std():.4f}")
    for col in sorted(ADV_MIA_FEATURE_COLS):
        feat = col.split("/")[-1]
        vals = df[col].dropna()
        print(f"    {feat:25s}  {vals.mean():.4f} +/- {vals.std():.4f}")

    if ADV_MIA_COL in df.columns:
        vals = df[ADV_MIA_COL].dropna()
        print(f"\n    {'BEST (across features)':25s}  {vals.mean():.4f} +/- {vals.std():.4f}")

# %% [markdown]
# ### 3h. UQ Results (detection + purification across all runs)

# %%
if not df.empty and COMPUTE_UQ and UQ_PURIFY_ACC_COLS:
    print(f"\nUQ Purification Results (mean +/- std across runs, test split):\n")
    for adv_col in UQ_ADV_ACC_COLS:
        eps = adv_col.split("/")[-1]
        adv_vals = df[adv_col].dropna()
        print(f"  Adv Acc (eps={eps}, no defense): {adv_vals.mean():.4f} +/- {adv_vals.std():.4f}")
        for col in [c for c in UQ_PURIFY_ACC_COLS if f"/{eps}/" in c]:
            radius = col.split("/")[-1]
            purify_vals = df[col].dropna()
            adv_mean = adv_vals.mean()
            purify_mean, purify_std = purify_vals.mean(), purify_vals.std()
            delta = purify_mean - adv_mean
            print(f"  Purify Acc (eps={eps}, r={radius}): "
                  f"{purify_mean:.4f} +/- {purify_std:.4f}  (Δ={delta:+.4f})")
    if UQ_DETECTION_COLS:
        print(f"\nUQ Detection Rates (mean +/- std across runs):\n")
        for col in UQ_DETECTION_COLS:
            parts = col.split("/")
            pct_str, eps = parts[-2], parts[-1]
            vals = df[col].dropna()
            print(f"  tau={pct_str}, eps={eps}: {vals.mean():.4f} +/- {vals.std():.4f}")

# %% [markdown]
# ### 3i. Summary Statistics Table (test)

# %%
from analysis.utils import create_summary_table

if not df.empty and TEST_ACC:
    summary_df = create_summary_table(
        df, acc_col=TEST_ACC, rob_cols=TEST_ROB, mia_col=MIA_COL,
        effective_n=EFFECTIVE_N,
    )

    if not summary_df.empty:
        print(f"\nEffective N: {EFFECTIVE_N if EFFECTIVE_N else len(df)} (actual runs: {len(df)})")
        print()
        print(summary_df.to_string(index=False))

# %% [markdown]
# ## 4. Sanity Check vs W&B Summary Metrics

# %%
if not df.empty and SANITY_CHECK_METRICS:
    from analysis.utils import load_local_hpo_runs

    print("\n" + "=" * 60)
    print("Sanity Check: Post-hoc vs W&B Summary Metrics")
    print("=" * 60)

    # Load W&B summary data
    wb_df = load_local_hpo_runs(sweep_path)

    if not wb_df.empty:
        # Merge on run_name
        merged = df.merge(wb_df, on="run_name", suffixes=("_eval", "_wb"))

        for eval_col, wb_col in SANITY_CHECK_METRICS.items():
            if eval_col not in merged.columns:
                print(f"\n  {eval_col}: not in eval results")
                continue
            # Handle suffix from merge
            wb_actual = wb_col if wb_col in merged.columns else wb_col + "_wb"
            if wb_actual not in merged.columns:
                print(f"\n  {wb_col}: not in W&B summary data")
                continue

            eval_vals = merged[eval_col].astype(float)
            wb_vals = merged[wb_actual].astype(float)
            diff = (eval_vals - wb_vals).abs()

            print(f"\n  {eval_col} vs {wb_col}:")
            print(f"    Max absolute diff: {diff.max():.6f}")
            print(f"    Mean absolute diff: {diff.mean():.6f}")

            mismatches = diff > SANITY_CHECK_TOL
            if mismatches.any():
                print(f"    WARNING: {mismatches.sum()} runs differ by > {SANITY_CHECK_TOL}")
                mismatch_df = merged.loc[mismatches, ["run_name", eval_col, wb_actual]]
                print(mismatch_df.to_string(index=False))
            else:
                print(f"    All runs match within tolerance {SANITY_CHECK_TOL}")
    else:
        print("Could not load W&B summary data for comparison.")

# %% [markdown]
# ## 5. Learned Distribution Visualization

# %%
if COMPUTE_DISTRIBUTIONS and not df.empty and best_run is not None:
    best_run_path = best_run.get("run_path")
    if best_run_path:
        print(f"\n--- Visualizing distributions for best run ({best_run['run_name']}) ---")

        # Generated samples
        try:
            from src.tracking.visualisation import visualise_samples
            from analysis.utils import find_model_checkpoint
            from src.models import BornMachine

            bm = BornMachine.load(str(find_model_checkpoint(best_run_path)))
            bm.to(torch.device(DEVICE))
            bm.sync_tensors(after="classification")

            cfg = load_run_config(best_run_path)
            with torch.no_grad():
                synths = bm.sample(cfg=cfg.tracking.sampling)
            ax = visualise_samples(synths, input_range=bm.input_range)
            if ax is not None:
                fig = ax.get_figure()
                fig.savefig(output_dir / "best_run_samples.png", bbox_inches="tight", dpi=DPI)
                print(f"Saved best_run_samples.png")
                plt.show()

            del bm
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not generate samples: {e}")

        # Distribution heatmaps
        try:
            from analysis.visualize.distributions import visualize_from_run_dir

            visualize_from_run_dir(
                run_dir=best_run_path,
                resolution=150,
                normalize_joint=True,
                show_data=True,
                device=DEVICE,
                save_dir=str(output_dir),
            )
            # directly saves best_class_dist.png + best_joint.png
            print(f"Saved best_class_dist.png + best_joint.png")
            plt.show()
        except Exception as e:
            print(f"Warning: Could not generate distribution visualization: {e}")

# %% [markdown]
# ## 6. Summary Export

# %%
if not df.empty:
    summary_path = output_dir / "evaluation_summary.txt"

    def _smean(col):
        if col and col in df.columns:
            v = df[col].dropna()
            return v.mean() if len(v) > 0 else float("nan")
        return float("nan")

    def _sstd(col):
        if col and col in df.columns:
            v = df[col].dropna()
            return v.std() if len(v) > 0 else float("nan")
        return float("nan")

    def _sfmt(v, na="—", w=8):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return na.rjust(w)
        return f"{v:.4f}".rjust(w)

    # Map float eps → string suffix as it appears in column names
    _eps_key = {}
    for _c in TEST_ROB:
        _s = _c.split("/")[-1]
        _eps_key[float(_s)] = _s
    _all_eps = [0.0] + sorted(_eps_key.keys())

    # Parse UQ structure from available columns
    _det_pcts = sorted(set(
        int(c.split("/")[-2].replace("pct", "")) for c in UQ_DETECTION_COLS
    )) if UQ_DETECTION_COLS else []
    _purify_radii = sorted(set(
        c.split("/")[-1] for c in UQ_PURIFY_ACC_COLS + UQ_CLEAN_PURIFY_ACC_COLS
    )) if (UQ_PURIFY_ACC_COLS or UQ_CLEAN_PURIFY_ACC_COLS) else []
    _gibbs_ns = sorted(set(
        int(c.split("/")[-1]) for c in GIBBS_PURIFY_ACC_COLS + GIBBS_CLEAN_PURIFY_ACC_COLS
    )) if (GIBBS_PURIFY_ACC_COLS or GIBBS_CLEAN_PURIFY_ACC_COLS) else []

    # Build main table rows: (label, [mean_per_eps])
    _table_rows = []

    # Row 1: no defense
    _nodef = []
    for _e in _all_eps:
        _nodef.append(_smean(TEST_ACC) if _e == 0.0 else _smean(f"eval/test/rob/{_eps_key[_e]}"))
    _table_rows.append(("No defense", _nodef))

    # Row 2: detection at smallest percentile (not applicable at eps=0)
    if _det_pcts:
        _min_pct = _det_pcts[0]
        _det = [float("nan")] + [_smean(f"eval/uq_detection/{_min_pct}pct/{_eps_key[_e]}") for _e in _all_eps[1:]]
        _table_rows.append((f"Detection (τ={_min_pct}%)", _det))

    # Row(s) 3: likelihood purification (one row per radius)
    for _r in _purify_radii:
        _clean_pur = _smean(f"eval/uq_clean_purify_acc/{_r}")
        _purify = [_clean_pur] + [_smean(f"eval/uq_purify_acc/{_eps_key[_e]}/{_r}") for _e in _all_eps[1:]]
        _table_rows.append((f"Purify (r={_r})", _purify))

    # Row(s) 4: Gibbs purification (one row per n_sweeps)
    for _n in _gibbs_ns:
        _clean_gibbs = _smean(f"eval/gibbs_clean_purify_acc/{_n}")
        _gibbs = [_clean_gibbs] + [_smean(f"eval/gibbs_purify_acc/{_eps_key[_e]}/{_n}") for _e in _all_eps[1:]]
        _table_rows.append((f"Gibbs (k={_n})", _gibbs))

    # Column widths
    _label_w = max((len(r[0]) for r in _table_rows), default=12) + 2
    _col_w = 9
    _eps_hdr = ["eps=0"] + [_eps_key[_e] for _e in _all_eps[1:]]

    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Seed Sweep: {sweep_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Regime: {REGIME}  |  Runs: {len(df)}  |  Device: {DEVICE}\n")
        if EVASION_CONFIG:
            _norm = EVASION_CONFIG.get("norm", "?")
            _steps = EVASION_CONFIG.get("num_steps", "?")
            _method = EVASION_CONFIG.get("method", "?")
            f.write(f"Attack: {_method} L{_norm}  steps={_steps}\n")
        f.write("\n")

        # --- Main accuracy table (means only) ---
        f.write("-" * 60 + "\n")
        f.write("Accuracy vs Perturbation Strength (means)\n")
        f.write("-" * 60 + "\n\n")
        _hdr_line = " " * _label_w + "".join(h.rjust(_col_w) for h in _eps_hdr)
        f.write(_hdr_line + "\n")
        for _lbl, _vals in _table_rows:
            f.write(f"{_lbl:<{_label_w}}" + "".join(_sfmt(v) for v in _vals) + "\n")
        f.write("\n")

        # --- Detection rates table at 2nd-smallest non-zero eps ---
        if _det_pcts and len(_all_eps) >= 2:
            _nonzero = [_e for _e in _all_eps if _e > 0.0]
            _det_eps = _nonzero[1] if len(_nonzero) >= 2 else _nonzero[0]
            _det_eps_s = _eps_key[_det_eps]
            f.write("-" * 60 + "\n")
            f.write(f"Detection Rates at eps={_det_eps_s} (mean / std)\n")
            f.write("-" * 60 + "\n\n")
            _pct_hdr = " " * 6 + "".join(f"τ={p}%".rjust(_col_w) for p in _det_pcts)
            f.write(_pct_hdr + "\n")
            _dmeans = [_smean(f"eval/uq_detection/{p}pct/{_det_eps_s}") for p in _det_pcts]
            _dstds  = [_sstd( f"eval/uq_detection/{p}pct/{_det_eps_s}") for p in _det_pcts]
            f.write(f"{'mean':<6}" + "".join(_sfmt(v) for v in _dmeans) + "\n")
            f.write(f"{'std':<6}"  + "".join(_sfmt(v) for v in _dstds)  + "\n")
            f.write("\n")

        # --- Joint Attack table ---
        if COMPUTE_JOINT_ATTACK and JOINT_ADV_ACC_COLS:
            _joint_eps_keys = {}
            for _c in JOINT_ADV_ACC_COLS:
                _s = _c.split("/")[-1]
                _joint_eps_keys[float(_s)] = _s
            _joint_all_eps = sorted(_joint_eps_keys.keys())
            _joint_purify_radii = sorted(set(c.split("/")[-1] for c in JOINT_PURIFY_ACC_COLS)) if JOINT_PURIFY_ACC_COLS else []
            _joint_det_pcts = sorted(set(
                int(c.split("/")[-2].replace("pct", "")) for c in JOINT_DETECTION_COLS
            )) if JOINT_DETECTION_COLS else []
            _gibbs_joint_ns = sorted(set(int(c.split("/")[-1]) for c in GIBBS_JOINT_PURIFY_ACC_COLS)) if GIBBS_JOINT_PURIFY_ACC_COLS else []

            _joint_rows = []
            _joint_rows.append(("No defense", [_smean(f"eval/uq_joint_adv_acc/{_joint_eps_keys[_e]}") for _e in _joint_all_eps]))
            if _joint_det_pcts:
                _min_pct = _joint_det_pcts[0]
                _joint_rows.append((f"Detection (τ={_min_pct}%)", [
                    _smean(f"eval/uq_joint_detection/{_min_pct}pct/{_joint_eps_keys[_e]}") for _e in _joint_all_eps
                ]))
            for _r in _joint_purify_radii:
                _joint_rows.append((f"Purify (r={_r})", [
                    _smean(f"eval/uq_joint_purify_acc/{_joint_eps_keys[_e]}/{_r}") for _e in _joint_all_eps
                ]))
            for _n in _gibbs_joint_ns:
                _joint_rows.append((f"Gibbs (k={_n})", [
                    _smean(f"eval/gibbs_joint_purify_acc/{_joint_eps_keys[_e]}/{_n}") for _e in _joint_all_eps
                ]))

            _jlabel_w = max((len(r[0]) for r in _joint_rows), default=12) + 2
            _jeps_hdr = [_joint_eps_keys[_e] for _e in _joint_all_eps]

            f.write("-" * 60 + "\n")
            f.write("Joint Attack (JOINT_PGD) — Accuracy vs Perturbation Strength\n")
            f.write("-" * 60 + "\n\n")
            f.write(" " * _jlabel_w + "".join(h.rjust(_col_w) for h in _jeps_hdr) + "\n")
            for _lbl, _vals in _joint_rows:
                f.write(f"{_lbl:<{_jlabel_w}}" + "".join(_sfmt(v) for v in _vals) + "\n")
            f.write("\n")

        # --- MIA ---
        if MIA_COL and MIA_COL in df.columns:
            f.write("-" * 60 + "\n")
            f.write("Membership Inference Attack\n")
            f.write("-" * 60 + "\n\n")
            f.write(f"  LR accuracy:  {_smean(MIA_COL):.4f} ± {_sstd(MIA_COL):.4f}\n")
            if "eval/mia_auc_roc" in df.columns:
                f.write(f"  AUC-ROC:      {_smean('eval/mia_auc_roc'):.4f} ± {_sstd('eval/mia_auc_roc'):.4f}\n")
            if MIA_WC_COL and MIA_WC_COL in df.columns:
                f.write(f"  WC threshold (clean): {_smean(MIA_WC_COL):.4f} ± {_sstd(MIA_WC_COL):.4f}\n")
            if ADV_MIA_COL and ADV_MIA_COL in df.columns:
                f.write(f"  WC threshold (adv, eps={MIA_ADV_STRENGTH}): "
                        f"{_smean(ADV_MIA_COL):.4f} ± {_sstd(ADV_MIA_COL):.4f}\n")
                for _col in sorted(ADV_MIA_FEATURE_COLS):
                    _feat = _col.split("/")[-1]
                    f.write(f"    {_feat}: {_smean(_col):.4f} ± {_sstd(_col):.4f}\n")
            f.write("\n")

        # --- Pareto frontier ---
        if TEST_ACC and PARETO_TEST_ROB_COL:
            _pf_df = get_pareto_runs(df, TEST_ACC, PARETO_TEST_ROB_COL, True, True)
            if not _pf_df.empty:
                _pf_strength = PARETO_TEST_ROB_COL.split("/")[-1]
                f.write("-" * 60 + "\n")
                f.write(f"Pareto Frontier — test acc vs rob at eps={_pf_strength}\n")
                f.write("-" * 60 + "\n\n")
                _pf_cols = [c for c in ["run_name", TEST_ACC, PARETO_TEST_ROB_COL] if c in _pf_df.columns]
                f.write(_pf_df[_pf_cols].to_string(index=False) + "\n\n")

        # --- Metric correlations ---
        if not corr_test.empty:
            f.write("-" * 60 + "\n")
            f.write("Metric Correlations (test)\n")
            f.write("-" * 60 + "\n\n")
            f.write(corr_test.round(3).to_string() + "\n\n")

        f.write("=" * 60 + "\n")

    print(f"\nExported summary to: {summary_path}")

# %%
# --- Export full per-run evaluation data ---
# Column groups (see analysis/GUIDE.md for full reference and pandas query examples):
#   run_name, run_path          — identity
#   config/<key>                — extracted Hydra config values (from CONFIG_KEYS)
#   eval/<split>/acc            — clean classification accuracy per split
#   eval/<split>/clsloss        — classification NLL loss (if COMPUTE_CLS_LOSS)
#   eval/<split>/rob/<eps>      — robust accuracy at each epsilon, per split
#   eval/mia_accuracy, eval/mia_auc_roc         — LR attack accuracy and AUC-ROC
#   eval/mia_wc/<feat>, eval/mia_wc_best        — per-feature worst-case threshold accuracy
#   eval/adv_mia_wc/<feat>, eval/adv_mia_wc_best — adversarial MIA worst-case accuracy
#   eval/uq_clean_accuracy      — UQ clean accuracy on test
#   eval/uq_adv_acc/<eps>       — adversarial accuracy (no defense) at each eps
#   eval/uq_detection/<pct>pct/<eps>  — detection rate at threshold percentile & eps
#   eval/uq_purify_acc/<eps>/<r>      — purification accuracy at (eps, radius)
#   eval/uq_purify_recovery/<eps>/<r> — recovery rate at (eps, radius)
if not df.empty:
    _eval_export_cols = [c for c in df.columns if not c.startswith("_")]
    df[_eval_export_cols].to_csv(output_dir / "evaluation_data.csv", index=False)
    print(f"Saved evaluation_data.csv ({len(df)} runs, {len(_eval_export_cols)} columns)")

# %% [markdown]
# ## Completion

# %%
print("\n" + "=" * 60)
print("Post-Hoc Seed Sweep Analysis Complete!")
print("=" * 60)
print(f"\nSweep: {sweep_name}")
print(f"Regime: {REGIME}")
print(f"Total runs: {len(df) if not df.empty else 0}")
print(f"\nOutputs saved to: {output_dir}")
print("=" * 60)

# %%
