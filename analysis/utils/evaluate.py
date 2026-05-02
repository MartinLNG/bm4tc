"""
Post-hoc per-model evaluation for sweep analysis.

Loads each saved model from a sweep directory and recomputes metrics
using ``MetricFactory.create()`` directly, giving the user control over
metric selection, evasion/sampling configs, and evaluation splits.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from .mia_utils import load_run_config, find_model_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Configuration for post-hoc evaluation.

    Attributes:
        compute_acc: Evaluate clean classification accuracy.
        compute_rob: Evaluate adversarial robustness.
        compute_mia: Run membership inference attack evaluation.
        compute_cls_loss: Evaluate classification loss (NLL).
        compute_gen_loss: Evaluate generative loss (joint NLL).
        compute_fid: Evaluate FID-like score.
        splits: Data splits to evaluate on (e.g. ["valid", "test"]).
        evasion_override: Dict of evasion config fields to override,
            or None to use each run's own config.
        sampling_override: Dict of sampling config fields to override,
            or None to use each run's own config.
        mia_features: Dict of MIA feature toggles (passed to MIAFeatureConfig).
            Includes use_true_labels (bool) for correct_prob/loss features.
        mia_adversarial_strength: Epsilon for adversarial MIA (PGD attack on
            inputs before feature extraction). None = skip adversarial MIA.
        mia_adversarial_num_steps: Number of PGD steps for adversarial MIA.
        mia_adversarial_step_size: PGD step size. None = auto (2.5 * eps / steps).
        mia_adversarial_norm: Lp norm for adversarial MIA perturbation ball.
        device: Torch device string.
    """
    compute_acc: bool = True
    compute_rob: bool = True
    compute_mia: bool = True
    compute_cls_loss: bool = False
    compute_gen_loss: bool = False
    compute_fid: bool = False
    compute_uq: bool = False
    splits: List[str] = field(default_factory=lambda: ["test"])
    evasion_override: Optional[Dict[str, Any]] = None
    sampling_override: Optional[Dict[str, Any]] = None
    mia_features: Optional[Dict[str, bool]] = None
    mia_adversarial_strength: Optional[float] = None
    mia_adversarial_num_steps: int = 20
    mia_adversarial_step_size: Optional[float] = None
    mia_adversarial_norm: Any = "inf"
    uq_config: Optional[Dict[str, Any]] = None
    joint_uq_config: Optional[Dict[str, Any]] = None  # second UQ pass with JOINT_PGD
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_config_overrides(
    cfg,
    evasion_override: Optional[Dict[str, Any]],
    sampling_override: Optional[Dict[str, Any]],
):
    """Mutate *cfg* in-place, overriding tracking.evasion / tracking.sampling."""
    if evasion_override:
        for key, value in evasion_override.items():
            OmegaConf.update(cfg, f"tracking.evasion.{key}", value, force_add=True)
    if sampling_override:
        for key, value in sampling_override.items():
            OmegaConf.update(cfg, f"tracking.sampling.{key}", value, force_add=True)


def resolve_stop_criterion(
    stop_crit: str,
    df: pd.DataFrame,
    eval_split: str = "test",
) -> Tuple[Optional[str], bool, str]:
    """Map a stop-criterion name to the corresponding eval column.

    Args:
        stop_crit: Short name like "acc", "loss", "rob".
        df: Eval DataFrame (to check available columns).
        eval_split: Split used during evaluation (e.g. "test").

    Returns:
        (column_name, minimize, label) or (None, True, "") if unresolvable.
    """
    prefix = f"eval/{eval_split}"
    if stop_crit == "acc":
        col = f"{prefix}/acc"
        return (col, False, "Accuracy (stop crit)") if col in df.columns else (None, True, "")
    elif stop_crit == "loss" or stop_crit == "clsloss":
        col = f"{prefix}/clsloss"
        return (col, True, "Cls Loss (stop crit)") if col in df.columns else (None, True, "")
    elif stop_crit == "rob":
        rob_cols = [c for c in df.columns if c.startswith(f"{prefix}/rob/")]
        if rob_cols:
            df["_avg_rob_acc"] = df[rob_cols].mean(axis=1)
            return ("_avg_rob_acc", False, "Avg Robust Accuracy (stop crit)")
        return (None, True, "")
    elif stop_crit == "genloss":
        col = f"{prefix}/genloss"
        return (col, True, "Gen Loss (stop crit)") if col in df.columns else (None, True, "")
    return (None, True, "")


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: Path,
    eval_cfg: EvalConfig,
) -> Dict[str, float]:
    """Load a single run's model and compute metrics post-hoc.

    Args:
        run_dir: Path to the Hydra output directory for one run.
        eval_cfg: Evaluation configuration.

    Returns:
        Flat dict with keys like ``eval/test/acc``, ``eval/valid/rob/0.1``,
        ``eval/mia_accuracy``, ``eval/mia_auc_roc``.
    """
    from src.models import BornMachine
    from src.data.handler import DataHandler
    from src.tracking.evaluator import MetricFactory

    run_dir = Path(run_dir)
    device = torch.device(eval_cfg.device)
    results: Dict[str, float] = {}

    # 1. Load config
    cfg = load_run_config(run_dir)

    # 2. Apply evasion/sampling overrides
    _apply_config_overrides(cfg, eval_cfg.evasion_override, eval_cfg.sampling_override)

    # 3. Load model
    checkpoint_path = find_model_checkpoint(run_dir)
    bm = BornMachine.load(str(checkpoint_path))
    bm.to(device)

    # 4. Sync generator to match classifier tensors (needed for gen loss & FID)
    if eval_cfg.compute_gen_loss or eval_cfg.compute_fid:
        bm.sync_tensors(after="classification")

    # 5. Ensure dataset is regenerated with correct seed
    OmegaConf.update(cfg, "dataset.overwrite", True, force_add=True)

    # 6. Load data
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bm)
    datahandler.get_classification_loaders()

    # 7. Evaluate non-rob metrics on all splits; rob on non-test splits only.
    #    Test-split rob is deferred to step 10 to reuse UQ's adversarial examples.
    non_rob_metrics = []
    if eval_cfg.compute_acc:
        non_rob_metrics.append("acc")
    if eval_cfg.compute_cls_loss:
        non_rob_metrics.append("clsloss")
    if eval_cfg.compute_gen_loss:
        non_rob_metrics.append("genloss")
    if eval_cfg.compute_fid:
        non_rob_metrics.append("fid")

    for split in eval_cfg.splits:
        context: Dict[str, Any] = {}
        for metric_name in non_rob_metrics:
            try:
                metric = MetricFactory.create(
                    metric_name=metric_name,
                    freq=1,
                    cfg=cfg,
                    datahandler=datahandler,
                    device=device,
                )
                result = metric.evaluate(bm, split, context)
                results[f"eval/{split}/{metric_name}"] = float(result)
            except Exception as e:
                logger.warning(f"Metric '{metric_name}' failed on split '{split}': {e}")
                results[f"eval/{split}/{metric_name}"] = np.nan

    # This is only for non-seed-sweep evals.
    # Valid-split (and any non-test split) rob via MetricFactory. 
    # UQ only evaluates on test; valid rob still uses MetricFactory directly.
    if eval_cfg.compute_rob:
        for split in [s for s in eval_cfg.splits if s != "test"]:
            context: Dict[str, Any] = {}
            try:
                metric = MetricFactory.create(
                    metric_name="rob",
                    freq=1,
                    cfg=cfg,
                    datahandler=datahandler,
                    device=device,
                )
                result = metric.evaluate(bm, split, context)
                strengths = cfg.tracking.evasion.strengths
                for i, strength in enumerate(strengths):
                    results[f"eval/{split}/rob/{strength}"] = float(result[i])
            except Exception as e:
                logger.warning(f"Metric 'rob' failed on split '{split}': {e}")
                strengths = getattr(cfg.tracking.evasion, "strengths", [])
                for strength in strengths:
                    results[f"eval/{split}/rob/{strength}"] = np.nan

    # 8. MIA
    if eval_cfg.compute_mia:
        try:
            from .mia import MIAEvaluation, MIAFeatureConfig

            feature_kwargs = eval_cfg.mia_features or {}
            feature_config = MIAFeatureConfig(**feature_kwargs)
            mia_eval = MIAEvaluation(
                feature_config=feature_config,
                adversarial_strength=eval_cfg.mia_adversarial_strength,
                adversarial_num_steps=eval_cfg.mia_adversarial_num_steps,
                adversarial_step_size=eval_cfg.mia_adversarial_step_size,
                adversarial_norm=eval_cfg.mia_adversarial_norm,
            )

            mia_results = mia_eval.evaluate(
                bm,
                datahandler.classification["train"],
                datahandler.classification["test"],
                device,
            )
            results["eval/mia_accuracy"] = mia_results.attack_accuracy
            results["eval/mia_auc_roc"] = mia_results.auc_roc

            # Store correct_prob arrays for histogram analysis
            if "correct_prob" in mia_results.feature_names:
                cp_idx = mia_results.feature_names.index("correct_prob")
                results["eval/mia_train_correct_probs"] = mia_results.train_features[:, cp_idx].tolist()
                results["eval/mia_test_correct_probs"] = mia_results.test_features[:, cp_idx].tolist()

            # Store adversarial MIA worst-case threshold results
            if mia_results.adversarial_worst_case_threshold is not None:
                for feat_name, metrics in mia_results.adversarial_worst_case_threshold.items():
                    results[f"eval/adv_mia_wc/{feat_name}"] = metrics["accuracy"]
                # Store the best adversarial MIA accuracy across features
                best_adv_mia = max(
                    m["accuracy"] for m in mia_results.adversarial_worst_case_threshold.values()
                )
                results["eval/adv_mia_wc_best"] = best_adv_mia

                # Store clean worst-case threshold for clean-vs-adversarial comparison
                # (same oracle threshold metric as adversarial_worst_case_threshold).
                if mia_results.worst_case_threshold:
                    for feat_name, metrics in mia_results.worst_case_threshold.items():
                        results[f"eval/mia_wc/{feat_name}"] = metrics["accuracy"]
                    results["eval/mia_wc_best"] = max(
                        m["accuracy"] for m in mia_results.worst_case_threshold.values()
                    )

        except Exception as e:
            logger.warning(f"MIA evaluation failed: {e}")
            results["eval/mia_accuracy"] = np.nan
            results["eval/mia_auc_roc"] = np.nan

    # 9. UQ (detection + purification) — runs before test-split rob to share
    #    adversarial examples generated on the test set.
    uq_results = None
    if eval_cfg.compute_uq:
        try:
            from .uq import UQEvaluation, UQConfig

            uq_kwargs = eval_cfg.uq_config or {}
            uq_config = UQConfig(**uq_kwargs)
            uq_eval = UQEvaluation(config=uq_config)

            uq_results = uq_eval.evaluate(
                bm,
                datahandler.classification["test"],
                device,
            )
            results["eval/uq_clean_accuracy"] = uq_results.clean_accuracy
            results["eval/uq_clean_log_px_mean"] = float(uq_results.clean_log_px.mean())

            for eps, acc in uq_results.adv_accuracies.items():
                results[f"eval/uq_adv_acc/{eps}"] = acc

            for (pct, eps), rate in uq_results.detection_rates.items():
                results[f"eval/uq_detection/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in uq_results.err_rate_detected.items():
                results[f"eval/uq_det_err_detected/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in uq_results.err_rate_passed.items():
                results[f"eval/uq_det_err_passed/{pct}pct/{eps}"] = rate

            for (eps, radius), metrics in uq_results.purification_results.items():
                results[f"eval/uq_purify_acc/{eps}/{radius}"] = metrics.accuracy_after_purify
                results[f"eval/uq_purify_recovery/{eps}/{radius}"] = metrics.recovery_rate

            for (eps, n_sweeps), metrics in uq_results.gibbs_purification_results.items():
                results[f"eval/gibbs_purify_acc/{eps}/{n_sweeps}"] = metrics.accuracy_after_purify
                results[f"eval/gibbs_purify_recovery/{eps}/{n_sweeps}"] = metrics.recovery_rate

            for radius, metrics in uq_results.clean_purification_results.items():
                results[f"eval/uq_clean_purify_acc/{radius}"] = metrics.accuracy_after_purify

            for n_sweeps, metrics in uq_results.clean_gibbs_purification_results.items():
                results[f"eval/gibbs_clean_purify_acc/{n_sweeps}"] = metrics.accuracy_after_purify
        except Exception as e:
            logger.warning(f"UQ evaluation failed: {e}")
            results["eval/uq_clean_accuracy"] = np.nan

    # 9b. Joint-attack UQ: second pass with JOINT_PGD, reusing cached log Z.
    if eval_cfg.compute_uq and eval_cfg.joint_uq_config is not None:
        try:
            from .uq import UQEvaluation, UQConfig

            joint_uq_config = UQConfig(**eval_cfg.joint_uq_config)
            joint_uq_eval = UQEvaluation(config=joint_uq_config)
            joint_uq_results = joint_uq_eval.evaluate(
                bm,
                datahandler.classification["test"],
                device,
            )

            for eps, acc in joint_uq_results.adv_accuracies.items():
                results[f"eval/uq_joint_adv_acc/{eps}"] = acc

            for (pct, eps), rate in joint_uq_results.detection_rates.items():
                results[f"eval/uq_joint_detection/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in joint_uq_results.err_rate_detected.items():
                results[f"eval/uq_joint_det_err_detected/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in joint_uq_results.err_rate_passed.items():
                results[f"eval/uq_joint_det_err_passed/{pct}pct/{eps}"] = rate

            for (eps, radius), metrics in joint_uq_results.purification_results.items():
                results[f"eval/uq_joint_purify_acc/{eps}/{radius}"] = metrics.accuracy_after_purify
                results[f"eval/uq_joint_purify_recovery/{eps}/{radius}"] = metrics.recovery_rate

            for (eps, n_sweeps), metrics in joint_uq_results.gibbs_purification_results.items():
                results[f"eval/gibbs_joint_purify_acc/{eps}/{n_sweeps}"] = metrics.accuracy_after_purify
                results[f"eval/gibbs_joint_purify_recovery/{eps}/{n_sweeps}"] = metrics.recovery_rate
        except Exception as e:
            logger.warning(f"Joint-attack UQ evaluation failed: {e}")

    # 10. Test-split rob: reuse UQ's adv_accuracies where possible to avoid
    #     generating adversarial examples twice on the test set.
    # TODO: Check whether this is necessary or if strengths of purification are always equal to strengths of raw rob acc.
    #       If yes, then cut this last block and save robust accuracies in block 9.
    if eval_cfg.compute_rob and "test" in eval_cfg.splits:
        _uq_adv_acc_cache: Dict[float, float] = {}
        if eval_cfg.compute_uq and uq_results is not None:
            _uq_adv_acc_cache = {
                float(eps): acc for eps, acc in uq_results.adv_accuracies.items()
            }

        strengths = cfg.tracking.evasion.strengths
        missing = []
        for strength in strengths:
            s = float(strength)
            if s in _uq_adv_acc_cache:
                results[f"eval/test/rob/{strength}"] = _uq_adv_acc_cache[s]
            else:
                missing.append(strength)

        if missing:
            logger.warning(
                f"Test rob eps {missing} not in UQ cache; generating adversarial examples separately"
            )
            original_strengths = list(cfg.tracking.evasion.strengths)
            try:
                OmegaConf.update(cfg, "tracking.evasion.strengths", missing, force_add=True)
                context: Dict[str, Any] = {}
                metric = MetricFactory.create(
                    metric_name="rob",
                    freq=1,
                    cfg=cfg,
                    datahandler=datahandler,
                    device=device,
                )
                result = metric.evaluate(bm, "test", context)
                for i, strength in enumerate(missing):
                    results[f"eval/test/rob/{strength}"] = float(result[i])
            except Exception as e:
                logger.warning(f"Fallback test rob generation failed: {e}")
                for strength in missing:
                    results[f"eval/test/rob/{strength}"] = np.nan
            finally:
                OmegaConf.update(
                    cfg, "tracking.evasion.strengths", original_strengths, force_add=True
                )

    # 11. Cleanup
    del bm
    torch.cuda.empty_cache()

    return results


def evaluate_pretrained_model(
    model_path: str,
    ref_run_dir: Path,
    eval_cfg: EvalConfig,
) -> Dict[str, float]:
    """Evaluate a pretrained model using the dataset config from a reference run.

    Identical to evaluate_run() except the model is loaded from ``model_path``
    directly (not from find_model_checkpoint(ref_run_dir)).  Used in cls_reg
    sweeps to add a max_epoch=0 baseline row to the evaluation CSV.
    """
    from src.models import BornMachine
    from src.data.handler import DataHandler
    from src.tracking.evaluator import MetricFactory

    ref_run_dir = Path(ref_run_dir)
    device = torch.device(eval_cfg.device)
    results: Dict[str, float] = {}

    cfg = load_run_config(ref_run_dir)
    _apply_config_overrides(cfg, eval_cfg.evasion_override, eval_cfg.sampling_override)

    bm = BornMachine.load(model_path)
    bm.to(device)

    OmegaConf.update(cfg, "dataset.overwrite", True, force_add=True)
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bm)
    datahandler.get_classification_loaders()

    # Non-rob metrics (skip gen/fid — pretrained is classifier only)
    non_rob_metrics = []
    if eval_cfg.compute_acc:
        non_rob_metrics.append("acc")
    if eval_cfg.compute_cls_loss:
        non_rob_metrics.append("clsloss")

    for split in eval_cfg.splits:
        context: Dict[str, Any] = {}
        for metric_name in non_rob_metrics:
            try:
                metric = MetricFactory.create(metric_name, 1, cfg, datahandler, device)
                result = metric.evaluate(bm, split, context)
                results[f"eval/{split}/{metric_name}"] = float(result)
            except Exception as e:
                logger.warning(f"Metric '{metric_name}' on '{split}': {e}")
                results[f"eval/{split}/{metric_name}"] = np.nan

    # Valid-split rob
    if eval_cfg.compute_rob:
        for split in [s for s in eval_cfg.splits if s != "test"]:
            context: Dict[str, Any] = {}
            try:
                metric = MetricFactory.create("rob", 1, cfg, datahandler, device)
                result = metric.evaluate(bm, split, context)
                strengths = cfg.tracking.evasion.strengths
                for i, strength in enumerate(strengths):
                    results[f"eval/{split}/rob/{strength}"] = float(result[i])
            except Exception as e:
                logger.warning(f"Rob on '{split}': {e}")
                strengths = getattr(cfg.tracking.evasion, "strengths", [])
                for strength in strengths:
                    results[f"eval/{split}/rob/{strength}"] = np.nan

    # MIA (train vs test)
    if eval_cfg.compute_mia:
        try:
            from .mia import MIAEvaluation, MIAFeatureConfig
            feature_kwargs = eval_cfg.mia_features or {}
            mia_eval = MIAEvaluation(
                feature_config=MIAFeatureConfig(**feature_kwargs),
                adversarial_strength=eval_cfg.mia_adversarial_strength,
                adversarial_num_steps=eval_cfg.mia_adversarial_num_steps,
                adversarial_step_size=eval_cfg.mia_adversarial_step_size,
                adversarial_norm=eval_cfg.mia_adversarial_norm,
            )
            mia_results = mia_eval.evaluate(
                bm, datahandler.classification["train"],
                datahandler.classification["test"], device,
            )
            results["eval/mia_accuracy"] = mia_results.attack_accuracy
            results["eval/mia_auc_roc"] = mia_results.auc_roc
        except Exception as e:
            logger.warning(f"MIA failed: {e}")
            results["eval/mia_accuracy"] = np.nan
            results["eval/mia_auc_roc"] = np.nan

    # UQ
    uq_results = None
    if eval_cfg.compute_uq:
        try:
            from .uq import UQEvaluation, UQConfig
            uq_eval = UQEvaluation(UQConfig(**(eval_cfg.uq_config or {})))
            uq_results = uq_eval.evaluate(bm, datahandler.classification["test"], device)
            results["eval/uq_clean_accuracy"] = uq_results.clean_accuracy
            for eps, acc in uq_results.adv_accuracies.items():
                results[f"eval/uq_adv_acc/{eps}"] = acc
            for (pct, eps), rate in uq_results.detection_rates.items():
                results[f"eval/uq_detection/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in uq_results.err_rate_detected.items():
                results[f"eval/uq_det_err_detected/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in uq_results.err_rate_passed.items():
                results[f"eval/uq_det_err_passed/{pct}pct/{eps}"] = rate
            for (eps, radius), m in uq_results.purification_results.items():
                results[f"eval/uq_purify_acc/{eps}/{radius}"] = m.accuracy_after_purify
                results[f"eval/uq_purify_recovery/{eps}/{radius}"] = m.recovery_rate
            for (eps, n_sweeps), m in uq_results.gibbs_purification_results.items():
                results[f"eval/gibbs_purify_acc/{eps}/{n_sweeps}"] = m.accuracy_after_purify
                results[f"eval/gibbs_purify_recovery/{eps}/{n_sweeps}"] = m.recovery_rate
            for radius, m in uq_results.clean_purification_results.items():
                results[f"eval/uq_clean_purify_acc/{radius}"] = m.accuracy_after_purify
            for n_sweeps, m in uq_results.clean_gibbs_purification_results.items():
                results[f"eval/gibbs_clean_purify_acc/{n_sweeps}"] = m.accuracy_after_purify
        except Exception as e:
            logger.warning(f"UQ failed: {e}")

    # Test-split rob (reuse UQ cache where possible)
    if eval_cfg.compute_rob and "test" in eval_cfg.splits:
        _uq_cache: Dict[float, float] = (
            {float(e): a for e, a in uq_results.adv_accuracies.items()}
            if uq_results is not None else {}
        )
        strengths = cfg.tracking.evasion.strengths
        missing = []
        for strength in strengths:
            if float(strength) in _uq_cache:
                results[f"eval/test/rob/{strength}"] = _uq_cache[float(strength)]
            else:
                missing.append(strength)
        if missing:
            try:
                OmegaConf.update(cfg, "tracking.evasion.strengths", missing, force_add=True)
                metric = MetricFactory.create("rob", 1, cfg, datahandler, device)
                result = metric.evaluate(bm, "test", {})
                for i, strength in enumerate(missing):
                    results[f"eval/test/rob/{strength}"] = float(result[i])
            except Exception as e:
                logger.warning(f"Fallback test rob: {e}")
                for strength in missing:
                    results[f"eval/test/rob/{strength}"] = np.nan

    del bm
    torch.cuda.empty_cache()
    return results


def evaluate_model_at_path(
    model_path: str,
    ref_run_dir: Path,
    eval_cfg: EvalConfig,
    sync_gen: bool = False,
) -> Dict[str, float]:
    """Full evaluate_run() but loads model from explicit path.

    Args:
        model_path: Direct path to BornMachine checkpoint (e.g. run_dir/models/cls).
        ref_run_dir: Run dir to read dataset/evasion config from.
        eval_cfg: Evaluation configuration.
        sync_gen: If True, call bm.sync_tensors(after="classification") before
                  gen_loss/fid (correct for cls-phase model). Set False for
                  gen-phase model whose gen tensors are already trained.
    """
    from src.models import BornMachine
    from src.data.handler import DataHandler
    from src.tracking.evaluator import MetricFactory

    ref_run_dir = Path(ref_run_dir)
    device = torch.device(eval_cfg.device)
    results: Dict[str, float] = {}

    # 1. Load config
    cfg = load_run_config(ref_run_dir)

    # 2. Apply evasion/sampling overrides
    _apply_config_overrides(cfg, eval_cfg.evasion_override, eval_cfg.sampling_override)

    # 3. Load model from explicit path
    bm = BornMachine.load(str(model_path))
    bm.to(device)

    # 4. Sync generator to match classifier tensors (needed for gen loss & FID)
    if sync_gen and (eval_cfg.compute_gen_loss or eval_cfg.compute_fid):
        bm.sync_tensors(after="classification")

    # 5. Ensure dataset is regenerated with correct seed
    OmegaConf.update(cfg, "dataset.overwrite", True, force_add=True)

    # 6. Load data
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bm)
    datahandler.get_classification_loaders()

    # 7. Evaluate non-rob metrics on all splits; rob on non-test splits only.
    non_rob_metrics = []
    if eval_cfg.compute_acc:
        non_rob_metrics.append("acc")
    if eval_cfg.compute_cls_loss:
        non_rob_metrics.append("clsloss")
    if eval_cfg.compute_gen_loss:
        non_rob_metrics.append("genloss")
    if eval_cfg.compute_fid:
        non_rob_metrics.append("fid")

    for split in eval_cfg.splits:
        context: Dict[str, Any] = {}
        for metric_name in non_rob_metrics:
            try:
                metric = MetricFactory.create(
                    metric_name=metric_name,
                    freq=1,
                    cfg=cfg,
                    datahandler=datahandler,
                    device=device,
                )
                result = metric.evaluate(bm, split, context)
                results[f"eval/{split}/{metric_name}"] = float(result)
            except Exception as e:
                logger.warning(f"Metric '{metric_name}' failed on split '{split}': {e}")
                results[f"eval/{split}/{metric_name}"] = np.nan

    # Valid-split rob
    if eval_cfg.compute_rob:
        for split in [s for s in eval_cfg.splits if s != "test"]:
            context: Dict[str, Any] = {}
            try:
                metric = MetricFactory.create(
                    metric_name="rob",
                    freq=1,
                    cfg=cfg,
                    datahandler=datahandler,
                    device=device,
                )
                result = metric.evaluate(bm, split, context)
                strengths = cfg.tracking.evasion.strengths
                for i, strength in enumerate(strengths):
                    results[f"eval/{split}/rob/{strength}"] = float(result[i])
            except Exception as e:
                logger.warning(f"Metric 'rob' failed on split '{split}': {e}")
                strengths = getattr(cfg.tracking.evasion, "strengths", [])
                for strength in strengths:
                    results[f"eval/{split}/rob/{strength}"] = np.nan

    # 8. MIA
    if eval_cfg.compute_mia:
        try:
            from .mia import MIAEvaluation, MIAFeatureConfig

            feature_kwargs = eval_cfg.mia_features or {}
            feature_config = MIAFeatureConfig(**feature_kwargs)
            mia_eval = MIAEvaluation(
                feature_config=feature_config,
                adversarial_strength=eval_cfg.mia_adversarial_strength,
                adversarial_num_steps=eval_cfg.mia_adversarial_num_steps,
                adversarial_step_size=eval_cfg.mia_adversarial_step_size,
                adversarial_norm=eval_cfg.mia_adversarial_norm,
            )
            mia_results = mia_eval.evaluate(
                bm,
                datahandler.classification["train"],
                datahandler.classification["test"],
                device,
            )
            results["eval/mia_accuracy"] = mia_results.attack_accuracy
            results["eval/mia_auc_roc"] = mia_results.auc_roc
        except Exception as e:
            logger.warning(f"MIA evaluation failed: {e}")
            results["eval/mia_accuracy"] = np.nan
            results["eval/mia_auc_roc"] = np.nan

    # 9. UQ (detection + purification)
    uq_results = None
    if eval_cfg.compute_uq:
        try:
            from .uq import UQEvaluation, UQConfig

            uq_kwargs = eval_cfg.uq_config or {}
            uq_config = UQConfig(**uq_kwargs)
            uq_eval = UQEvaluation(config=uq_config)
            uq_results = uq_eval.evaluate(
                bm,
                datahandler.classification["test"],
                device,
            )
            results["eval/uq_clean_accuracy"] = uq_results.clean_accuracy
            results["eval/uq_clean_log_px_mean"] = float(uq_results.clean_log_px.mean())

            for eps, acc in uq_results.adv_accuracies.items():
                results[f"eval/uq_adv_acc/{eps}"] = acc

            for (pct, eps), rate in uq_results.detection_rates.items():
                results[f"eval/uq_detection/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in uq_results.err_rate_detected.items():
                results[f"eval/uq_det_err_detected/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in uq_results.err_rate_passed.items():
                results[f"eval/uq_det_err_passed/{pct}pct/{eps}"] = rate

            for (eps, radius), metrics in uq_results.purification_results.items():
                results[f"eval/uq_purify_acc/{eps}/{radius}"] = metrics.accuracy_after_purify
                results[f"eval/uq_purify_recovery/{eps}/{radius}"] = metrics.recovery_rate

            for (eps, n_sweeps), metrics in uq_results.gibbs_purification_results.items():
                results[f"eval/gibbs_purify_acc/{eps}/{n_sweeps}"] = metrics.accuracy_after_purify
                results[f"eval/gibbs_purify_recovery/{eps}/{n_sweeps}"] = metrics.recovery_rate

            for radius, metrics in uq_results.clean_purification_results.items():
                results[f"eval/uq_clean_purify_acc/{radius}"] = metrics.accuracy_after_purify

            for n_sweeps, metrics in uq_results.clean_gibbs_purification_results.items():
                results[f"eval/gibbs_clean_purify_acc/{n_sweeps}"] = metrics.accuracy_after_purify
        except Exception as e:
            logger.warning(f"UQ evaluation failed: {e}")
            results["eval/uq_clean_accuracy"] = np.nan

    # 10. Test-split rob: reuse UQ's adv_accuracies where possible
    if eval_cfg.compute_rob and "test" in eval_cfg.splits:
        _uq_adv_acc_cache: Dict[float, float] = {}
        if eval_cfg.compute_uq and uq_results is not None:
            _uq_adv_acc_cache = {
                float(eps): acc for eps, acc in uq_results.adv_accuracies.items()
            }

        strengths = cfg.tracking.evasion.strengths
        missing = []
        for strength in strengths:
            s = float(strength)
            if s in _uq_adv_acc_cache:
                results[f"eval/test/rob/{strength}"] = _uq_adv_acc_cache[s]
            else:
                missing.append(strength)

        if missing:
            logger.warning(
                f"Test rob eps {missing} not in UQ cache; generating adversarial examples separately"
            )
            original_strengths = list(cfg.tracking.evasion.strengths)
            try:
                OmegaConf.update(cfg, "tracking.evasion.strengths", missing, force_add=True)
                context: Dict[str, Any] = {}
                metric = MetricFactory.create(
                    metric_name="rob",
                    freq=1,
                    cfg=cfg,
                    datahandler=datahandler,
                    device=device,
                )
                result = metric.evaluate(bm, "test", context)
                for i, strength in enumerate(missing):
                    results[f"eval/test/rob/{strength}"] = float(result[i])
            except Exception as e:
                logger.warning(f"Fallback test rob generation failed: {e}")
                for strength in missing:
                    results[f"eval/test/rob/{strength}"] = np.nan
            finally:
                OmegaConf.update(
                    cfg, "tracking.evasion.strengths", original_strengths, force_add=True
                )

    # 11. Cleanup
    del bm
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Sweep-level evaluation
# ---------------------------------------------------------------------------

def evaluate_sweep(
    sweep_dir: str,
    eval_cfg: EvalConfig,
    config_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Evaluate all runs in a sweep directory.

    Args:
        sweep_dir: Path to the sweep directory (contains numbered sub-dirs).
        eval_cfg: Evaluation configuration.
        config_keys: Hydra config keys to extract into the DataFrame
            (e.g. ``["dataset.name", "tracking.seed"]``).

    Returns:
        DataFrame with one row per run, containing config values and
        eval metric columns.
    """
    sweep_path = Path(sweep_dir)
    run_dirs = sorted(
        [d for d in sweep_path.iterdir() if d.is_dir() and (d / ".hydra" / "config.yaml").exists()],
        key=lambda d: d.name,
    )

    if not run_dirs:
        print(f"No valid run directories found in {sweep_path}")
        return pd.DataFrame()

    print(f"Found {len(run_dirs)} runs in {sweep_path}")

    rows = []
    for i, run_dir in enumerate(run_dirs):
        print(f"  [{i + 1}/{len(run_dirs)}] Evaluating {run_dir.name}...")

        row: Dict[str, Any] = {
            "run_name": run_dir.name,
            "run_path": str(run_dir),
        }

        # Extract config values
        if config_keys:
            try:
                cfg = load_run_config(run_dir)
                for key in config_keys:
                    try:
                        val = OmegaConf.select(cfg, key)
                        col_name = "config/" + key
                        row[col_name] = val
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Could not load config for {run_dir.name}: {e}")

        # Evaluate
        try:
            metrics = evaluate_run(run_dir, eval_cfg)
            row.update(metrics)
        except Exception as e:
            print(f"    FAILED: {e}")
            logger.warning(f"Run {run_dir.name} failed: {e}")

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nEvaluation complete. {len(df)} runs processed.")
    return df
