"""
Uncertainty Quantification (UQ) evaluation for Born Machines.

Born Machines learn the joint distribution p(x,c), enabling computation of
the marginal input likelihood p(x) = sum_c p(x,c). This provides two defense
mechanisms against adversarial examples:

1. **Detection**: Reject inputs whose likelihood falls below a threshold tau
2. **Purification**: For rejected inputs, find a nearby point x* maximizing
   likelihood within a perturbation ball, then classify x* instead

This module provides tools to evaluate both defenses by:
- Computing log p(x) on clean and adversarial data
- Calibrating detection thresholds from clean data percentiles
- Purifying adversarial examples and measuring accuracy recovery
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


@dataclass
class UQConfig:
    """Configuration for UQ evaluation.

    Attributes:
        norm: Lp norm for purification perturbation ball.
        num_steps: Gradient descent iterations for purification.
        step_size: Step size per iteration (None = auto).
        radii: List of purification radii to evaluate.
        eps: Clamping floor for log p(x) stability.
        percentiles: Percentiles of clean log p(x) for threshold candidates.
        attack_method: Attack method for generating adversarial inputs.
        attack_strengths: List of attack epsilons.
        attack_num_steps: PGD steps for attack generation.
        random_start: Random start for purification.
    """
    # Purification params
    norm: int | str = "inf"
    num_steps: int = 20
    step_size: float | None = None
    radii: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    eps: float = 1e-12
    random_start: bool = False

    # Threshold params
    percentiles: List[float] = field(default_factory=lambda: [1, 5, 10, 20])

    # Attack params
    attack_method: str = "PGD"
    attack_strengths: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    attack_num_steps: int = 20

    # Gibbs purification params
    run_gibbs: bool = False
    gibbs_n_sweeps: List[int] = field(default_factory=lambda: [1, 3, 5])
    gibbs_num_bins: int = 200
    gibbs_batch_size: int = 8
    gibbs_radius: Optional[float] = 0.1  # relative to input range; None = unrestricted


def compute_log_px(
    born,
    loader: DataLoader,
    device: torch.device,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute marginal log p(x) for all samples in a loader.

    Args:
        born: BornMachine instance.
        loader: DataLoader yielding (data, labels) tuples.
        device: Torch device.
        eps: Clamping floor for log stability.

    Returns:
        Tuple of (log_px, labels) tensors concatenated over all batches.
    """
    all_log_px = []
    all_labels = []

    born.to(device)

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            log_px = born.marginal_log_probability(batch_data, eps=eps)
            all_log_px.append(log_px.cpu())
            all_labels.append(batch_labels)

    return torch.cat(all_log_px), torch.cat(all_labels)


def compute_thresholds(
    born,
    clean_loader: DataLoader,
    percentiles: List[float],
    device: torch.device,
    eps: float = 1e-12,
) -> Tuple[Dict[float, float], torch.Tensor]:
    """Compute percentile-based detection thresholds from clean data.

    Standard approach from OOD detection literature: thresholds are set
    at percentiles of the clean data's log p(x) distribution.

    Args:
        born: BornMachine instance.
        clean_loader: DataLoader for clean (in-distribution) data.
        percentiles: List of percentile values (e.g., [1, 5, 10, 20]).
        device: Torch device.
        eps: Clamping floor for log stability.

    Returns:
        Tuple of:
            - Dict mapping percentile -> threshold value.
            - Tensor of all clean log p(x) values.
    """
    clean_log_px, _ = compute_log_px(born, clean_loader, device, eps)

    thresholds = {}
    for p in percentiles:
        thresholds[p] = float(np.percentile(clean_log_px.numpy(), p))

    return thresholds, clean_log_px


@dataclass
class PurificationMetrics:
    """Metrics for a single (epsilon, radius) purification evaluation.

    Attributes:
        accuracy_after_purify: Classification accuracy on purified samples.
        recovery_rate: Fraction of correctly classified after purification
            among those misclassified before purification.
        mean_log_px_before: Mean log p(x) of adversarial inputs.
        mean_log_px_after: Mean log p(x) of purified inputs.
        rejection_rate: Fraction of inputs below threshold after purification.
    """
    accuracy_after_purify: float
    recovery_rate: float
    mean_log_px_before: float
    mean_log_px_after: float
    rejection_rate: float


@dataclass
class UQResults:
    """Complete UQ evaluation results.

    Attributes:
        clean_log_px: Log p(x) values for clean test data.
        clean_accuracy: Clean classification accuracy.
        thresholds: Dict mapping percentile -> threshold value.
        adv_log_px: Dict mapping epsilon -> log p(x) values for adversarial data.
        adv_accuracies: Dict mapping epsilon -> adversarial accuracy.
        detection_rates: Dict mapping (percentile, epsilon) -> detection rate.
        purification_results: Dict mapping (epsilon, radius) -> PurificationMetrics.
    """
    clean_log_px: np.ndarray
    clean_accuracy: float
    thresholds: Dict[float, float]
    adv_log_px: Dict[float, np.ndarray]
    adv_accuracies: Dict[float, float]
    detection_rates: Dict[Tuple[float, float], float]
    purification_results: Dict[Tuple[float, float], PurificationMetrics]
    gibbs_purification_results: Dict[Tuple[float, int], PurificationMetrics] = field(
        default_factory=dict
    )
    clean_purification_results: Dict[float, PurificationMetrics] = field(
        default_factory=dict
    )
    clean_gibbs_purification_results: Dict[int, PurificationMetrics] = field(
        default_factory=dict
    )

    def summary(self) -> str:
        """Return a formatted summary of UQ evaluation results."""
        lines = [
            "=" * 60,
            "Uncertainty Quantification Results",
            "=" * 60,
            f"Clean Accuracy: {self.clean_accuracy:.4f}",
            f"Clean log p(x): mean={self.clean_log_px.mean():.2f}, "
            f"std={self.clean_log_px.std():.2f}",
            "",
            "--- Detection Thresholds ---",
        ]
        for pct, tau in sorted(self.thresholds.items()):
            lines.append(f"  {pct}th percentile: tau = {tau:.4f}")

        lines.extend(["", "--- Adversarial Results ---"])
        for eps in sorted(self.adv_accuracies.keys()):
            adv_lp = self.adv_log_px[eps]
            lines.append(
                f"  eps={eps}: acc={self.adv_accuracies[eps]:.4f}, "
                f"mean log p(x)={adv_lp.mean():.2f}"
            )

        lines.extend(["", "--- Detection Rates ---"])
        for (pct, eps), rate in sorted(self.detection_rates.items()):
            lines.append(f"  tau={pct}th pct, eps={eps}: {rate:.2%} detected")

        lines.extend(["", "--- Purification Results ---"])
        for (eps, radius), metrics in sorted(self.purification_results.items()):
            lines.append(
                f"  eps={eps}, radius={radius}: "
                f"acc={metrics.accuracy_after_purify:.4f}, "
                f"recovery={metrics.recovery_rate:.2%}, "
                f"log p(x) {metrics.mean_log_px_before:.2f} -> {metrics.mean_log_px_after:.2f}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class UQEvaluation:
    """Main class for running UQ evaluation.

    Evaluates both detection and purification defenses against adversarial
    examples, using the Born Machine's marginal likelihood p(x).

    Example:
        >>> uq_eval = UQEvaluation(uq_config)
        >>> results = uq_eval.evaluate(born, test_loader, device)
        >>> print(results.summary())
    """

    def __init__(self, config: Optional[UQConfig] = None):
        """Initialize UQ evaluation.

        Args:
            config: UQ evaluation configuration. Uses defaults if None.
        """
        self.config = config or UQConfig()

    def evaluate(
        self,
        born,
        clean_loader: DataLoader,
        device: torch.device,
    ) -> UQResults:
        """Run the full UQ evaluation pipeline.

        Steps:
        1. Cache log Z on the Born Machine
        2. Compute clean log p(x) and derive detection thresholds
        3. For each attack epsilon: generate adversarial examples,
           compute log p(x_adv), detection rate
        4. For each (epsilon, radius): purify adversarial examples,
           classify, compute metrics
        5. Package into UQResults

        Args:
            born: BornMachine instance.
            clean_loader: DataLoader for clean test data.
            device: Torch device.

        Returns:
            UQResults with all evaluation metrics.
        """
        from src.utils.evasion.minimal import RobustnessEvaluation
        from src.utils.schemas import CriterionConfig
        from src.utils.purification.minimal import LikelihoodPurification

        cfg = self.config
        born.to(device)

        # 1. Cache log Z
        logger.info("Computing partition function...")
        born.cache_log_Z()

        # 2. Compute clean log p(x) and thresholds
        logger.info("Computing clean log p(x) and thresholds...")
        thresholds, clean_log_px_tensor = compute_thresholds(
            born, clean_loader, cfg.percentiles, device, cfg.eps
        )
        clean_log_px = clean_log_px_tensor.numpy()

        # Compute clean accuracy
        clean_correct = 0
        clean_total = 0
        with torch.no_grad():
            for batch_data, batch_labels in clean_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                probs = born.class_probabilities(batch_data)
                preds = probs.argmax(dim=1)
                clean_correct += (preds == batch_labels).sum().item()
                clean_total += len(batch_labels)
        clean_accuracy = clean_correct / clean_total
        logger.info(f"Clean accuracy: {clean_accuracy:.4f}")

        # 3. Generate adversarial examples and evaluate detection
        attack = RobustnessEvaluation(
            method=cfg.attack_method,
            norm=cfg.norm,
            criterion=CriterionConfig(name="nll", kwargs=None),
            strengths=cfg.attack_strengths,
            num_steps=cfg.attack_num_steps,
            random_start=True,
        )

        adv_log_px: Dict[float, np.ndarray] = {}
        adv_accuracies: Dict[float, float] = {}
        detection_rates: Dict[Tuple[float, float], float] = {}
        # Store adversarial examples for purification
        adv_examples_cache: Dict[float, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        for eps in cfg.attack_strengths:
            logger.info(f"Generating adversarial examples (eps={eps})...")
            all_adv_log_px = []
            all_adv_correct = 0
            all_adv_total = 0
            adv_batches = []

            for batch_data, batch_labels in clean_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                # Generate adversarial examples
                adv_data = attack.generate(
                    born, batch_data, batch_labels, eps, device
                )

                # Classify adversarial examples
                with torch.no_grad():
                    adv_probs = born.class_probabilities(adv_data)
                    adv_preds = adv_probs.argmax(dim=1)
                    all_adv_correct += (adv_preds == batch_labels).sum().item()
                    all_adv_total += len(batch_labels)

                    # Compute log p(x_adv)
                    log_px_adv = born.marginal_log_probability(adv_data, eps=cfg.eps)
                    all_adv_log_px.append(log_px_adv.cpu())

                adv_batches.append((adv_data.detach().cpu(), batch_labels.cpu()))

            adv_log_px_arr = torch.cat(all_adv_log_px).numpy()
            adv_log_px[eps] = adv_log_px_arr
            adv_accuracies[eps] = all_adv_correct / all_adv_total
            adv_examples_cache[eps] = adv_batches

            logger.info(
                f"  eps={eps}: adv_acc={adv_accuracies[eps]:.4f}, "
                f"mean log p(x_adv)={adv_log_px_arr.mean():.2f}"
            )

            # Detection rates at each threshold
            for pct, tau in thresholds.items():
                detected = (adv_log_px_arr < tau).mean()
                detection_rates[(pct, eps)] = float(detected)

        # 4. Purification
        purifier = LikelihoodPurification(
            norm=cfg.norm,
            num_steps=cfg.num_steps,
            step_size=cfg.step_size,
            random_start=cfg.random_start,
            eps=cfg.eps,
        )

        purification_results: Dict[Tuple[float, float], PurificationMetrics] = {}

        for eps in cfg.attack_strengths:
            for radius in cfg.radii:
                logger.info(f"Purifying (eps={eps}, radius={radius})...")
                all_purified_correct = 0
                all_recovered = 0
                all_misclassified_before = 0
                all_log_px_before = []
                all_log_px_after = []
                all_total = 0
                all_below_threshold = 0

                # Use median threshold for rejection rate
                median_pct = cfg.percentiles[len(cfg.percentiles) // 2]
                tau = thresholds[median_pct]

                for adv_data_cpu, labels_cpu in adv_examples_cache[eps]:
                    adv_data = adv_data_cpu.to(device)
                    labels = labels_cpu.to(device)

                    # Log p(x) before purification
                    with torch.no_grad():
                        log_px_before = born.marginal_log_probability(
                            adv_data, eps=cfg.eps
                        )
                        # Classify before purification
                        adv_probs = born.class_probabilities(adv_data)
                        adv_preds = adv_probs.argmax(dim=1)
                        misclassified = (adv_preds != labels)

                    # Purify
                    purified, log_px_after = purifier.purify(
                        born, adv_data, radius, device
                    )

                    # Classify after purification
                    with torch.no_grad():
                        pur_probs = born.class_probabilities(purified)
                        pur_preds = pur_probs.argmax(dim=1)
                        correct_after = (pur_preds == labels)

                    # Recovery: correctly classified after purification
                    # among those misclassified before
                    recovered = (misclassified & correct_after).sum().item()

                    all_purified_correct += correct_after.sum().item()
                    all_recovered += recovered
                    all_misclassified_before += misclassified.sum().item()
                    all_log_px_before.append(log_px_before.cpu())
                    all_log_px_after.append(log_px_after.cpu())
                    all_total += len(labels)
                    all_below_threshold += (log_px_after.cpu() < tau).sum().item()

                acc_after = all_purified_correct / all_total
                recovery = (
                    all_recovered / all_misclassified_before
                    if all_misclassified_before > 0
                    else 1.0
                )
                mean_before = torch.cat(all_log_px_before).mean().item()
                mean_after = torch.cat(all_log_px_after).mean().item()
                rejection_rate = all_below_threshold / all_total

                purification_results[(eps, radius)] = PurificationMetrics(
                    accuracy_after_purify=acc_after,
                    recovery_rate=recovery,
                    mean_log_px_before=mean_before,
                    mean_log_px_after=mean_after,
                    rejection_rate=rejection_rate,
                )

                logger.info(
                    f"  eps={eps}, r={radius}: "
                    f"acc={acc_after:.4f}, recovery={recovery:.2%}"
                )

        # 5. Clean purification (natural examples, no attack)
        clean_purification_results: Dict[float, PurificationMetrics] = {}
        for radius in cfg.radii:
            logger.info(f"Clean purification (radius={radius})...")
            all_correct = 0
            all_total = 0
            all_log_px_before = []
            all_log_px_after = []

            for batch_data, batch_labels in clean_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                with torch.no_grad():
                    log_px_before = born.marginal_log_probability(batch_data, eps=cfg.eps)

                purified, log_px_after = purifier.purify(born, batch_data, radius, device)

                with torch.no_grad():
                    preds = born.class_probabilities(purified).argmax(dim=1)
                    all_correct += (preds == batch_labels).sum().item()
                    all_total += len(batch_labels)

                all_log_px_before.append(log_px_before.cpu())
                all_log_px_after.append(log_px_after.cpu())

            acc = all_correct / all_total
            clean_purification_results[radius] = PurificationMetrics(
                accuracy_after_purify=acc,
                recovery_rate=float("nan"),
                mean_log_px_before=torch.cat(all_log_px_before).mean().item(),
                mean_log_px_after=torch.cat(all_log_px_after).mean().item(),
                rejection_rate=0.0,
            )
            logger.info(f"  radius={radius}: clean_purify_acc={acc:.4f}")

        # 6. Gibbs purification
        gibbs_purification_results: Dict[Tuple[float, int], PurificationMetrics] = {}
        clean_gibbs_purification_results: Dict[int, PurificationMetrics] = {}

        if cfg.run_gibbs:
            from src.utils.purification.gibbs import GibbsPurification

            gibbs_purifier = GibbsPurification(
                num_bins=cfg.gibbs_num_bins,
                gibbs_batch_size=cfg.gibbs_batch_size,
                radius=cfg.gibbs_radius,
            )
            for eps in cfg.attack_strengths:
                all_adv = torch.cat([b[0] for b in adv_examples_cache[eps]])
                all_labels = torch.cat([b[1] for b in adv_examples_cache[eps]])

                with torch.no_grad():
                    adv_probs = born.class_probabilities(all_adv.to(device))
                    adv_preds = adv_probs.argmax(dim=1).cpu()
                    misclassified = adv_preds != all_labels

                for n_sw in cfg.gibbs_n_sweeps:
                    logger.info(f"Gibbs purification (eps={eps}, n_sweeps={n_sw})...")
                    x_purified, log_px_after = gibbs_purifier.purify(
                        born, all_adv, n_sw, device
                    )

                    with torch.no_grad():
                        pur_probs = born.class_probabilities(x_purified.to(device))
                        pur_preds = pur_probs.argmax(dim=1).cpu()
                    correct_after = pur_preds == all_labels
                    acc_after = correct_after.float().mean().item()
                    misclassified_before = misclassified.sum().item()
                    recovered = (misclassified & correct_after).sum().item()
                    recovery = (
                        recovered / misclassified_before
                        if misclassified_before > 0
                        else 1.0
                    )
                    gibbs_purification_results[(eps, n_sw)] = PurificationMetrics(
                        accuracy_after_purify=acc_after,
                        recovery_rate=recovery,
                        mean_log_px_before=float(adv_log_px[eps].mean()),
                        mean_log_px_after=float(log_px_after.mean()),
                        rejection_rate=0.0,
                    )
                    logger.info(
                        f"  eps={eps}, sweeps={n_sw}: "
                        f"acc={acc_after:.4f}, recovery={recovery:.2%}"
                    )

            # Clean Gibbs purification
            all_clean = torch.cat([b for b, _ in clean_loader])
            all_clean_labels = torch.cat([lb for _, lb in clean_loader])
            for n_sw in cfg.gibbs_n_sweeps:
                logger.info(f"Clean Gibbs purification (n_sweeps={n_sw})...")
                x_purified, log_px_after = gibbs_purifier.purify(
                    born, all_clean, n_sw, device
                )
                with torch.no_grad():
                    pur_probs = born.class_probabilities(x_purified.to(device))
                    pur_preds = pur_probs.argmax(dim=1).cpu()
                acc = (pur_preds == all_clean_labels).float().mean().item()
                clean_gibbs_purification_results[n_sw] = PurificationMetrics(
                    accuracy_after_purify=acc,
                    recovery_rate=float("nan"),
                    mean_log_px_before=float(clean_log_px.mean()),
                    mean_log_px_after=float(log_px_after.mean()),
                    rejection_rate=0.0,
                )
                logger.info(f"  sweeps={n_sw}: clean_gibbs_acc={acc:.4f}")

        return UQResults(
            clean_log_px=clean_log_px,
            clean_accuracy=clean_accuracy,
            thresholds=thresholds,
            adv_log_px=adv_log_px,
            adv_accuracies=adv_accuracies,
            detection_rates=detection_rates,
            purification_results=purification_results,
            gibbs_purification_results=gibbs_purification_results,
            clean_purification_results=clean_purification_results,
            clean_gibbs_purification_results=clean_gibbs_purification_results,
        )
