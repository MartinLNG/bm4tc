"""
Adversarial Training for Born Machine classifiers.

Implements two adversarial training methods:
- PGD-AT (Madry et al.): Train on adversarial examples
- TRADES (Zhang et al.): Clean loss + KL regularization for robustness
"""

import hydra
from pathlib import Path
import time
import torch
from typing import Dict, Optional
import src.utils.schemas as schemas
import src.utils.get as get
import wandb
from src.tracking import PerformanceEvaluator, log_grads, record
from src.data.handler import DataHandler
from src.models import BornMachine
from src.utils.evasion.minimal import ProjectedGradientDescent, FastGradientMethod

import logging
logger = logging.getLogger(__name__)


class Trainer:
    """
    Adversarial training trainer for BornMachine classifiers.

    Supports two methods:
    - PGD-AT: Replace inputs with adversarial examples, minimize L(x_adv, y)
    - TRADES: Minimize L(x, y) + beta * KL(p(x) || p(x_adv))

    Features:
    - Early stopping via patience counter
    - Curriculum training over epsilon (optional)
    - Mixed clean/adversarial training (optional, for PGD-AT)
    - W&B logging and checkpoint saving

    Attributes:
        best: Dict of best metric values achieved during training.
        best_tensors: Tensors from the best-performing epoch.
    """

    def __init__(
            self,
            bornmachine: BornMachine,
            cfg: schemas.Config,
            stage: str,
            datahandler: DataHandler,
            device: torch.device
    ):
        """
        Initialize the adversarial trainer.

        Args:
            bornmachine: BornMachine instance to train.
            cfg: Complete configuration object.
            stage: Training stage identifier (e.g., "adv").
            datahandler: DataHandler with loaded datasets.
            device: Torch device for training.
        """
        self.datahandler = datahandler
        self.device = device
        self.cfg = cfg
        self.stage = stage
        self.train_cfg = cfg.trainer.adversarial

        if self.datahandler.classification is None:
            self.datahandler.get_classification_loaders(batch_size=self.train_cfg.batch_size)

        # Validate method
        if self.train_cfg.method not in ["pgd_at", "trades"]:
            raise ValueError(f"Unknown adversarial training method: {self.train_cfg.method}")

        # Initialize W&B metrics
        wandb.define_metric(f"{stage}/train/loss", summary="none")
        wandb.define_metric(f"{stage}/train/epsilon", summary="none")
        if self.train_cfg.method == "pgd_at" and self.train_cfg.clean_weight > 0:
            wandb.define_metric(f"{stage}/train/clean_loss", summary="none")
            wandb.define_metric(f"{stage}/train/adv_loss", summary="none")
        if self.train_cfg.method == "trades":
            wandb.define_metric(f"{stage}/train/clean_loss", summary="none")
            wandb.define_metric(f"{stage}/train/kl_loss", summary="none")

        # Initialize evaluator and best performance tracking
        self.evaluator = PerformanceEvaluator(cfg, self.datahandler, self.train_cfg, self.device)
        self._best_perf_factory(self.train_cfg.metrics)

        # Born machine reference
        self.bornmachine = bornmachine
        self.best_tensors = [t.cpu().clone().detach() for t in self.bornmachine.classifier.tensors]

        # Initialize attack method
        self._init_attack()

    def _init_attack(self):
        """Initialize the attack method based on evasion config."""
        evasion = self.train_cfg.evasion

        if evasion.method == "PGD":
            self.attack = ProjectedGradientDescent(
                norm=evasion.norm,
                criterion=evasion.criterion,
                num_steps=evasion.num_steps,
                step_size=evasion.step_size,
                random_start=evasion.random_start
            )
        elif evasion.method == "FGM":
            self.attack = FastGradientMethod(
                norm=evasion.norm,
                criterion=evasion.criterion
            )
        else:
            raise ValueError(f"Unknown attack method: {evasion.method}")

        range_size = self.bornmachine.input_range[1] - self.bornmachine.input_range[0]
        self.range_size = range_size
        self.base_epsilon = (evasion.strengths[0] if evasion.strengths else 0.1) * range_size
        self._abs_curriculum_start = self.train_cfg.curriculum_start * range_size

    def _best_perf_factory(self, metrics: Dict[str, int]):
        """Initialize best performance tracking dict."""
        self.best = dict.fromkeys(metrics.keys())
        for metric_name in self.best.keys():
            if metric_name in ["acc", "rob"]:
                self.best[metric_name] = 0.0
            elif metric_name in ["clsloss", "genloss", "fid"]:
                self.best[metric_name] = float("Inf")

        self.stopping_criterion_name = self.train_cfg.stop_crit
        valid_criteria = ["clsloss", "genloss", "acc", "fid", "rob"]
        if self.stopping_criterion_name not in valid_criteria:
            raise ValueError(
                f"Invalid stop_crit '{self.stopping_criterion_name}'. "
                f"Must be one of: {valid_criteria}"
            )
        # Always track rob for averaging (even if not in metrics explicitly)
        if self.stopping_criterion_name == "rob" and "rob" not in self.best:
            self.best["rob"] = 0.0

    def _get_epsilon(self, epoch: int) -> float:
        """
        Get current epsilon based on curriculum schedule.

        If curriculum is enabled, linearly interpolate from curriculum_start
        to base_epsilon over curriculum_end_epoch epochs.
        """
        if not self.train_cfg.curriculum:
            return self.base_epsilon

        end_epoch = self.train_cfg.curriculum_end_epoch or self.train_cfg.max_epoch
        progress = min(1.0, epoch / end_epoch)

        start_eps = self._abs_curriculum_start
        return start_eps + progress * (self.base_epsilon - start_eps)

    def _generate_adversarial(
            self,
            data: torch.Tensor,
            labels: torch.Tensor,
            epsilon: float
    ) -> torch.Tensor:
        """Generate adversarial examples using the configured attack."""
        return self.attack.generate(
            born=self.bornmachine,
            naturals=data,
            labels=labels,
            strength=epsilon,
            device=self.device
        )

    def _compute_kl_divergence(
            self,
            clean_probs: torch.Tensor,
            adv_probs: torch.Tensor,
            eps: float = 1e-12
    ) -> torch.Tensor:
        """
        Compute KL divergence KL(clean || adv) for TRADES.

        KL(P || Q) = sum(P * log(P/Q))
        """
        clean_probs = clean_probs.clamp(min=eps)
        adv_probs = adv_probs.clamp(min=eps)
        return (clean_probs * (clean_probs.log() - adv_probs.log())).sum(dim=1).mean()

    def _train_epoch_pgd_at(self, epsilon: float):
        """
        Execute one epoch of PGD-AT training.

        Loss = (1 - clean_weight) * L(x_adv, y) + clean_weight * L(x, y)
        """
        losses = []
        self.bornmachine.classifier.train()

        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            # Generate adversarial examples (eval mode for attack generation)
            self.bornmachine.classifier.eval()
            adv_data = self._generate_adversarial(data, labels, epsilon)
            self.bornmachine.classifier.train()

            # Forward pass on adversarial examples
            adv_probs = self.bornmachine.class_probabilities(adv_data)
            adv_loss = self.criterion(adv_probs, labels)

            # Optionally mix with clean loss
            if self.train_cfg.clean_weight > 0:
                clean_probs = self.bornmachine.class_probabilities(data)
                clean_loss = self.criterion(clean_probs, labels)
                loss = (1 - self.train_cfg.clean_weight) * adv_loss + \
                       self.train_cfg.clean_weight * clean_loss
            else:
                loss = adv_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            log_grads(bm_view=self.bornmachine.classifier, watch_freq=self.train_cfg.watch_freq,
                      step=self.step, stage=self.stage)
            self.optimizer.step()

            losses.append(loss.detach().cpu().item())

        avg_loss = sum(losses) / len(losses)
        wandb.log({f"{self.stage}/train/loss": avg_loss, f"{self.stage}/train/epsilon": epsilon})

    def _train_epoch_trades(self, epsilon: float):
        """
        Execute one epoch of TRADES training.

        Loss = L(x, y) + beta * KL(p(x) || p(x_adv))
        """
        total_losses, clean_losses, kl_losses = [], [], []
        self.bornmachine.classifier.train()
        beta = self.train_cfg.trades_beta

        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            # Clean forward pass (needed for both clean loss and KL)
            clean_probs = self.bornmachine.class_probabilities(data)
            clean_loss = self.criterion(clean_probs, labels)

            # Generate adversarial examples
            self.bornmachine.classifier.eval()
            adv_data = self._generate_adversarial(data, labels, epsilon)
            self.bornmachine.classifier.train()

            # Adversarial forward pass
            adv_probs = self.bornmachine.class_probabilities(adv_data)

            # KL divergence term (use detached clean_probs as reference)
            kl_loss = self._compute_kl_divergence(clean_probs.detach(), adv_probs)

            # Combined TRADES loss
            loss = clean_loss + beta * kl_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            log_grads(bm_view=self.bornmachine.classifier, watch_freq=self.train_cfg.watch_freq,
                      step=self.step, stage=self.stage)
            self.optimizer.step()

            total_losses.append(loss.detach().cpu().item())
            clean_losses.append(clean_loss.detach().cpu().item())
            kl_losses.append(kl_loss.detach().cpu().item())

        wandb.log({
            f"{self.stage}/train/loss": sum(total_losses) / len(total_losses),
            f"{self.stage}/train/clean_loss": sum(clean_losses) / len(clean_losses),
            f"{self.stage}/train/kl_loss": sum(kl_losses) / len(kl_losses),
            f"{self.stage}/train/epsilon": epsilon
        })

    def _train_epoch(self, epsilon: float):
        """Dispatch to appropriate training method."""
        if self.train_cfg.method == "pgd_at":
            self._train_epoch_pgd_at(epsilon)
        elif self.train_cfg.method == "trades":
            self._train_epoch_trades(epsilon)

    def _update(self):
        """
        Check if model improved on validation set.
        Update best tensors and patience counter accordingly.

        For rob metric: averages all rob/{strength} values since robustness is evaluated
        at multiple perturbation strengths.
        """
        # Handle rob specially: average all rob/{strength} values
        if self.stopping_criterion_name == "rob" or self.stopping_criterion_name.startswith("rob/"):
            rob_values = [v for k, v in self.valid_perf.items()
                         if k.startswith("rob/") and isinstance(v, (int, float))]
            current_value = sum(rob_values) / len(rob_values) if rob_values else None
        else:
            current_value = self.valid_perf.get(self.stopping_criterion_name)

        if current_value is None:
            return

        former_best = self.best.get(
            self.stopping_criterion_name,
            0.0 if self.stopping_criterion_name in ["acc", "rob"] else float("Inf")
        )

        # Check whether the monitored metric improved
        if self.stopping_criterion_name in ["acc", "rob"] or self.stopping_criterion_name.startswith("rob"):
            improved = current_value > former_best
        elif self.stopping_criterion_name in ["clsloss", "genloss", "fid"]:
            improved = current_value < former_best
        else:
            raise ValueError(f"Unknown stopping criterion: {self.stopping_criterion_name}")

        # Check if we reached target (optional shortcut)
        goal_key = list(self.goal.keys())[0] if self.goal else None
        if self.goal is None:
            reached_goal = False
        elif goal_key == "rob":
            # Handle rob goal by averaging rob/* values
            rob_values = [v for k, v in self.valid_perf.items()
                         if k.startswith("rob/") and isinstance(v, (int, float))]
            goal_value = sum(rob_values) / len(rob_values) if rob_values else 0.0
            reached_goal = goal_value > self.goal["rob"]
        elif goal_key in ["acc"]:
            reached_goal = self.valid_perf.get(goal_key, 0.0) > self.goal[goal_key]
        elif goal_key in ["clsloss", "genloss", "fid"]:
            reached_goal = self.valid_perf.get(goal_key, float("Inf")) < self.goal[goal_key]
        else:
            reached_goal = False

        is_better = improved or reached_goal

        if is_better:
            self.best = dict(self.valid_perf)
            # Store averaged rob value if using rob as stopping criterion
            if self.stopping_criterion_name == "rob" or self.stopping_criterion_name.startswith("rob/"):
                self.best["rob"] = current_value
            self.best_tensors = [t.clone().detach() for t in self.bornmachine.classifier.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0

            if reached_goal:
                self.patience_counter = self.train_cfg.patience + 1
                logger.info("Goal reached.")
        else:
            self.patience_counter += 1

    def _summarise_training(self):
        """
        Restore best tensors, evaluate on test set, log summaries, optionally save.
        """
        # Restore classifier to best tensors
        self.bornmachine.classifier.prepare(tensors=self.best_tensors, device=self.device,
                                            train_cfg=self.train_cfg)
        self.bornmachine.sync_tensors(after="classification", verify=True)
        self.bornmachine.to(self.device)

        # Evaluate on test set
        test_results = self.evaluator.evaluate(self.bornmachine, "test", self.epoch)

        # Log summaries to W&B
        for metric_name in ["acc", "clsloss"]:
            if metric_name in test_results:
                wandb.summary[f"{self.stage}/test/{metric_name}"] = test_results[metric_name]
            if metric_name in self.best:
                wandb.summary[f"{self.stage}/valid/{metric_name}"] = self.best[metric_name]

        # Log robustness metrics
        for key, value in test_results.items():
            if key.startswith("rob/"):
                wandb.summary[f"{self.stage}/test/{key}"] = value
        for key, value in self.best.items():
            if key is not None and key.startswith("rob/"):
                wandb.summary[f"{self.stage}/valid/{key}"] = value

        wandb.summary[f"{self.stage}/epoch/best"] = self.best_epoch
        wandb.summary[f"{self.stage}/epoch/last"] = self.epoch

        if self.epoch_times:
            wandb.summary[f"{self.stage}/avg_epoch_time_s"] = sum(self.epoch_times) / len(self.epoch_times)

        self.bornmachine.reset()
        self.bornmachine.to("cpu")
        del self.valid_perf

        # Save model if configured
        if self.train_cfg.save:
            run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            folder = run_dir / "models"
            folder.mkdir(parents=True, exist_ok=True)

            filename = "adv"

            save_path = folder / filename
            self.bornmachine.save(path=str(save_path))
            if wandb.run is not None and not wandb.run.disabled:
                wandb.log_model(str(save_path))

        logger.info(f"Adversarial Trainer ({self.train_cfg.method}) finished.")

    def train(self, goal: Optional[Dict[str, float]] = None):
        """
        Run the adversarial training loop.

        Args:
            goal: Optional target metrics to reach early (e.g., {"acc": 0.95}).
        """
        self.step = 0
        self.patience_counter = 0
        self.goal = goal
        self.best_epoch = 0
        self.epoch_times = []

        # Prepare classifier
        self.bornmachine.classifier.prepare(device=self.device, train_cfg=self.train_cfg)
        self.criterion = get.criterion("classification", self.train_cfg.criterion)
        self.optimizer = get.optimizer(self.bornmachine.classifier.parameters(),
                                       self.train_cfg.optimizer)

        logger.info(f"Adversarial training ({self.train_cfg.method}) begins.")

        for epoch in range(self.train_cfg.max_epoch):
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1

            # Get current epsilon (handles curriculum)
            epsilon = self._get_epsilon(self.epoch)

            # Train epoch
            self._train_epoch(epsilon)

            # Sync tensors for evaluation
            self.bornmachine.sync_tensors(after="classification", verify=False)

            # Evaluate on validation set
            self.valid_perf = self.evaluator.evaluate(self.bornmachine, "valid", epoch)
            record(results=self.valid_perf, stage=self.stage, set="valid")

            # Update best and check early stopping
            self._update()
            self.epoch_times.append(time.perf_counter() - epoch_start)

            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training()
