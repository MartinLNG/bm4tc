# Minimal implementation of evasion attacks against Born Machines

import torch
from torch.utils.data import DataLoader
from typing import List
import src.utils.get as get
from src.models import *
from src.utils.schemas import CriterionConfig


def normalizing(x: torch.FloatTensor, norm: int | str):
    """
    Normalize a tensor of shape (batch size, data dim)
    along the data dim (flattened).
    """
    if norm == "inf":
        normalized = x.sign()

    elif isinstance(norm, int):
        if norm < 1:
            raise ValueError("Only accept p >= 1.")
        x_norm = x.norm(p=norm, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-12)
        normalized = x / x_norm

    else:
        raise ValueError(f"{norm=}, but expected to be int or 'inf'.")
    
    return normalized

class FastGradientMethod:
    """
    Fast Gradient Method (FGM) adversarial attack.

    Single-step attack that perturbs inputs in the direction of the loss gradient,
    normalized according to the specified Lp norm.
    """

    def __init__(
            self,
            norm: int | str = "inf",
            criterion: CriterionConfig = CriterionConfig(name="nll", kwargs=None)
    ):
        """
        Initialize FGM with chosen norm and loss function.

        Args:
            norm: Lp norm for gradient normalization ("inf" or int >= 1).
            criterion: Loss function configuration.
        """
        self.norm = norm
        self.criterion = get.criterion("classification", criterion)

    def generate(
            self,
            born: BornMachine,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            strength: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """
        Generate adversarial examples from natural examples
        based on a given criterion.
        """
        born.to(device)
        naturals = naturals.to(device).detach().clone().requires_grad_(True)
        labels = labels.to(device)

        # Forward and backward pass
        probabilities = born.classifier.probabilities(naturals)
        loss = self.criterion(probabilities, labels)

        born.classifier.zero_grad()
        if naturals.grad is not None:
            naturals.grad.zero_()

        loss.backward()

        grad = naturals.grad.detach()  # shape: (batch size, data dim)
        normalized_gradient = normalizing(grad, norm=self.norm)

        ad_examples = naturals + strength * normalized_gradient
        ad_examples = ad_examples.detach()

        return ad_examples


class ProjectedGradientDescent:
    """
    Projected Gradient Descent (PGD) adversarial attack.

    Iterative attack that performs multiple gradient ascent steps with projection
    back onto the epsilon ball. Stronger than FGM but more expensive.
    """

    def __init__(
            self,
            norm: int | str = "inf",
            criterion: CriterionConfig = CriterionConfig(name="nll", kwargs=None),
            num_steps: int = 10,
            step_size: float | None = None,
            random_start: bool = True
    ):
        """
        Initialize PGD with chosen norm, loss function, and iteration parameters.

        Args:
            norm: Lp norm for perturbation ball ("inf" or int >= 1).
            criterion: Loss function configuration.
            num_steps: Number of gradient ascent iterations.
            step_size: Step size per iteration. If None, defaults to 2.5 * strength / num_steps.
            random_start: Whether to start from random point within epsilon ball.
        """
        self.norm = norm
        self.criterion = get.criterion("classification", criterion)
        self.num_steps = num_steps if num_steps is not None else 10
        self.step_size = step_size
        self.random_start = random_start

    def _project(self, perturbation: torch.Tensor, strength: float) -> torch.Tensor:
        """Project perturbation back into the epsilon ball."""
        if self.norm == "inf":
            return perturbation.clamp(-strength, strength)
        elif isinstance(self.norm, int):
            # Project onto Lp ball
            norms = perturbation.norm(p=self.norm, dim=1, keepdim=True)
            scale = torch.clamp(norms / strength, min=1.0)
            return perturbation / scale
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _random_init(self, shape: torch.Size, strength: float, device: torch.device) -> torch.Tensor:
        """Initialize random perturbation within epsilon ball."""
        if self.norm == "inf":
            return (2 * torch.rand(shape, device=device) - 1) * strength
        elif isinstance(self.norm, int):
            # Sample uniformly from Lp ball (approximate via normalize + scale)
            delta = torch.randn(shape, device=device)
            delta = normalizing(delta, self.norm) * strength * torch.rand(shape[0], 1, device=device)
            return delta
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def generate(
            self,
            born: BornMachine,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            strength: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """
        Generate adversarial examples using PGD.
        """
        born.to(device)
        naturals = naturals.to(device).detach()
        labels = labels.to(device)

        step_size = self.step_size if self.step_size is not None else 2.5 * strength / self.num_steps

        # Initialize perturbation
        if self.random_start:
            delta = self._random_init(naturals.shape, strength, device)
        else:
            delta = torch.zeros_like(naturals)

        # Iterative gradient ascent
        for _ in range(self.num_steps):
            delta.requires_grad_(True)
            ad_examples = naturals + delta

            probabilities = born.classifier.probabilities(ad_examples)
            loss = self.criterion(probabilities, labels)

            born.classifier.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()

            loss.backward()

            grad = delta.grad.detach()
            normalized_gradient = normalizing(grad, norm=self.norm)

            # Gradient ascent step
            delta = delta.detach() + step_size * normalized_gradient
            # Project back into epsilon ball
            delta = self._project(delta, strength)

        ad_examples = naturals + delta
        return ad_examples.detach()


class JointProjectedGradientDescent:
    """PGD maximising max_{c'≠c} ln|ψ(x̃, c')|²  (joint generative attack).

    Loss per step: +mean( max_{c'≠c}  2·log|ψ(x̃, c')| )  — gradient ascent.
    The worst-case wrong class is re-selected dynamically at every gradient step.
    """

    def __init__(
            self,
            norm: int | str = "inf",
            num_steps: int = 10,
            step_size: float | None = None,
            random_start: bool = True,
            eps: float = 1e-12,
    ):
        self.norm = norm
        self.num_steps = num_steps if num_steps is not None else 10
        self.step_size = step_size
        self.random_start = random_start
        self.eps = eps

    def _project(self, perturbation: torch.Tensor, strength: float) -> torch.Tensor:
        if self.norm == "inf":
            return perturbation.clamp(-strength, strength)
        elif isinstance(self.norm, int):
            norms = perturbation.norm(p=self.norm, dim=1, keepdim=True)
            scale = torch.clamp(norms / strength, min=1.0)
            return perturbation / scale
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _random_init(self, shape: torch.Size, strength: float, device: torch.device) -> torch.Tensor:
        if self.norm == "inf":
            return (2 * torch.rand(shape, device=device) - 1) * strength
        elif isinstance(self.norm, int):
            delta = torch.randn(shape, device=device)
            delta = normalizing(delta, self.norm) * strength * torch.rand(shape[0], 1, device=device)
            return delta
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def generate(
            self,
            born: BornMachine,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            strength: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """Generate adversarial examples using the joint generative attack."""
        born.to(device)
        naturals = naturals.to(device).detach()
        labels   = labels.to(device)

        step_size = self.step_size if self.step_size is not None \
                    else 2.5 * strength / self.num_steps

        delta = (self._random_init(naturals.shape, strength, device)
                 if self.random_start else torch.zeros_like(naturals))

        batch = len(labels)
        K = born.out_dim
        true_class_mask = torch.zeros(batch, K, dtype=torch.bool, device=device)
        true_class_mask[torch.arange(batch), labels] = True

        for _ in range(self.num_steps):
            delta.requires_grad_(True)
            amplitudes  = born.classifier.amplitudes(naturals + delta)          # (B, K)
            log_joint   = 2 * torch.log(amplitudes.abs().clamp(min=self.eps))   # (B, K)
            log_joint_w = log_joint.masked_fill(true_class_mask, float('-inf'))
            loss        = log_joint_w.max(dim=-1).values.mean()

            born.classifier.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            grad  = delta.grad.detach()
            delta = delta.detach() + step_size * normalizing(grad, norm=self.norm)
            delta = self._project(delta, strength)

        return (naturals + delta).detach()


_METHOD_MAP = {
    "FGM":       FastGradientMethod,
    "PGD":       ProjectedGradientDescent,
    "JOINT_PGD": JointProjectedGradientDescent,
}


class RobustnessEvaluation:
    """
    Evaluate adversarial robustness of a BornMachine classifier.

    Generates adversarial examples using FGM or PGD and computes accuracy
    under attack at multiple perturbation strengths.
    """

    def __init__(
            self,
            method: str = "FGM",
            norm: int | str = "inf",
            criterion: CriterionConfig = CriterionConfig(name="nll", kwargs=None),
            strengths: List[float] = [0.1, 0.3],
            # PGD-specific parameters (ignored for FGM)
            num_steps: int = 10,
            step_size: float | None = None,
            random_start: bool = True
    ):
        """
        Initialize robustness evaluator.

        Args:
            method: Attack method - "FGM" or "PGD".
            norm: Lp norm for perturbation ball.
            criterion: Loss function configuration.
            strengths: List of epsilon values to evaluate.
            num_steps: PGD iterations (ignored for FGM).
            step_size: PGD step size (ignored for FGM).
            random_start: PGD random initialization (ignored for FGM).
        """
        self.strengths = strengths
        method_cls = _METHOD_MAP[method]
        if method == "PGD":
            self.method = method_cls(
                norm=norm,
                criterion=criterion,
                num_steps=num_steps,
                step_size=step_size,
                random_start=random_start
            )
        elif method == "JOINT_PGD":
            self.method = method_cls(
                norm=norm,
                num_steps=num_steps,
                step_size=step_size,
                random_start=random_start
            )
        else:
            self.method = method_cls(
                norm=norm,
                criterion=criterion
            )

    def generate(
            self,
            born: BornMachine,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            strength: float,
            device: torch.device | str = "cpu"
    ):
        return self.method.generate(
            born, naturals, labels, strength, device
        )

    def evaluate(
            self,
            born: BornMachine,
            loader: DataLoader,
            device: torch.device | str = "cpu"
    ):
        """
        Evaluate robustness of a classifier over multiple perturbation strengths.
        """
        born.to(device)
        born.classifier.eval()

        range_size = born.input_range[1] - born.input_range[0]
        strength_acc = []

        for strength in self.strengths:
            abs_strength = strength * range_size
            batch_acc = []

            for naturals, labels in loader:
                ad_examples = self.generate(
                    born, naturals, labels, abs_strength, device
                )

                with torch.no_grad():
                    ad_probs = born.classifier.probabilities(ad_examples)
                    ad_pred = torch.argmax(ad_probs, dim=1)
                    acc = (ad_pred == labels.to(device)).float().mean().item()
                    batch_acc.append(acc)

            mean_acc = sum(batch_acc) / len(batch_acc)
            strength_acc.append(mean_acc)

        return strength_acc

