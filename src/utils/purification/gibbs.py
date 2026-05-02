"""Gibbs-sampling purification for Born Machines (class-marginalized)."""

import torch
from typing import Optional, Tuple

from src.models.generator.differential_sampling import multinomial_sampling


class GibbsPurification:
    """Purify adversarial examples via class-marginalized Gibbs sampling.

    For each feature in turn, resample it from the conditional Born distribution
    p(x_i | x_{-i}), marginalizing over class label.  Multiple sweeps over all
    features produce a sample that is more consistent with the Born model's
    learned distribution.  Classification is performed on the purified sample.

    Args:
        num_bins: Resolution of the discrete grid over the input range.
        gibbs_batch_size: Number of adversarial samples processed per
            sequential() call.  Controls density-matrix memory:
            (gibbs_batch_size × num_bins)² × 4 bytes.
            At bs=8, bins=200: ~10 MB.  At bs=32, bins=200: ~160 MB.
        radius: Perturbation budget as a fraction of the input range size
            (b - a).  Each sweep restricts resampling of feature i to
            [x̄_i ± delta_abs] where delta_abs = radius * (hi - lo).
            The same budget applies regardless of n_sweeps.  Intervals are
            clamped to input_range.  If None, samples from the full input range.
    """

    def __init__(
        self,
        num_bins: int,
        gibbs_batch_size: int = 8,
        radius: Optional[float] = 0.1,
    ):
        self.num_bins = num_bins
        self.gibbs_batch_size = gibbs_batch_size
        self.radius = radius

    def purify(
        self,
        born,
        x_adv: torch.Tensor,
        n_sweeps: int,
        device: torch.device | str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Purify adversarial examples using Gibbs sampling.

        Args:
            born: BornMachine instance.  Generator tensors must already be
                in sync with the classifier (call born.sync_tensors if needed).
            x_adv: Adversarial inputs, shape (n_samples, data_dim).
            n_sweeps: Number of full sweeps over all features.
            device: Torch device.

        Returns:
            Tuple of:
                - x_purified: Purified inputs on CPU, shape (n_samples, data_dim).
                - log_px_after: Marginal log p(x) of purified inputs on CPU,
                  shape (n_samples,).
        """
        generator = born.generator
        cls_pos = generator.cls_pos
        n_samples = len(x_adv)
        lo, hi = born.input_range

        input_space = torch.linspace(lo, hi, self.num_bins, device=device)
        # (num_bins, in_dim) — fixed grid embedding, same for every feature and sweep
        in_emb_grid = generator.embedding(input_space)

        # Convert relative radius to absolute.
        delta: Optional[float] = (
            self.radius * (hi - lo)
            if self.radius is not None
            else None
        )

        results = []
        for batch_start in range(0, n_samples, self.gibbs_batch_size):
            batch = x_adv[batch_start : batch_start + self.gibbs_batch_size].to(device)
            bs = len(batch)
            x_cur = batch.clone()

            for _ in range(n_sweeps):
                # Snapshot feature values at sweep start; restriction intervals are
                # centered on these values, not on the within-sweep updated values.
                x_bar = x_cur.clone() if delta is not None else None

                for s in range(generator.n_features):
                    if s == cls_pos:
                        continue
                    # Map MPS site → data column
                    k = s if s < cls_pos else s - 1

                    # Build embs: all data sites except current use their present value;
                    # current site uses the full candidate grid.  cls_pos is absent →
                    # tensorkrowch auto-designates it as the sole output site →
                    # marginalize_output=True traces over the class physical dim.
                    embs = {}
                    for s2 in range(generator.n_features):
                        if s2 == cls_pos or s2 == s:
                            continue
                        k2 = s2 if s2 < cls_pos else s2 - 1
                        embs[s2] = (
                            generator.embedding(x_cur[:, k2])
                            .unsqueeze(1)
                            .expand(bs, self.num_bins, -1)
                            .reshape(bs * self.num_bins, -1)
                        )
                    embs[s] = (
                        in_emb_grid.unsqueeze(0)
                        .expand(bs, -1, -1)
                        .reshape(bs * self.num_bins, -1)
                    )

                    generator.prepare()
                    p = generator.sequential(embs).view(bs, self.num_bins)

                    if delta is not None:
                        # Restrict each sample's conditional to [x̄_k ± delta] ∩ [lo, hi].
                        # Vectorised: lo_k/hi_k are (bs, 1), input_space is (num_bins,).
                        lo_k = (x_bar[:, k] - delta).clamp(lo, hi).unsqueeze(1)
                        hi_k = (x_bar[:, k] + delta).clamp(lo, hi).unsqueeze(1)
                        mask = (input_space[None, :] >= lo_k) & (input_space[None, :] <= hi_k)
                        p = p * mask

                    x_cur[:, k] = multinomial_sampling(p, input_space)

            results.append(x_cur.cpu())

        x_purified = torch.cat(results, dim=0)
        log_px_after = born.marginal_log_probability(x_purified.to(device))
        generator.prepare()

        return x_purified, log_px_after.detach().cpu()
