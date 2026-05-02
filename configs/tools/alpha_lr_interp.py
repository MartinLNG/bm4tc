#!/usr/bin/env python3
"""
Compute geometric-interpolated learning rates for alpha_curve_mixed sweeps.

MixedNLL alpha convention (see src/utils/criterions.py):
    loss = (1-alpha)*ClassificationNLL + alpha*GenerativeNLL
    alpha=0 → pure classification
    alpha=1 → pure generative

For each alpha, the optimal LR is geometrically interpolated:
    lr(alpha) = lr_cls^(1-alpha) * lr_gen^alpha

where lr_cls = HPO-best LR for pure classification (alpha=0)
      lr_gen  = HPO-best LR for pure generative    (alpha=1)

Patches the d10D6 alpha_curve_mixed configs so that:
  - trainer.generative.optimizer.kwargs.lr uses the OmegaConf resolver
    ${geom_lr:${trainer.generative.criterion.kwargs.alpha},lr_cls,lr_gen}
    (resolver registered in experiments/generative.py)
  - sweeper.params stays as the original alpha × seed cartesian sweep

Usage:
    python configs/tools/alpha_lr_interp.py            # print table + patch
    python configs/tools/alpha_lr_interp.py --dry-run  # print table only
"""
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MIXED = PROJECT_ROOT / "configs" / "experiments" / "generative" / "legendre" / "d10D6" / "alpha_curve_mixed"

ALPHA_VALUES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 8e-1, 1.0]

# LR anchors sourced from HPO comments in each config file.
# lr_cls: best LR from pure-classification HPO  (corresponds to alpha=0 in MixedNLL)
# lr_gen: best LR from pure-generative HPO      (corresponds to alpha=1 in MixedNLL)
CONFIGS = {
    "circles_4k": {
        "path": _MIXED / "circles_4k.yaml",
        "lr_cls": 0.02931565161670948,    # pure classification (alpha=0)
        "lr_gen": 0.0006624310605949989,  # pure generative     (alpha=1)
    },
    "moons_4k": {
        "path": _MIXED / "moons_4k.yaml",
        "lr_cls": 0.041376293143363094,   # pure classification (alpha=0)
        "lr_gen": 0.0002636875533972306,  # pure generative     (alpha=1)
    },
    "spirals_4k": {
        "path": _MIXED / "spirals_4k.yaml",
        "lr_cls": 0.04360959649429807,    # pure classification (alpha=0)
        "lr_gen": 0.00017820395899344066, # pure generative     (alpha=1)
    },
}

_ALPHA_CHOICE = "choice(0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 8e-1, 1.0)"
_ALPHA_KEY = "trainer.generative.criterion.kwargs.alpha"

# Matches the alpha (or paired alpha+lr) line in sweeper.params.
_SWEEPER_LINE_RE = re.compile(
    r'^(\s*)"?trainer\.generative\.criterion\.kwargs\.alpha'
    r'(?:,trainer\.generative\.optimizer\.kwargs\.lr)?"?\s*:.*$',
    re.MULTILINE,
)

# Matches the lr: line inside optimizer.kwargs (8-space indent).
_LR_LINE_RE = re.compile(r'^( {8}lr):.*$', re.MULTILINE)


def geom_lr(lr_cls: float, lr_gen: float, alpha: float) -> float:
    """lr_cls^(1-alpha) * lr_gen^alpha — matches MixedNLL alpha convention."""
    return math.exp((1 - alpha) * math.log(lr_cls) + alpha * math.log(lr_gen))


def patch_config(path: Path, lr_cls: float, lr_gen: float) -> None:
    text = path.read_text()

    # 1. Restore sweeper param to plain alpha sweep (idempotent).
    new_text, n1 = _SWEEPER_LINE_RE.subn(
        lambda m: f"{m.group(1)}{_ALPHA_KEY}: {_ALPHA_CHOICE}",
        text,
    )
    if n1 == 0:
        print(f"  WARNING: alpha sweep line not found in {path.name} — skipped.")
        return

    # 2. Replace lr: <value> with resolver expression.
    # Resolver arg order: lr_cls first (alpha=0), lr_gen second (alpha=1).
    expr = f"${{geom_lr:${{{_ALPHA_KEY}}},{lr_cls},{lr_gen}}}"
    new_text, n2 = _LR_LINE_RE.subn(
        lambda m: f"{m.group(1)}: {expr}  # geom interp; resolver in experiments/generative.py",
        new_text,
    )
    if n2 == 0:
        print(f"  WARNING: lr line not found in {path.name} — skipped.")
        return

    path.write_text(new_text)
    print(f"  Patched {path.name} (sweeper: {n1}, lr: {n2}).")


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    for name, cfg in CONFIGS.items():
        lr_cls, lr_gen = cfg["lr_cls"], cfg["lr_gen"]

        print(f"\n{name}  (lr_cls={lr_cls:.3e} at alpha=0, lr_gen={lr_gen:.3e} at alpha=1, ratio={lr_cls / lr_gen:.0f}x)")
        print(f"  {'alpha':>6}  {'lr':>14}  {'regime':}")
        for a in ALPHA_VALUES:
            regime = "pure cls" if a == 0.0 else ("pure gen" if a == 1.0 else "mixed")
            print(f"  {a:>6.1f}  {geom_lr(lr_cls, lr_gen, a):>14.6e}  {regime}")

        if dry_run:
            print(f"  [dry-run] would patch {cfg['path'].name}")
        else:
            patch_config(cfg["path"], lr_cls, lr_gen)

    if dry_run:
        print("\nDry run — no files written. Remove --dry-run to apply.")


if __name__ == "__main__":
    main()
