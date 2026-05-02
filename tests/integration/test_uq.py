import pytest
import torch
import numpy as np
from analysis.utils.uq import (
    UQConfig,
    UQResults,
    UQEvaluation,
    PurificationMetrics,
    compute_log_px,
    compute_thresholds,
)

pytestmark = pytest.mark.slow


# ---- Unit-level dataclass tests (no model needed) ----

def test_uq_config_defaults():
    cfg = UQConfig()
    assert cfg.norm == "inf"
    assert cfg.num_steps == 20
    assert isinstance(cfg.radii, list)
    assert isinstance(cfg.percentiles, list)


def test_uq_config_gibbs_fields():
    cfg = UQConfig()
    assert cfg.run_gibbs is False
    assert isinstance(cfg.gibbs_n_sweeps, list)
    assert cfg.gibbs_num_bins > 0
    assert cfg.gibbs_batch_size > 0


def test_purification_metrics_construct():
    m = PurificationMetrics(
        accuracy_after_purify=0.8,
        recovery_rate=0.5,
        mean_log_px_before=-2.0,
        mean_log_px_after=-1.0,
        rejection_rate=0.1,
    )
    assert m.accuracy_after_purify == pytest.approx(0.8)


def test_uq_results_gibbs_default_empty():
    results = UQResults(
        clean_log_px=np.array([-1.0, -2.0]),
        clean_accuracy=0.9,
        thresholds={5: -3.0},
        adv_log_px={},
        adv_accuracies={},
        detection_rates={},
        purification_results={},
    )
    assert results.gibbs_purification_results == {}


def test_uq_results_new_det_fields_default_empty():
    results = UQResults(
        clean_log_px=np.array([-1.0]),
        clean_accuracy=0.9,
        thresholds={},
        adv_log_px={},
        adv_accuracies={},
        detection_rates={},
        purification_results={},
    )
    assert results.err_rate_detected == {}
    assert results.err_rate_passed == {}


def test_uq_results_summary_returns_string():
    results = UQResults(
        clean_log_px=np.array([-1.0, -2.0]),
        clean_accuracy=0.9,
        thresholds={5: -3.0},
        adv_log_px={0.1: np.array([-5.0, -6.0])},
        adv_accuracies={0.1: 0.5},
        detection_rates={(5, 0.1): 0.8},
        purification_results={},
    )
    summary = results.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


# ---- Integration tests (need born_machine + clean_loader) ----

def test_compute_log_px_shape(born_machine, clean_loader):
    log_px, labels = compute_log_px(born_machine, clean_loader, device="cpu")
    assert log_px.shape == (32,)
    assert labels.shape == (32,)


def test_compute_log_px_finite(born_machine, clean_loader):
    log_px, _ = compute_log_px(born_machine, clean_loader, device="cpu")
    assert torch.isfinite(log_px).all()


def test_compute_thresholds_all_percentiles_present(born_machine, clean_loader):
    percentiles = [1, 5, 10, 20]
    thresholds, _ = compute_thresholds(born_machine, clean_loader, percentiles, device="cpu")
    for p in percentiles:
        assert p in thresholds


def test_compute_thresholds_ordered(born_machine, clean_loader):
    percentiles = [5, 20]
    thresholds, _ = compute_thresholds(born_machine, clean_loader, percentiles, device="cpu")
    assert thresholds[5] <= thresholds[20]


def test_uq_evaluate_completes(born_machine, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(born_machine, clean_loader, device="cpu")
    assert isinstance(results, UQResults)


def test_uq_clean_accuracy_range(born_machine, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(born_machine, clean_loader, device="cpu")
    assert 0.0 <= results.clean_accuracy <= 1.0


def test_uq_detection_rates_range(born_machine, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(born_machine, clean_loader, device="cpu")
    for rate in results.detection_rates.values():
        assert 0.0 <= rate <= 1.0


def test_uq_purification_results_present(born_machine, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(born_machine, clean_loader, device="cpu")
    assert len(results.purification_results) > 0


def test_uq_purification_acc_range(born_machine, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(born_machine, clean_loader, device="cpu")
    for metrics in results.purification_results.values():
        assert 0.0 <= metrics.accuracy_after_purify <= 1.0


def test_uq_gibbs_empty_when_disabled(born_machine, clean_loader):
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
        run_gibbs=False,
    )
    evaluator = UQEvaluation(cfg)
    results = evaluator.evaluate(born_machine, clean_loader, device="cpu")
    assert results.gibbs_purification_results == {}


def test_uq_err_rate_detected_range(born_machine, clean_loader):
    import math
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    results = UQEvaluation(cfg).evaluate(born_machine, clean_loader, device="cpu")
    assert len(results.err_rate_detected) > 0
    for v in results.err_rate_detected.values():
        assert math.isnan(v) or 0.0 <= v <= 1.0


def test_uq_err_rate_passed_range(born_machine, clean_loader):
    import math
    cfg = UQConfig(
        attack_strengths=[0.1],
        radii=[0.1],
        percentiles=[10],
        attack_num_steps=2,
        num_steps=2,
    )
    results = UQEvaluation(cfg).evaluate(born_machine, clean_loader, device="cpu")
    assert len(results.err_rate_passed) > 0
    for v in results.err_rate_passed.values():
        assert math.isnan(v) or 0.0 <= v <= 1.0
