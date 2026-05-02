import pytest
import torch
from tests.conftest import DATA_DIM
from src.utils.purification.gibbs import GibbsPurification

pytestmark = pytest.mark.slow

NUM_BINS = 10
BATCH_SIZE = 4


@pytest.fixture
def x_adv():
    return torch.rand(BATCH_SIZE, DATA_DIM)


def test_purify_output_shape(born_machine, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    purified, _ = purifier.purify(born_machine, x_adv, n_sweeps=1, device="cpu")
    assert purified.shape == (BATCH_SIZE, DATA_DIM)


def test_purify_log_px_shape(born_machine, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    _, log_px = purifier.purify(born_machine, x_adv, n_sweeps=1, device="cpu")
    assert log_px.shape == (BATCH_SIZE,)


def test_purify_in_input_range(born_machine, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    purified, _ = purifier.purify(born_machine, x_adv, n_sweeps=1, device="cpu")
    lo, hi = born_machine.input_range
    assert (purified >= lo - 1e-5).all()
    assert (purified <= hi + 1e-5).all()


def test_purify_log_px_finite(born_machine, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    _, log_px = purifier.purify(born_machine, x_adv, n_sweeps=1, device="cpu")
    assert torch.isfinite(log_px).all()


def test_purify_one_sweep(born_machine, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    purified, log_px = purifier.purify(born_machine, x_adv, n_sweeps=1, device="cpu")
    assert purified.shape[0] == BATCH_SIZE


def test_purify_three_sweeps(born_machine, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    purified, log_px = purifier.purify(born_machine, x_adv, n_sweeps=3, device="cpu")
    assert purified.shape[0] == BATCH_SIZE


def test_purify_batch_size_one(born_machine):
    x = torch.rand(1, DATA_DIM)
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=1)
    purified, _ = purifier.purify(born_machine, x, n_sweeps=1, device="cpu")
    assert purified.shape == (1, DATA_DIM)


def test_purify_partial_batch(born_machine):
    n_samples = 5
    x = torch.rand(n_samples, DATA_DIM)
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=3)
    purified, _ = purifier.purify(born_machine, x, n_sweeps=1, device="cpu")
    assert purified.shape == (n_samples, DATA_DIM)


# --- Restricted Gibbs (radius) ---

def test_restricted_purify_output_shape(born_machine, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, radius=0.3)
    purified, _ = purifier.purify(born_machine, x_adv, n_sweeps=1, device="cpu")
    assert purified.shape == (BATCH_SIZE, DATA_DIM)


def test_restricted_purify_stays_in_input_range(born_machine, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, radius=0.3)
    purified, _ = purifier.purify(born_machine, x_adv, n_sweeps=1, device="cpu")
    lo, hi = born_machine.input_range
    assert (purified >= lo - 1e-5).all()
    assert (purified <= hi + 1e-5).all()


def test_restricted_purify_stays_near_start(born_machine):
    # With a small radius, purified values must be within radius of x_adv
    # (per feature, since each feature is sampled from [x_adv_k ± delta]).
    torch.manual_seed(0)
    radius = 0.1
    x_adv = torch.full((BATCH_SIZE, DATA_DIM), 0.5)  # well inside input_range [0,1]
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, radius=radius)
    purified, _ = purifier.purify(born_machine, x_adv, n_sweeps=1, device="cpu")
    lo, hi = born_machine.input_range
    delta = radius * (hi - lo)
    # Each feature must stay within [x_adv_k - delta, x_adv_k + delta] ∩ [lo, hi]
    lo_bound = (x_adv - delta).clamp(lo, hi)
    hi_bound = (x_adv + delta).clamp(lo, hi)
    assert (purified >= lo_bound - 1e-5).all(), "Purified values below lower restriction bound"
    assert (purified <= hi_bound + 1e-5).all(), "Purified values above upper restriction bound"


def test_restricted_purify_log_px_finite(born_machine, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, radius=0.3)
    _, log_px = purifier.purify(born_machine, x_adv, n_sweeps=3, device="cpu")
    assert torch.isfinite(log_px).all()
