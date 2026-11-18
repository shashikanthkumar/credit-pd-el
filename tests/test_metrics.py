import numpy as np
from src.evaluation.metrics import auc, gini, ks, psi, calibration_by_decile, brier


def test_perfect_and_random_rankers():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([.1, .2, .3, .7, .8, .9])
    assert auc(y, p) == 1 and gini(y, p) == 1 and ks(y, p) == 1
    r = np.random.default_rng(0); y = r.integers(0, 2, 50000); p = r.random(50000)
    assert abs(auc(y, p) - .5) < .01 and ks(y, p) < .02


def test_gini_equals_2auc_minus_1_and_known_ks():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .6, .4, .9])
    assert np.isclose(gini(y, p), 2 * auc(y, p) - 1)
    assert np.isclose(ks(y, p), .5)


def test_brier_known_value():
    assert np.isclose(brier(np.array([1, 0]), np.array([.8, .2])), .04)


def test_psi_zero_for_identical_and_large_for_shift():
    x = np.random.default_rng(1).normal(size=20000)
    assert psi(x, x) < 1e-9
    assert psi(x, x + 1.5) > 0.25
    assert psi(x, np.random.default_rng(2).normal(size=20000)) < 0.02


def test_calibration_deciles_perfectly_calibrated():
    r = np.random.default_rng(3); p = r.uniform(.02, .5, 200000); y = (r.random(200000) < p).astype(int)
    t = calibration_by_decile(y, p)
    assert len(t) == 10 and (t["abs_gap"].abs() < .01).all()
