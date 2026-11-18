import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression


def _logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def _sigmoid(z): return 1 / (1 + np.exp(-z))


class PlattCalibrator:
    """Logistic recalibration p' = sigmoid(a + b*logit(p)). Fit on TRAIN out-of-fold scores only, so it
    restores the training base rate (undoing class-weight inflation) without touching validation data."""

    def fit(self, p_raw, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(np.asarray(p_raw)).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, p_raw):
        return self.lr_.predict_proba(_logit(np.asarray(p_raw)).reshape(-1, 1))[:, 1]

    @property
    def params(self):
        return dict(intercept=float(self.lr_.intercept_[0]), slope=float(self.lr_.coef_[0, 0]))


def oracle_intercept_shift(p, target_rate: float) -> np.ndarray:
    """DIAGNOSTIC ONLY: shifts the logit intercept so mean(p') equals the VALIDATION default rate.
    It uses validation outcomes, so it is not a deployable model -- it shows how much of the calibration
    gap is pure base-rate shift (vs a ranking/shape problem)."""
    z = _logit(np.asarray(p))
    d = brentq(lambda s: _sigmoid(z + s).mean() - target_rate, -6, 6)
    return _sigmoid(z + d)
