import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss


def auc(y, p): return roc_auc_score(y, p)
def gini(y, p): return 2 * roc_auc_score(y, p) - 1


def ks(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    o = np.argsort(p)
    y = y[o]
    cb = np.cumsum(y) / max(y.sum(), 1)
    cg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cb - cg)))


def brier(y, p): return brier_score_loss(y, p)


def calibration_by_decile(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Deciles of PREDICTED PD; observed vs expected default rate plus a two-sided binomial p-value."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False) + 1
    g = d.groupby("decile").agg(n=("y", "size"), defaults=("y", "sum"),
                                expected_pd=("p", "mean"), observed_dr=("y", "mean"))
    g["abs_gap"] = g["observed_dr"] - g["expected_pd"]
    g["rel_gap"] = g["abs_gap"] / g["expected_pd"]
    g["binom_p"] = [stats.binomtest(int(r.defaults), int(r.n), float(np.clip(r.expected_pd, 1e-9, 1 - 1e-9))).pvalue
                    for r in g.itertuples()]
    return g.reset_index()


def calibration_summary(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(mean_pred=float(p.mean()), observed=float(y.mean()), cal_ratio=float(y.mean() / p.mean()),
                brier=float(brier(y, p)))


def psi(expected, actual, bins: int = 10, categorical: bool = False) -> float:
    """Population Stability Index: a DISTRIBUTION-SHIFT metric. It compares input/score distributions
    between two samples and uses NO outcomes, so it says nothing about discrimination or calibration.
    Common rule of thumb (a convention, not a theorem): <0.10 stable, 0.10-0.25 moderate, >0.25 major."""
    e, a = pd.Series(expected), pd.Series(actual)
    if not categorical:
        edges = np.unique(np.nanquantile(e.dropna(), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            # low-resolution feature (e.g. delinq_2yrs, mostly 0): quantile edges collapse to one bin and
            # PSI would be 0 BY CONSTRUCTION. Fall back to value counts so a real shift is still detectable.
            categorical = True
    if categorical:
        lab = lambda s: s.astype(object).where(s.notna(), "MISSING").astype(str)
        e, a = lab(e), lab(a)
        cats = sorted(set(e) | set(a))
        pe = e.value_counts(normalize=True).reindex(cats).fillna(0)
        pa = a.value_counts(normalize=True).reindex(cats).fillna(0)
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        pe = pd.cut(e, edges).value_counts(normalize=True, sort=False)
        pa = pd.cut(a, edges).value_counts(normalize=True, sort=False)
    pe, pa = pe.clip(lower=1e-4), pa.clip(lower=1e-4)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def performance_row(y, p, label: str) -> dict:
    y = np.asarray(y)
    return dict(sample=label, n=len(y), default_rate=float(y.mean()), auc=auc(y, p), gini=gini(y, p),
                ks=ks(y, p), brier=brier(y, p), mean_pd=float(np.mean(p)))
