"""WoE / IV implemented from scratch (cross-checked against scorecardpy/skorecard in notebook 03).

Convention: WoE = ln(%bad / %good) so a logistic regression on WoE should have POSITIVE
coefficients (higher WoE = riskier). IV is identical under either sign convention.
Counts are smoothed by +0.5 to avoid infinite WoE in empty cells.
Rare categorical levels are pooled into the reserved bin '_POOLED_' (never 'OTHER', which is a real
Lending Club home_ownership value).
NaN always gets its own bin. Numeric bins are monotone in bad-rate (direction chosen from the data).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

SMOOTH = 0.5
POOLED = "_POOLED_"      # sentinel for rare-level pooling; must never equal a real category value
MISSING = "MISSING"


def _woe_iv(bad, good):
    b, g = np.asarray(bad, float) + SMOOTH, np.asarray(good, float) + SMOOTH
    pb, pg = b / b.sum(), g / g.sum()
    woe = np.log(pb / pg)
    return woe, (pb - pg) * woe


def _bin_stats(edges, x, y):
    idx = np.searchsorted(edges, x, side="right")
    k = len(edges) + 1
    n = np.bincount(idx, minlength=k).astype(float)
    bad = np.bincount(idx, weights=y, minlength=k)
    rate = np.divide(bad, n, out=np.zeros(k), where=n > 0)
    return n, rate


def _monotone_merge(edges, x, y, min_n):
    """Merge adjacent numeric bins until every bin has >= min_n obs and bad-rate is monotone."""
    edges = list(edges)
    while edges:
        n, rate = _bin_stats(edges, x, y)
        k = len(n)
        small = np.where(n < min_n)[0]
        if len(small):
            i = small[np.argmin(n[small])]
            # drop the edge separating bin i from its smaller neighbour
            if i == 0:
                j = 0
            elif i == k - 1:
                j = k - 2
            else:
                j = i - 1 if n[i - 1] <= n[i + 1] else i
            del edges[j]
            continue
        if k <= 2:
            break
        trend = np.sign(np.corrcoef(np.arange(k), rate)[0, 1])
        trend = 1.0 if (np.isnan(trend) or trend == 0) else trend
        d = np.diff(rate) * trend
        viol = np.where(d < 0)[0]
        if len(viol) == 0:
            break
        del edges[viol[np.argmin(d[viol])]]
    return edges


@dataclass
class WoEBinner:
    fine_bins: int = 20
    min_bin_frac: float = 0.05
    numeric: list = field(default_factory=list)
    categorical: list = field(default_factory=list)
    tables_: dict = field(default_factory=dict)
    iv_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = np.asarray(y, float)
        min_n = int(self.min_bin_frac * len(X))
        for c in self.numeric:
            self.tables_[c] = self._fit_numeric(X[c].to_numpy(float), y, min_n)
        for c in self.categorical:
            self.tables_[c] = self._fit_categorical(X[c].astype("object"), y, min_n)
        self.iv_ = pd.Series({c: t["iv"].sum() for c, t in self.tables_.items()}).sort_values(ascending=False)
        return self

    def _fit_numeric(self, x, y, min_n):
        ok = ~np.isnan(x)
        xv, yv = x[ok], y[ok]
        qs = np.unique(np.quantile(xv, np.linspace(0, 1, self.fine_bins + 1)[1:-1]))
        edges = _monotone_merge(qs, xv, yv, min_n)
        n, _ = _bin_stats(edges, xv, yv)
        idx = np.searchsorted(edges, xv, side="right")
        bad = np.bincount(idx, weights=yv, minlength=len(edges) + 1)
        b = [-np.inf] + list(edges) + [np.inf]
        labels = [f"[{b[i]:.4g}, {b[i+1]:.4g})" for i in range(len(b) - 1)]
        n_all, bad_all = list(n), list(bad)
        if (~ok).any():
            labels.append("MISSING"); n_all.append((~ok).sum()); bad_all.append(y[~ok].sum())
        t = pd.DataFrame(dict(bin=labels, n=n_all, bad=bad_all))
        t["good"] = t["n"] - t["bad"]
        t["woe"], t["iv"] = _woe_iv(t["bad"], t["good"])
        t["bad_rate"] = t["bad"] / t["n"]
        t.attrs = dict(kind="num", edges=list(edges), has_missing=bool((~ok).any()))
        return t

    def _fit_categorical(self, s, y, min_n):
        s = s.where(s.notna(), MISSING)
        counts = s.value_counts()
        # real data may legitimately contain e.g. home_ownership == "OTHER": the pooling bucket uses a sentinel
        # that cannot collide with it, so a genuine category is never blended with pooled rare levels
        assert POOLED not in set(counts.index), f"data contains the reserved pooling label {POOLED!r}"
        keep = set(counts[counts >= min_n].index) | ({MISSING} & set(counts.index))
        s2 = s.where(s.isin(keep), POOLED)
        t = pd.DataFrame({"bin": s2.to_numpy(), "y": y}).groupby("bin")["y"].agg(n="size", bad="sum").reset_index()
        t["good"] = t["n"] - t["bad"]
        t["bad_rate"] = t["bad"] / t["n"]
        t = t.sort_values("bad_rate").reset_index(drop=True)
        t["woe"], t["iv"] = _woe_iv(t["bad"], t["good"])
        t.attrs = dict(kind="cat", keep=sorted(keep))
        return t

    def _apply(self, c, series):
        t = self.tables_[c]
        if t.attrs["kind"] == "num":
            x = series.to_numpy(float)
            edges = t.attrs["edges"]
            woe = t["woe"].to_numpy()
            out = woe[np.searchsorted(edges, np.nan_to_num(x), side="right")]
            return np.where(np.isnan(x), woe[-1] if t.attrs["has_missing"] else 0.0, out)
        s = series.astype("object")
        s = s.where(s.notna(), MISSING)
        s = s.where(s.isin(t.attrs["keep"]), POOLED)
        # unseen level -> pooled-bin WoE if a pooled bin exists, else neutral 0
        return s.map(dict(zip(t["bin"], t["woe"]))).fillna(0.0).to_numpy(float)

    def transform(self, X: pd.DataFrame, features=None) -> pd.DataFrame:
        feats = features or list(self.tables_)
        return pd.DataFrame({f"{c}_woe": self._apply(c, X[c]) for c in feats}, index=X.index)

    def select(self, iv_min: float) -> list:
        return list(self.iv_[self.iv_ >= iv_min].index)

    def bin_table(self, c: str) -> pd.DataFrame:
        return self.tables_[c].copy()


def bin_stability(binner: WoEBinner, feature: str, X_other: pd.DataFrame, y_other) -> pd.DataFrame:
    """Apply TRAIN bins to another sample; compare bad-rate per bin (does the risk ordering hold out-of-time?)."""
    t = binner.tables_[feature]
    d = pd.DataFrame({"woe": binner._apply(feature, X_other[feature]), "y": np.asarray(y_other, float)})
    o = d.groupby("woe")["y"].agg(n_other="size", bad_rate_other="mean").reset_index()
    return t.merge(o, on="woe", how="left")[["bin", "n", "bad_rate", "n_other", "bad_rate_other", "woe"]]
