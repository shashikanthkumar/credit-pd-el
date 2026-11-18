import numpy as np, pandas as pd
from src.features.woe import WoEBinner, SMOOTH


def _toy(n=20000, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=n)
    y = (r.random(n) < 1 / (1 + np.exp(-(-1.5 + 0.9 * x)))).astype(int)   # risk rises with x
    cat = r.choice(["a", "b", "c", "rare"], n, p=[.45, .35, .17, .03])
    x = np.where(r.random(n) < .02, np.nan, x)
    return pd.DataFrame({"x": x, "cat": cat}), pd.Series(y)


def test_numeric_bins_are_monotone_and_big_enough():
    X, y = _toy()
    b = WoEBinner(min_bin_frac=0.05, numeric=["x"], categorical=["cat"]).fit(X, y)
    t = b.bin_table("x"); t = t[t["bin"] != "MISSING"]
    assert t["bad_rate"].is_monotonic_increasing or t["bad_rate"].is_monotonic_decreasing
    assert (t["n"] >= 0.05 * len(X)).all()


def test_woe_sums_and_iv_identity():
    X, y = _toy()
    b = WoEBinner(numeric=["x"], categorical=["cat"]).fit(X, y)
    for c in ["x", "cat"]:
        t = b.bin_table(c)
        pb, pg = (t["bad"] + SMOOTH) / (t["bad"] + SMOOTH).sum(), (t["good"] + SMOOTH) / (t["good"] + SMOOTH).sum()
        assert np.allclose(t["woe"], np.log(pb / pg))
        assert np.isclose(t["iv"].sum(), ((pb - pg) * np.log(pb / pg)).sum())
        assert t["iv"].min() >= -1e-12                     # IV contributions are non-negative


def test_woe_sign_convention_and_transform():
    X, y = _toy()
    b = WoEBinner(numeric=["x"]).fit(X, y)
    W = b.transform(X)
    assert W["x_woe"].corr(X["x"]) > 0                    # riskier (higher x) -> higher WoE
    assert W["x_woe"].notna().all()                       # NaN gets its own bin


def test_rare_and_unseen_categories():
    from src.features.woe import POOLED
    X, y = _toy()
    b = WoEBinner(categorical=["cat"], min_bin_frac=0.05).fit(X, y)
    t = b.bin_table("cat").set_index("bin")
    assert "rare" not in set(t.index) and POOLED in set(t.index)       # pooled under the reserved sentinel
    X2 = X.copy(); X2["cat"] = "never_seen"
    assert (b.transform(X2)["cat_woe"] == t.loc[POOLED, "woe"]).all()  # unseen level -> pooled-bin WoE
