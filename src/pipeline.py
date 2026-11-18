"""End-to-end pipeline used by notebook 99 and `make train`. Notebooks 01-06 call the same functions step by step."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.utils.config import load_config, abs_path, set_seeds
from src.data.loader import load_model_frame
from src.data.cleaning import basic_clean, fit_winsor, apply_winsor
from src.features.woe import WoEBinner
from src.features.guards import feature_sets, assert_no_banned
from src.models.pd_models import fit_logit, GBMPrep, make_lgbm, oof_proba, smote_experiment
from src.models.calibration import PlattCalibrator, oracle_intercept_shift
from src.models.el_proxy import fit_lgd_ead_proxy
from src.evaluation.metrics import performance_row, calibration_by_decile, calibration_summary, psi
from src.decision.threshold import tradeoff_table, recommend, ks_optimal_cutoff


def _yr(df, rng): return df["vintage_year"].between(rng[0], rng[1])


@dataclass
class Data:
    df: pd.DataFrame          # all 36m loans, cleaned + winsorised (any status)
    train: pd.DataFrame
    valid: pd.DataFrame       # clean OOT (fully matured), resolved only
    flagged: pd.DataFrame     # partially censored, resolved only
    shift: dict               # year -> all-status loans (PSI only)
    winsor: dict
    valid_audit: pd.DataFrame = None   # per issue-month maturity/resolution audit of the clean-validation year(s)


def maturity_audit(df: pd.DataFrame, years, min_share: float) -> pd.DataFrame:
    """Per issue-month: loans, resolved share, fully-matured flag. A month is CLEAN only if it is fully matured
    AND at least `min_share` of its loans are resolved (guards residual censoring at the window edge,
    e.g. December 2015 loans whose 36th payment falls at the snapshot)."""
    d = df[df["vintage_year"].between(years[0], years[1])]
    g = (d.groupby("issue_month")
           .agg(loans=("is_resolved", "size"), resolved_share=("is_resolved", "mean"), fully_matured=("fully_matured", "mean"))
           .reset_index())
    g["clean"] = (g["fully_matured"] == 1.0) & (g["resolved_share"] >= min_share)
    return g


def prepare(cfg: dict, rebuild: bool = False) -> Data:
    s = cfg["split"]
    df = basic_clean(load_model_frame(cfg, rebuild))
    df["issue_month"] = pd.to_datetime(df["issue_date"]).dt.to_period("M").astype(str)
    res = df["is_resolved"].astype(bool)
    limits = fit_winsor(df[_yr(df, s["train_years"]) & res], tuple(cfg["binning"]["winsor_quantiles"]))
    df = apply_winsor(df, limits)
    res = df["is_resolved"].astype(bool)
    train = df[_yr(df, s["train_years"]) & res].copy()
    audit = maturity_audit(df, s["valid_clean_years"], s.get("clean_min_resolved_share", 0.95))
    clean_months = set(audit.loc[audit["clean"], "issue_month"])
    in_valid_years = _yr(df, s["valid_clean_years"]) & res
    valid = df[in_valid_years & df["issue_month"].isin(clean_months)].copy()
    # months of the nominal clean year that fail the audit are NOT silently kept: they move to the flagged sample
    demoted = df[in_valid_years & ~df["issue_month"].isin(clean_months)]
    flagged = pd.concat([df[_yr(df, s["valid_flagged_years"]) & res], demoted]).copy()
    shift = {y: df[df["vintage_year"] == y].copy() for y in range(s["shift_only_years"][0], s["shift_only_years"][1] + 1)}
    assert train["vintage_year"].max() < valid["vintage_year"].min(), "split boundary violated"
    assert valid["fully_matured"].all(), "clean validation sample contains non-matured loans"
    return Data(df, train, valid, flagged, shift, limits, audit)


class PDModel:
    """Binning + model + train-only OOF Platt calibration behind one interface."""

    def __init__(self, cfg, numeric, categorical, kind="logit", name=""):
        self.cfg, self.numeric, self.categorical, self.kind, self.name = cfg, list(numeric), list(categorical), kind, name
        assert_no_banned(self.numeric + self.categorical, cfg)

    def _X(self, df):
        return self.binner.transform(df, self.features) if self.kind == "logit" else self.prep.transform(df)

    def fit(self, train):
        c, y = self.cfg, train["target"].astype(int)
        if self.kind == "logit":
            b = c["binning"]
            self.binner = WoEBinner(b["fine_bins"], b["min_bin_frac"], self.numeric, self.categorical).fit(train, y)
            self.features = self.binner.select(b["iv_min"]) or list(self.binner.iv_.index[:1])
            X = self._X(train)
            self.model = fit_logit(X, y, c)
        else:
            self.prep = GBMPrep(self.numeric, self.categorical).fit(train)
            X = self.prep.transform(train)
            self.model = make_lgbm(c, self.prep).fit(X, y)
        oof = oof_proba(clone(self.model), X, y, c)       # train-only out-of-fold scores
        self.calibrator = PlattCalibrator().fit(oof, y)
        return self

    def predict_raw(self, df): return self.model.predict_proba(self._X(df))[:, 1]
    def predict(self, df): return self.calibrator.predict(self.predict_raw(df))


def build_models(data: Data, cfg: dict) -> dict:
    fs = feature_sets(cfg)
    spec = {"grade_only_logit": ("grade_only", "logit"), "champion_logit": ("core", "logit"),
            "with_lc_logit": ("with_lc", "logit"), "challenger_lgbm": ("core", "gbm"), "with_lc_lgbm": ("with_lc", "gbm")}
    return {n: PDModel(cfg, *fs[s], kind=k, name=n).fit(data.train) for n, (s, k) in spec.items()}


def get_models(data: Data, cfg: dict, refit: bool = False) -> dict:
    """Train-or-load. Cache key = hash of the config, so changing any setting invalidates the cache."""
    import hashlib, pickle
    key = hashlib.md5(json.dumps({k: v for k, v in cfg.items() if k != "_root"}, sort_keys=True, default=str).encode()).hexdigest()[:10]
    p = abs_path(cfg, "processed_dir") / f"models_{key}.pkl"
    if p.exists() and not refit:
        return pickle.loads(p.read_bytes())
    models = build_models(data, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(models))
    return models


def evaluate_models(models: dict, data: Data) -> pd.DataFrame:
    rows = []
    for n, m in models.items():
        for lab, d in [("train (in-sample)", data.train), ("valid_clean", data.valid), ("flagged_censored", data.flagged)]:
            for pdt, p in [("raw", m.predict_raw(d)), ("calibrated", m.predict(d))]:
                rows.append({"model": n, "pd_type": pdt, **performance_row(d["target"].astype(int), p, lab)})
    return pd.DataFrame(rows)


def calibration_tables(models: dict, data: Data) -> dict:
    y = data.valid["target"].astype(int).to_numpy()
    out = {}
    for n in ["champion_logit", "challenger_lgbm"]:
        raw, cal = models[n].predict_raw(data.valid), models[n].predict(data.valid)
        orc = oracle_intercept_shift(cal, y.mean())
        out[n] = dict(
            summary=pd.DataFrame([{"variant": "raw (class-weighted)", **calibration_summary(y, raw)},
                                  {"variant": "train-only Platt", **calibration_summary(y, cal)},
                                  {"variant": "ORACLE intercept shift (diagnostic)", **calibration_summary(y, orc)}]),
            decile_raw=calibration_by_decile(y, raw), decile_cal=calibration_by_decile(y, cal),
            decile_oracle=calibration_by_decile(y, orc))
    return out


def psi_tables(models: dict, data: Data) -> dict:
    m = models["champion_logit"]
    ref = data.train
    samples = {"valid_2015": data.valid, "flagged_2016": data.flagged, **{f"shift_{y}": d for y, d in data.shift.items()}}
    score = pd.DataFrame([{"sample": k, "score_psi": psi(m.predict(ref), m.predict(d))} for k, d in samples.items() if len(d)])
    feats = []
    for f in m.numeric + m.categorical:
        for k, d in samples.items():
            if len(d):
                feats.append(dict(feature=f, sample=k, psi=psi(ref[f], d[f], categorical=f in m.categorical)))
    return dict(score=score, features=pd.DataFrame(feats).pivot(index="feature", columns="sample", values="psi"))


def run_all(cfg: dict | None = None, rebuild: bool = False, make_reports: bool = True) -> dict:
    cfg = cfg or load_config()
    set_seeds(cfg["seed"])
    data = prepare(cfg, rebuild)
    models = build_models(data, cfg)
    perf = evaluate_models(models, data)
    cal = calibration_tables(models, data)
    champ = models["champion_logit"]

    Xw_tr, Xw_va = champ._X(data.train), champ._X(data.valid)
    smote = smote_experiment(Xw_tr, data.train["target"], Xw_va, data.valid["target"], cfg)

    s = cfg["split"]
    sens_tr = data.train[_yr(data.train, s["sensitivity_train_years"])]
    sens = PDModel(cfg, champ.numeric, champ.categorical, "logit", "sensitivity_logit").fit(sens_tr)
    sens_row = performance_row(data.valid["target"].astype(int), sens.predict(data.valid), "valid_clean | train 2012-14 only")

    psi_t = psi_tables(models, data)

    td = data.train[data.train["target"] == 1]
    el = fit_lgd_ead_proxy(td)
    v = data.valid.assign(pd_hat=champ.predict(data.valid))
    trade = tradeoff_table(v, "pd_hat", el["lgd"], el["ccf"], cfg["decision"]["approval_grid"], cfg["decision"]["el_appetite"])
    rec = recommend(trade)
    ks_cut = ks_optimal_cutoff(v["target"].astype(int), v["pd_hat"])

    res = dict(cfg=cfg, data=data, models=models, perf=perf, cal=cal, smote=smote, sens=sens_row, psi=psi_t,
               el=el, trade=trade, rec=rec, ks_cutoff=ks_cut,
               iv=champ.binner.iv_, coefs=pd.DataFrame({"feature": champ.features, "coef": champ.model.coef_[0]}))
    if make_reports:
        from src.reporting import write_all
        write_all(res)
    return res


if __name__ == "__main__":
    r = run_all()
    print(r["perf"].query("pd_type=='calibrated'").round(4).to_string())
