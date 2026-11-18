"""PD models. Champion: logistic regression on WoE inputs. Challenger: LightGBM with monotone constraints."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Sign of the economically expected effect on default; 0 = unconstrained.
MONOTONE = {"fico_mid": -1, "dti": 1, "revol_util": 1, "delinq_2yrs": 1, "loan_to_income": 1,
            "annual_inc": -1, "int_rate": 1}


def fit_logit(Xw: pd.DataFrame, y, cfg: dict, class_weight="cfg") -> LogisticRegression:
    cw = cfg["model"]["class_weight"] if class_weight == "cfg" else class_weight
    m = LogisticRegression(C=cfg["model"]["logit_C"], class_weight=None if cw in (None, "none") else cw,
                           max_iter=2000, solver="lbfgs")
    return m.fit(Xw, np.asarray(y))


def logit_coef_table(model: LogisticRegression, cols) -> pd.DataFrame:
    t = pd.DataFrame({"feature": list(cols), "coef": model.coef_[0]})
    t["sign_ok"] = t["coef"] > 0           # WoE = ln(bad/good) -> expect positive
    return t


def vif_table(Xw: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = Xw.assign(const=1.0)
    return pd.DataFrame({"feature": Xw.columns,
                         "vif": [variance_inflation_factor(X.values, i) for i in range(Xw.shape[1])]})


class GBMPrep:
    """Raw-feature matrix for LightGBM: categoricals get a fixed (train) category set."""

    def __init__(self, numeric, categorical):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.levels_ = {}

    def fit(self, df):
        for c in self.categorical:
            self.levels_[c] = sorted(df[c].astype(str).unique())
        return self

    def transform(self, df):
        X = df[self.numeric].copy()
        for c in self.categorical:
            X[c] = pd.Categorical(df[c].astype(str), categories=self.levels_[c])
        return X

    def constraints(self):
        return [MONOTONE.get(c, 0) for c in self.numeric] + [0] * len(self.categorical)


def make_lgbm(cfg: dict, prep: GBMPrep, class_weight="cfg"):
    import lightgbm as lgb
    p = dict(cfg["model"]["lgbm"])
    cw = cfg["model"]["class_weight"] if class_weight == "cfg" else class_weight
    return lgb.LGBMClassifier(**p, class_weight=None if cw in (None, "none") else cw,
                              monotone_constraints=prep.constraints(), monotone_constraints_method="basic",
                              random_state=cfg["seed"], deterministic=True, force_row_wise=True,
                              n_jobs=4, verbose=-1)


def oof_proba(model, X, y, cfg: dict) -> np.ndarray:
    """Out-of-fold probabilities used ONLY to fit the Platt calibrator, never to estimate performance.
    Shuffled StratifiedKFold is acceptable for that purpose: the calibrator needs scores that reflect the
    training-period base rate and out-of-sample score behaviour, not an unbiased forecast of future
    performance. (A row's OOF score may come from a model that saw later training years.)
    If this function is ever reused for evaluation, switch to a time-ordered split (TimeSeriesSplit)."""
    cv = StratifiedKFold(cfg["model"]["oof_folds"], shuffle=True, random_state=cfg["seed"])
    return cross_val_predict(model, X, np.asarray(y), cv=cv, method="predict_proba")[:, 1]


def smote_experiment(Xw_tr, y_tr, Xw_va, y_va, cfg: dict) -> pd.DataFrame:
    """Compare no-weights / class-weights / SMOTE on discrimination AND level calibration.
    SMOTE interpolates synthetic defaults, so the model sees a ~50% default rate. Predicted PDs are then
    inflated relative to the true base rate; since provisioning (EL = PD x LGD x EAD) is linear in PD,
    that bias flows 1:1 into the provision unless recalibrated -- and recalibrating on the real base rate
    removes most of SMOTE's claimed benefit."""
    from sklearn.metrics import roc_auc_score, brier_score_loss
    rows = []
    y_tr, y_va = np.asarray(y_tr), np.asarray(y_va)
    variants = {"no_weights": fit_logit(Xw_tr, y_tr, cfg, class_weight=None),
                "class_weight_balanced": fit_logit(Xw_tr, y_tr, cfg, class_weight="balanced")}
    try:
        from imblearn.over_sampling import SMOTE
        Xs, ys = SMOTE(random_state=cfg["seed"]).fit_resample(Xw_tr, y_tr)
        variants["smote"] = fit_logit(Xs, ys, cfg, class_weight=None)
    except ImportError:
        pass
    for name, m in variants.items():
        p = m.predict_proba(Xw_va)[:, 1]
        rows.append(dict(variant=name, valid_auc=roc_auc_score(y_va, p), valid_brier=brier_score_loss(y_va, p),
                         mean_pred_pd=p.mean(), observed_dr=y_va.mean(), over_prediction_ratio=p.mean() / y_va.mean()))
    return pd.DataFrame(rows)
