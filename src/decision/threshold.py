"""Threshold selection on an expected-loss vs approval-rate trade-off (NOT F1)."""
import numpy as np
import pandas as pd
from src.evaluation.metrics import ks as ks_stat


def tradeoff_table(df: pd.DataFrame, pd_col: str, lgd: float, ccf: float, grid, el_appetite: float) -> pd.DataFrame:
    """Approve the lowest-PD `a` share of applicants. EL uses PROXY LGD/EAD. All rates are exposure-weighted
    on loan_amnt. 'Realised' loss rate applies the same proxy LGD/CCF to observed defaults."""
    d = df.sort_values(pd_col).reset_index(drop=True)
    tot_def = d["target"].sum()
    rows = []
    for a in grid:
        k = max(int(round(a * len(d))), 1)
        ap, rj = d.iloc[:k], d.iloc[k:]
        expo = ap["loan_amnt"].sum()
        rows.append(dict(
            approval_rate=k / len(d),
            pd_cutoff=ap[pd_col].iloc[-1],
            approved_n=k,
            approved_default_rate=ap["target"].mean(),
            approved_mean_pd=ap[pd_col].mean(),
            el_rate_model=(ap[pd_col] * lgd * ccf * ap["loan_amnt"]).sum() / expo,
            loss_rate_realised_proxy=(ap["target"] * lgd * ccf * ap["loan_amnt"]).sum() / expo,
            defaults_rejected_share=(rj["target"].sum() / tot_def) if len(rj) else 0.0,
            good_loans_rejected_share=((1 - rj["target"]).sum() / (1 - d["target"]).sum()) if len(rj) else 0.0,
        ))
    t = pd.DataFrame(rows)
    t["within_appetite"] = t["el_rate_model"] <= el_appetite
    return t


def recommend(t: pd.DataFrame) -> pd.Series:
    ok = t[t["within_appetite"]]
    return (ok.iloc[ok["approval_rate"].argmax()] if len(ok) else t.iloc[t["el_rate_model"].argmin()])


def ks_optimal_cutoff(y, p) -> float:
    """PD cutoff at the max KS gap (reference only: it ignores exposure, LGD and risk appetite)."""
    y, p = np.asarray(y), np.asarray(p)
    o = np.argsort(p); y, p = y[o], p[o]
    gap = np.abs(np.cumsum(y) / y.sum() - np.cumsum(1 - y) / (1 - y).sum())
    return float(p[int(gap.argmax())])
