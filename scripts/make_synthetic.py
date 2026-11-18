"""SYNTHETIC Lending-Club-shaped data for smoke-testing the pipeline and CI ONLY.

Nothing produced from this file says anything about real credit performance.
Mimics: schema, 36/60m terms, vintages 2007-2018, right-censoring of late vintages
(unresolved loans are 'Current'; those resolved early skew to default).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    year_p = np.array([.002, .005, .01, .02, .04, .07, .12, .17, .22, .2, .09, .05])  # 2007..2018
    year = rng.choice(np.arange(2007, 2019), n, p=year_p / year_p.sum())
    month = rng.integers(1, 13, n)
    issue = pd.to_datetime(dict(year=year, month=month, day=1))
    term = rng.choice([36, 60], n, p=[.75, .25])
    fico = np.clip(rng.normal(695, 30, n), 610, 845).round(-0) // 5 * 5
    dti = np.clip(rng.gamma(4, 5, n), 0, 60)
    inc = np.exp(rng.normal(11.1, .55, n))
    amt = np.clip(rng.gamma(4, 3500, n), 1000, 40000).round(-2)
    revol = np.clip(rng.normal(52, 25, n), 0, 120)
    delinq = rng.poisson(.3, n).astype(float)
    purposes = ["debt_consolidation", "credit_card", "home_improvement", "small_business", "major_purchase", "other"]
    purpose = rng.choice(purposes, n, p=[.55, .2, .07, .04, .05, .09])
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[.42, .5, .08])
    emp = rng.choice(["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years", np.nan], n,
                     p=[.08, .07, .1, .1, .1, .5, .05])
    drift = (year - 2012) * 0.03  # mild population drift over time
    z = (-1.75 - 0.018 * (fico - 695) + 0.03 * (dti - 18) + 0.45 * (amt / inc) * 5
         + 0.006 * (revol - 50) + 0.15 * delinq + 0.35 * (purpose == "small_business")
         - 0.2 * (home == "MORTGAGE") + 0.35 * (term == 60) + drift)
    p_def = 1 / (1 + np.exp(-z))
    default = rng.random(n) < p_def
    pd_lc = 1 / (1 + np.exp(-(z + rng.normal(0, .35, n))))
    int_rate = np.clip(6 + 60 * pd_lc ** .8 + rng.normal(0, 1, n), 5.3, 30.9).round(2)
    grade = pd.cut(int_rate, [0, 8, 11, 14, 17, 21, 25, 40], labels=list("ABCDEFG")).astype(str)
    sub = grade + (pd.cut(int_rate % 3, [-1, .6, 1.2, 1.8, 2.4, 3.1], labels=list("12345")).astype(str))
    # maturity / censoring
    snap = pd.Timestamp("2018-12-01")
    mob = ((snap.year - issue.dt.year) * 12 + (snap.month - issue.dt.month)).to_numpy()
    t_def = rng.integers(5, 31, n) * (term / 36)
    t_pre = np.where(rng.random(n) < .4, rng.integers(12, 36, n) * (term / 36), term)
    t_event = np.where(default, t_def, t_pre)
    resolved = (t_event <= mob) | (mob >= term)
    status = np.where(resolved, np.where(default, "Charged Off", "Fully Paid"), "Current")
    funded = amt
    bal_frac = np.clip(1 - t_def / term, .1, 1)
    rec = np.where(status == "Charged Off", amt * bal_frac * np.clip(rng.beta(1.5, 8, n), 0, 1), 0)
    rec_prncp = np.where(status == "Charged Off", amt * (1 - bal_frac), np.where(status == "Fully Paid", amt, amt * .5))
    df = pd.DataFrame({
        "loan_status": status, "loan_amnt": amt, "funded_amnt": funded,
        "int_rate": int_rate, "grade": grade, "sub_grade": sub, "term": [f" {t} months" for t in term],
        "annual_inc": inc.round(0), "dti": dti.round(2), "emp_length": emp, "home_ownership": home,
        "purpose": purpose, "fico_range_low": fico, "fico_range_high": fico + 4, "delinq_2yrs": delinq,
        "revol_util": revol.round(1), "issue_d": issue.dt.strftime("%b-%Y"),
        "out_prncp": np.where(status == "Current", amt * .4, 0), "recoveries": rec.round(2),
        "total_rec_prncp": rec_prncp.round(2),
    })
    # realistic missingness
    df.loc[rng.random(n) < .003, "revol_util"] = np.nan
    df.loc[rng.random(n) < .002, "dti"] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/raw/SYNTHETIC_accepted.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make(a.n, a.seed).to_csv(a.out, index=False)
    print("wrote", a.out)
