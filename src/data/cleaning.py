"""Cleaning with documented missing-value rules (see docs/decision_log.md D-04).

Rules (all limits are learned on TRAIN only and re-applied, never refit on validation):
  R1 emp_length: parsed to ordered labels; missing or literal-null strings ('', 'nan', 'None', 'null', 'n/a')
     -> 'Unknown' (informative-missing category). Unmapped non-null values also fall to 'Unknown' deliberately.
  R2 revol_util, dti, delinq_2yrs, fico_mid, annual_inc: NaN is kept; the WoE binner gives NaN its own bin.
     No mean/median imputation (it would hide the missingness signal and distort bin WoE).
  R3 impossible values -> NaN: dti < 0 or dti > 100(*), annual_inc <= 0, revol_util < 0, loan_amnt <= 0.
  R4 winsorise continuous features at train [0.5%, 99.5%] quantiles so extreme tails do not drive bin edges.
  R5 rare categories (< min_bin_frac of train) are pooled to 'OTHER' inside the binner.
  (*) dti of 999 is a known Lending Club sentinel.
"""
import numpy as np
import pandas as pd

EMP_MAP = {"< 1 year": "0-1y", "1 year": "0-1y", "2 years": "2-4y", "3 years": "2-4y", "4 years": "2-4y",
           "5 years": "5-9y", "6 years": "5-9y", "7 years": "5-9y", "8 years": "5-9y", "9 years": "5-9y",
           "10+ years": "10y+"}
NULL_STRINGS = {"", "nan", "none", "null", "n/a"}
WINSOR_COLS = ["loan_amnt", "annual_inc", "dti", "revol_util", "loan_to_income"]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_emp = df["emp_length_raw"].astype("string").str.strip()
    raw_emp = raw_emp.mask(raw_emp.str.lower().isin(NULL_STRINGS), pd.NA)   # "nan"/"None"/"null" literals
    df["emp_length"] = raw_emp.map(EMP_MAP).astype(object).fillna("Unknown")
    df.loc[(df["dti"] < 0) | (df["dti"] > 100), "dti"] = np.nan
    df.loc[df["annual_inc"] <= 0, "annual_inc"] = np.nan
    df.loc[df["revol_util"] < 0, "revol_util"] = np.nan
    df.loc[df["loan_amnt"] <= 0, "loan_amnt"] = np.nan
    df["loan_to_income"] = np.where(df["annual_inc"] > 0, df["loan_amnt"] / df["annual_inc"], np.nan)
    return df


def fit_winsor(train: pd.DataFrame, q=(0.005, 0.995)) -> dict:
    return {c: (train[c].quantile(q[0]), train[c].quantile(q[1])) for c in WINSOR_COLS if c in train}


def apply_winsor(df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    df = df.copy()
    for c, (lo, hi) in limits.items():
        df[c] = df[c].clip(lo, hi)
    return df


def missingness_report(df: pd.DataFrame, cols: list, by: str = "vintage_year") -> pd.DataFrame:
    return df.groupby(by)[cols].agg(lambda s: s.isna().mean()).round(4)
