"""Regression tests for the model-review findings (decision_log D-16 .. D-20)."""
import copy
import duckdb
import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config
from src.features.woe import WoEBinner, POOLED
from src.evaluation.metrics import psi
from src.data.cleaning import basic_clean
from src.data.loader import _read_sql, build_database, load_audit
from src.pipeline import maturity_audit

COLS = ["loan_status", "loan_amnt", "funded_amnt", "int_rate", "grade", "sub_grade", "term", "annual_inc", "dti",
        "emp_length", "home_ownership", "purpose", "fico_range_low", "fico_range_high", "delinq_2yrs", "revol_util",
        "issue_d", "out_prncp", "recoveries", "total_rec_prncp"]


# ---- 1. OTHER vs pooled sentinel -------------------------------------------------------------
def _cat_frame(other_share, n=40000, seed=0):
    r = np.random.default_rng(seed)
    levels = ["RENT", "MORTGAGE", "OWN", "OTHER", "ANY"]
    p = np.array([0.0, 0.40, 0.10, other_share, 0.01])
    p[0] = 1 - p.sum()
    cat = r.choice(levels, n, p=p)
    risk = {"RENT": .25, "MORTGAGE": .15, "OWN": .20, "OTHER": .45, "ANY": .05}
    y = (r.random(n) < pd.Series(cat).map(risk).to_numpy()).astype(int)
    return pd.DataFrame({"home": cat}), pd.Series(y)


def test_real_OTHER_category_gets_own_bin_not_pooled_bin():
    X, y = _cat_frame(other_share=0.10)          # OTHER clears the 5% threshold; ANY (1%) is rare
    b = WoEBinner(categorical=["home"], min_bin_frac=0.05).fit(X, y)
    t = b.bin_table("home").set_index("bin")
    assert "OTHER" in t.index and POOLED in t.index         # distinct bins
    assert t.loc["OTHER", "n"] == (X["home"] == "OTHER").sum()   # NOT blended with pooled rows
    assert t.loc[POOLED, "n"] == (X["home"] == "ANY").sum()
    assert t.loc["OTHER", "woe"] != t.loc[POOLED, "woe"]
    W = b.transform(X)["home_woe"]
    assert W[X["home"] == "OTHER"].nunique() == 1 and W[X["home"] == "ANY"].nunique() == 1
    assert W[X["home"] == "OTHER"].iloc[0] != W[X["home"] == "ANY"].iloc[0]


def test_rare_real_OTHER_is_pooled_under_the_sentinel_label():
    X, y = _cat_frame(other_share=0.01)
    t = WoEBinner(categorical=["home"], min_bin_frac=0.05).fit(X, y).bin_table("home")
    assert "OTHER" not in set(t["bin"]) and POOLED in set(t["bin"])


def test_reserved_pooling_label_in_data_is_rejected():
    X, y = _cat_frame(0.10)
    X.loc[:5, "home"] = POOLED
    with pytest.raises(AssertionError):
        WoEBinner(categorical=["home"]).fit(X, y)


# ---- 2. raw-load row-count assertion ----------------------------------------------------------
def _write_csv(path, n_good, malformed=3):
    rows = [",".join(["Fully Paid", "1000", "1000", "10.0", "A", "A1", " 36 months", "50000", "10", "5 years", "RENT",
                      "other", "700", "704", "0", "30", "Dec-2014", "0", "0", "1000"])] * n_good
    bad = ["only,three,cols"] * malformed
    path.write_text(",".join(COLS) + "\n" + "\n".join(rows[:5] + bad + rows[5:]) + "\n")


def _cfg(tmp_path, csv, min_rows):
    cfg = copy.deepcopy(load_config("configs/config.yaml"))
    cfg["paths"].update(raw_csv=str(csv), duckdb=str(tmp_path / "t.duckdb"), reports_dir=str(tmp_path / "rep"))
    cfg["loader"] = {"min_raw_rows": min_rows}
    return cfg


def test_loader_fails_loudly_and_deletes_db_when_rows_are_lost(tmp_path):
    csv = tmp_path / "raw.csv"; _write_csv(csv, n_good=100)
    cfg = _cfg(tmp_path, csv, min_rows=103)      # a naive 'rows in file' expectation: malformed rows were dropped
    with pytest.raises(AssertionError, match="lost rows"):
        build_database(cfg, rebuild=True)
    assert not (tmp_path / "t.duckdb").exists()  # a bad load cannot be silently reused


def test_loader_writes_audit_when_load_is_ok(tmp_path):
    csv = tmp_path / "raw.csv"; _write_csv(csv, n_good=100)
    cfg = _cfg(tmp_path, csv, min_rows=100)
    build_database(cfg, rebuild=True)
    a = load_audit(cfg)
    assert a["n_raw_rows_parsed"] == 100 and a["raw_rows_by_loan_status"]["Fully Paid"] == 100
    assert a["ratio_to_min"] == 1.0


# ---- 3. maturity logic pinned against explicit dates, running the REAL SQL ---------------------
def _matured(issue_d, term, snapshot):
    cfg = copy.deepcopy(load_config("configs/config.yaml"))
    cfg["population"]["snapshot_date"] = snapshot
    cfg["population"]["term_months"] = term
    con = duckdb.connect()
    con.execute("CREATE TABLE raw_loans (" + ",".join(f"{c} VARCHAR" for c in COLS) + ")")
    vals = ["Fully Paid", "1000", "1000", "10", "A", "A1", f" {term} months", "5e4", "10", "5 years", "RENT", "other",
            "700", "704", "0", "30", issue_d, "0", "0", "1000"]
    con.execute("INSERT INTO raw_loans VALUES (" + ",".join("?" * len(COLS)) + ")", vals)
    con.execute(_read_sql("03_model_frame.sql", cfg))
    return bool(con.execute("SELECT fully_matured FROM model_frame").fetchone()[0])


def test_fully_matured_boundary():
    assert _matured("Nov-2015", 36, "2018-12-31")
    assert _matured("Dec-2015", 36, "2018-12-31")          # issue 2015-12-01 + 36m = 2018-12-01 <= snapshot
    assert not _matured("Jan-2016", 36, "2018-12-31")
    assert not _matured("Dec-2015", 36, "2018-11-30")      # the check is real: an earlier snapshot flips it
    assert not _matured("Dec-2013", 60, "2018-11-30") and _matured("Dec-2013", 60, "2018-12-31")


def test_config_snapshot_is_end_of_data_not_first_of_month():
    assert load_config("configs/config.yaml")["population"]["snapshot_date"] == "2018-12-31"


def test_maturity_audit_demotes_unresolved_edge_month():
    df = pd.DataFrame({
        "vintage_year": [2015] * 40,
        "issue_month": ["2015-10"] * 20 + ["2015-12"] * 20,
        "is_resolved": [True] * 20 + [True] * 15 + [False] * 5,      # Dec-2015: only 75% resolved
        "fully_matured": [True] * 40})
    a = maturity_audit(df, (2015, 2015), 0.95).set_index("issue_month")
    assert a.loc["2015-10", "clean"] and not a.loc["2015-12", "clean"]
    assert np.isclose(a.loc["2015-12", "resolved_share"], 0.75)


# ---- 5. PSI low-resolution fallback ------------------------------------------------------------
def test_psi_not_zero_by_construction_for_skewed_discrete_feature():
    r = np.random.default_rng(0)
    base = r.choice([0, 1, 2, 3], 50000, p=[.92, .05, .02, .01])
    shifted = r.choice([0, 1, 2, 3], 50000, p=[.70, .18, .08, .04])      # a real shift
    same = r.choice([0, 1, 2, 3], 50000, p=[.92, .05, .02, .01])
    assert psi(base, shifted) > 0.25            # would be exactly 0 with collapsed quantile bins
    assert psi(base, same) < 0.01
    assert psi(base, base) == pytest.approx(0, abs=1e-9)


# ---- 6. null-string normalisation ----------------------------------------------------------------
def test_emp_length_literal_null_strings_map_to_unknown():
    raw = pd.DataFrame({"emp_length_raw": ["10+ years", "nan", "None", " null ", "", None, "weird", "< 1 year"],
                        "dti": 10.0, "annual_inc": 5e4, "revol_util": 30.0, "loan_amnt": 1e3})
    out = basic_clean(raw)["emp_length"].tolist()
    assert out == ["10y+", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown", "0-1y"]
