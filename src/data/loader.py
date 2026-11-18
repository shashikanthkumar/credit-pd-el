"""DuckDB SQL layer: raw CSV -> typed model_frame -> pandas."""
from pathlib import Path
import json
import duckdb
import pandas as pd
from src.utils.config import ROOT, abs_path


def _read_sql(name: str, cfg: dict) -> str:
    sql = (ROOT / "sql" / name).read_text()
    return (sql.replace("{{RAW_CSV}}", str(abs_path(cfg, "raw_csv")))
               .replace("{{TERM}}", str(cfg["population"]["term_months"]))
               .replace("{{SNAPSHOT}}", cfg["population"]["snapshot_date"]))


def audit_path(cfg: dict) -> Path:
    return abs_path(cfg, "reports_dir") / "load_audit.json"


def build_database(cfg: dict, rebuild: bool = False) -> Path:
    """Build DuckDB from the raw CSV. `ignore_errors=true` skips structurally malformed rows (e.g. the free-text
    `desc` column with embedded quotes/newlines), and that loss is silent and possibly non-random. So after
    loading we (1) assert a minimum row count (config loader.min_raw_rows), (2) write reports/load_audit.json
    with raw counts by status, and (3) delete the half-built database if the assertion fails, so a bad load
    can never be silently reused by the next call."""
    db = abs_path(cfg, "duckdb")
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists() and not rebuild:
        return db
    if db.exists():
        db.unlink()
    con = duckdb.connect(str(db))
    try:
        con.execute(_read_sql("01_create_schema.sql", cfg))
        con.execute(_read_sql("02_load_raw.sql", cfg))
        n_raw = con.execute("SELECT count(*) FROM raw_loans").fetchone()[0]
        min_rows = (cfg.get("loader") or {}).get("min_raw_rows")
        status = dict(con.execute("SELECT coalesce(loan_status,'<NULL>'), count(*) FROM raw_loans GROUP BY 1 ORDER BY 2 DESC").fetchall())
        if min_rows is not None and n_raw < int(min_rows):
            raise AssertionError(f"raw load lost rows: parsed {n_raw:,} < loader.min_raw_rows {int(min_rows):,}. "
                                 f"Inspect the CSV (embedded quotes/newlines?) before modelling.")
        con.execute(_read_sql("03_model_frame.sql", cfg))
        n_frame = con.execute("SELECT count(*) FROM model_frame").fetchone()[0]
        audit = dict(n_raw_rows_parsed=int(n_raw), min_raw_rows=min_rows,
                     ratio_to_min=(n_raw / min_rows) if min_rows else None,
                     n_model_frame_rows=int(n_frame), raw_rows_by_loan_status={k: int(v) for k, v in status.items()})
        audit_path(cfg).parent.mkdir(parents=True, exist_ok=True)
        audit_path(cfg).write_text(json.dumps(audit, indent=2))
    except Exception:
        con.close()
        if db.exists():
            db.unlink()
        raise
    con.close()
    return db


def load_audit(cfg: dict) -> dict:
    p = audit_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {}


def load_model_frame(cfg: dict, rebuild: bool = False) -> pd.DataFrame:
    db = build_database(cfg, rebuild)
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute("SELECT * FROM model_frame").df()
    finally:
        con.close()


def sql_query(cfg: dict, query: str) -> pd.DataFrame:
    con = duckdb.connect(str(build_database(cfg)), read_only=True)
    try:
        return con.execute(query).df()
    finally:
        con.close()
