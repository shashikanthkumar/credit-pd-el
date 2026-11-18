"""Writes figures, threshold_tradeoff.csv, model_card.md and validation_memo.md from ACTUAL results.
Numbers are never hand-typed: every figure in the reports comes from the run that produced them."""
from datetime import date
import pandas as pd
from src.utils.config import abs_path
from src.data.loader import load_audit
from src.evaluation import plots


def md(df: pd.DataFrame, floatfmt=4) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join(["---"] * len(d.columns)) + "|\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False))


def vintage_table(df):
    g = df.groupby("vintage_year").agg(loans=("target", "size"), resolved=("is_resolved", "sum"),
                                       defaults=("target", "sum"), matured=("fully_matured", "mean")).reset_index()
    g["resolved_share"] = g["resolved"] / g["loans"]
    g["default_rate_resolved"] = g["defaults"] / g["resolved"]
    return g


def write_all(res: dict) -> None:
    cfg, d = res["cfg"], res["data"]
    fig, rep = abs_path(cfg, "figures_dir"), abs_path(cfg, "reports_dir")
    fig.mkdir(exist_ok=True, parents=True); rep.mkdir(exist_ok=True, parents=True)
    perf, m = res["perf"], res["models"]
    yv = d.valid["target"].astype(int)

    plots.roc_plot({n: (yv, m[n].predict(d.valid)) for n in ["champion_logit", "challenger_lgbm", "grade_only_logit"]}, fig / "roc_valid.png")
    c = res["cal"]["champion_logit"]
    plots.calibration_plot({"raw": c["decile_raw"], "train-only Platt": c["decile_cal"]}, fig / "calibration_champion.png", "Champion calibration by decile, 2015 OOT")
    plots.tradeoff_plot(res["trade"], cfg["decision"]["el_appetite"], fig / "el_vs_approval.png")
    vt = vintage_table(d.df); plots.vintage_plot(vt, fig / "vintage_default_rate.png")
    plots.psi_plot(res["psi"]["score"].set_index("sample")["score_psi"], fig / "score_psi.png")
    res["trade"].to_csv(rep / "threshold_tradeoff.csv", index=False)

    ptab = (perf[(perf["pd_type"] == "calibrated") & (perf["sample"] != "train (in-sample)")]
            [["model", "sample", "n", "default_rate", "mean_pd", "auc", "gini", "ks", "brier"]])
    rec, el = res["rec"], res["el"]
    aud = d.valid_audit
    demoted = list(aud.loc[~aud["clean"], "issue_month"])
    la = load_audit(cfg)
    load_txt = (f"Raw rows parsed by DuckDB: {la['n_raw_rows_parsed']:,} (assertion floor {la['min_raw_rows']:,}); model_frame rows (36m): {la['n_model_frame_rows']:,}. Full counts by status in `load_audit.json`."
                if la else "No load audit found (database was built before the audit existed; rebuild with `make sql`).")
    demoted_txt = (f"Issue months demoted from the clean year to the flagged sample because they were not fully matured or <{cfg['split'].get('clean_min_resolved_share', .95):.0%} resolved: **{', '.join(demoted)}**."
                   if demoted else "All issue months of the clean year passed the maturity/resolution audit.")
    tr = cfg["split"]["train_years"]
    (rep / "model_card.md").write_text(f"""# Model card: PD model, Lending Club 36-month loans

_Generated {date.today()} by `src/reporting.py` from the run's actual outputs. Scope: **PD model, not a Basel/IRB framework. LGD/EAD are illustrative proxies.**_

## Intended use
Rank-order and estimate 36-month default probability for accepted consumer loans on a public dataset, to demonstrate PD development, out-of-time validation, calibration and an EL-vs-approval-rate decision. **Not** for production lending or regulatory capital.

## Data and population
36-month loans only (uniform maturity window). Target: Charged Off/Default = 1, Fully Paid = 0; Current/Late/In Grace excluded.
Train {tr[0]}-{tr[1]} (n={len(d.train):,}); clean validation 2015 (n={len(d.valid):,}); flagged-censored sample = 2016 plus any demoted 2015 months (n={len(d.flagged):,}); 2017-2018 used for PSI only. {demoted_txt}

## Models
- **Champion**: logistic regression on monotone WoE bins, features: {", ".join(m["champion_logit"].features)}; class_weight=`{cfg["model"]["class_weight"]}` then train-only out-of-fold Platt recalibration.
- **Challenger**: LightGBM with monotone constraints, same recalibration.
- **Benchmarks**: LC grade only; core + LC int_rate/grade ("with_lc") to measure incremental lift over LC's own pricing.

## Performance (calibrated PD, out-of-time)
{md(ptab)}

Calibration summary, champion, 2015:

{md(c["summary"])}

## Threshold (illustrative)
Assumed EL appetite {cfg["decision"]["el_appetite"]:.1%} (an input, not a bank policy). Largest approval rate within appetite: **{rec["approval_rate"]:.0%}** (PD cutoff {rec["pd_cutoff"]:.3f}, model EL rate {rec["el_rate_model"]:.2%}).
Proxies: LGD={el["lgd"]:.2f}, CCF={el["ccf"]:.2f}. KS-optimal PD cutoff for reference: {res["ks_cutoff"]:.3f}.

## Limitations
See `validation_memo.md` section 6 and the README.
""")

    (rep / "validation_memo.md").write_text(f"""# Validation memo (independent-style review of the PD model)

_Generated {date.today()}; every number below comes from the run that produced this file._

## 1. Scope and conclusions
PD model with out-of-time validation. LGD/EAD are proxies. Conclusions are about **this public dataset of accepted loans**, not a bank book.

## 2. Population, target and censoring
The snapshot is a single 2018Q4 cut, so late vintages are right-censored. We restrict to 36-month loans and treat only 2015 as fully matured. Vintage view:

{md(vt[["vintage_year", "loans", "resolved_share", "default_rate_resolved", "matured"]])}

![vintage](../figures/vintage_default_rate.png)

**Clean-validation maturity audit (per issue month of the nominal clean year).** A month enters the clean sample only if fully matured and resolved share >= {cfg["split"].get("clean_min_resolved_share", .95):.0%}.

{md(aud)}

{demoted_txt}

**Load audit.** {load_txt}

## 3. Discrimination (calibrated PD)
{md(ptab)}

Sensitivity, champion trained on 2012-14 only: AUC {res["sens"]["auc"]:.4f}, Gini {res["sens"]["gini"]:.4f}, KS {res["sens"]["ks"]:.4f}, Brier {res["sens"]["brier"]:.4f} (2015).

![roc](../figures/roc_valid.png)

## 4. Calibration, 2015 champion
{md(c["summary"])}

Decile table (train-only Platt):

{md(c["decile_cal"])}

The ORACLE row uses validation outcomes and is a diagnostic: the gap between it and the deployable row is the part of miscalibration due to base-rate shift.

![cal](../figures/calibration_champion.png)

## 5. SMOTE vs class weights
{md(res["smote"])}

Both rebalancing methods shift the *level* of predicted PD upwards (see `over_prediction_ratio`: mean predicted PD / observed default rate), because the model is trained as if defaults were ~50% of the book. Provision = PD x LGD x EAD is linear in PD, so an unrecalibrated level bias goes straight into the provision. For class weights this is a pure logit-intercept shift and the train-only Platt step removes it. SMOTE has the same level problem, and in addition fabricates synthetic defaults by interpolating in WoE space, which can distort the slope/shape of the score-to-PD relation and not just its level; the ranking benefit (AUC column) should therefore be judged against the no-weights model, not assumed. Decision: class weights + train-only recalibration; SMOTE kept as a documented negative-result test.

## 6. Distribution shift (PSI): NOT a performance metric
PSI compares score/feature distributions between train and another sample using no outcomes. A low PSI does not imply good performance and a high PSI does not imply poor performance.

{md(res["psi"]["score"])}

Feature PSI vs train:

{md(res["psi"]["features"].reset_index())}

![psi](../figures/score_psi.png)

## 7. Expected-loss decision layer (illustrative)
LGD {el["lgd"]:.3f} and CCF {el["ccf"]:.3f} are proxies from train defaults (recoveries and balance at default; not discounted; no downturn).

{md(res["trade"])}

![tradeoff](../figures/el_vs_approval.png)

## 8. Limitations
- Accepted loans only: no reject inference, so the model describes approved applicants, not the through-the-door population.
- `ignore_errors=true` can drop malformed CSV rows silently; mitigated by a row-count floor and `load_audit.json`, but the loss cannot be proven zero.
- Single snapshot, not a performance panel; no behavioural or macro covariates; the 2015 clean window is one vintage (macro regime of one year).
- LC grade/int_rate encode LC's own model (circular), so excluded from the champion; "with_lc" rows show incremental lift only.
- LGD/EAD are proxies; EL appetite is an assumed input; no Basel/IRB (no downturn LGD, margin of conservatism, or PD floors).
- Platt/class-weight recalibration restores the training base rate; it cannot anticipate a changed base rate. Monitor the observed/expected ratio.
- Recoveries in the snapshot are undiscounted.
""")
