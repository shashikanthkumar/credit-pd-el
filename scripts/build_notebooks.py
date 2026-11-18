"""Generates notebooks/*.ipynb (unexecuted). Edit here, then `python scripts/build_notebooks.py`."""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.utils.config import load_config, set_seeds
cfg = load_config(); set_seeds(cfg["seed"])
pd.set_option("display.float_format", "{:.4f}".format); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
print("config seed:", cfg["seed"], "| raw file:", cfg["paths"]["raw_csv"])'''


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    n.cells = [nbf.v4.new_code_cell(SETUP)] + [
        (nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])) for c in cells]
    n.cells.insert(0, nbf.v4.new_markdown_cell(cells[0][1])); n.cells.pop(2)  # title first, setup second
    nbf.write(n, OUT / name)


M = lambda s: ("md", s)
C = lambda s: ("code", s)

# ---------------------------------------------------------------- 01
nb("01_data_prep_and_sql.ipynb", [
M("""# 01 · Data preparation and SQL layer
**Scope reminder:** PD model on public Lending Club data. Not a Basel/IRB framework; LGD/EAD are illustrative proxies.

**What this notebook does:** loads the raw CSV into DuckDB, builds the typed `model_frame` (`sql/03_model_frame.sql`), applies the status mapping, the 36-month filter and the maturity flag, and documents the missing-value rules.

### Why these choices (see `docs/decision_log.md`)
- **DuckDB, not SQLite/pandas read_csv:** ~2M rows x ~150 columns. DuckDB reads only the columns we select and everything lands as VARCHAR first, so mixed-type columns cannot silently corrupt the load; typing is explicit in SQL.
- **36-month loans only (D-01):** the file is one 2018Q4 snapshot (outcomes, not a panel). 36m loans issued up to Dec-2015 are fully matured; 60m loans only up to Dec-2013. Mixing terms mixes maturity windows and makes default rates incomparable across vintages.
- **Target (D-02):** Charged Off/Default = 1, Fully Paid = 0. Current/Late/In Grace have **unknown** final outcome, so they are excluded from modelling rather than assumed good or bad. They are kept in `model_frame` (target NULL) because PSI needs features, not outcomes."""),
C('''from src.data.loader import build_database, load_model_frame, sql_query
REBUILD = False          # set True after changing anything in sql/ or the raw file
build_database(cfg, rebuild=REBUILD)
sql_query(cfg, "SELECT trim(term) AS term, count(*) AS loans FROM raw_loans GROUP BY 1 ORDER BY 2 DESC")'''),
M("Status mapping: every status and what happens to it."),
M("""**Load audit.** `ignore_errors=true` skips malformed CSV rows silently (e.g. the free-text `desc` column). `build_database` therefore asserts a raw-row floor (`loader.min_raw_rows`), writes `reports/load_audit.json`, and deletes a failed build so it cannot be reused. Check it before trusting anything downstream."""),
C('''from src.data.loader import load_audit
a = load_audit(cfg); print({k: v for k, v in a.items() if k != "raw_rows_by_loan_status"}); pd.Series(a.get("raw_rows_by_loan_status", {}), name="raw_rows").to_frame()'''),
C('''sql_query(cfg, """SELECT loan_status, count(*) AS loans, any_value(target) AS target, any_value(is_resolved) AS is_resolved
                  FROM model_frame GROUP BY 1 ORDER BY 2 DESC""")'''),
M("Vintage, resolution share and maturity flag, straight from SQL."),
C('''sql_query(cfg, """SELECT vintage_year, count(*) AS loans, avg(is_resolved::INT) AS resolved_share,
                         avg(fully_matured::INT) AS fully_matured_share, avg(target) AS default_rate_resolved
                  FROM model_frame GROUP BY 1 ORDER BY 1""")'''),
M("""### Documented missing-value and cleaning rules (`src/data/cleaning.py`)
| Rule | Treatment | Justification |
|---|---|---|
| R1 `emp_length` | ordered buckets; missing -> `Unknown` | missingness is itself informative; avoids dropping rows |
| R2 numeric NaN | **kept**, own WoE bin | imputation would hide the missingness signal; WoE handles it natively |
| R3 impossible values | `dti<0 or >100`, `annual_inc<=0`, `revol_util<0`, `loan_amnt<=0` -> NaN | 999 `dti` is a known sentinel |
| R4 winsorising | train 0.5/99.5 percentiles, re-applied (never refit) | stops tails driving bin edges; no leakage from validation |
| R5 rare categories | pooled to `OTHER` inside the binner (<5% of train) | stable WoE |"""),
C('''from src.data.cleaning import basic_clean, missingness_report
raw = load_model_frame(cfg)
print("rows:", len(raw))
imp = pd.Series({"dti<0 or >100": ((raw.dti < 0) | (raw.dti > 100)).sum(), "annual_inc<=0": (raw.annual_inc <= 0).sum(),
                 "revol_util<0": (raw.revol_util < 0).sum(), "loan_amnt<=0": (raw.loan_amnt <= 0).sum()})
print("impossible values found (set to NaN by R3):"); print(imp.to_string())
df = basic_clean(raw)
cols = cfg["features"]["numeric"] + ["emp_length_raw"]
missingness_report(df, cols).loc[2007:]'''),
])

# ---------------------------------------------------------------- 02
nb("02_eda_and_vintage.ipynb", [
M("""# 02 · EDA, vintage and censoring diagnostics
**Questions this notebook answers**
1. Is the validation window *observable*? (resolved share and maturity by vintage)
2. Does the default rate move by vintage for reasons that are economic vs. **censoring artefacts**?
3. Does missingness or population mix drift between train and validation?
4. Are there outliers that justify the winsorising rule?

### Why censoring dominates the design
A snapshot loan is "resolved" only once it is paid off or charged off. For vintages that are not yet fully matured, the resolved subset over-represents loans that defaulted **early** (and prepaid early), so the observed default rate is biased. Mixing those into a headline validation would distort calibration by decile - and therefore provisioning. Hence: clean validation = 2015 only; 2016 reported separately and labelled; 2017-18 for PSI only."""),
C('''from src.pipeline import prepare
from src.reporting import vintage_table
data = prepare(cfg)
vt = vintage_table(data.df); vt'''),
C('''fig, ax = plt.subplots(figsize=(7,4))
ax.bar(vt.vintage_year, vt.default_rate_resolved*100, color="#4c78a8"); ax.set_ylabel("Default rate, resolved loans (%)")
ax2 = ax.twinx(); ax2.plot(vt.vintage_year, vt.resolved_share*100, "o-", color="#e45756"); ax2.set_ylabel("Resolved share (%)")
for y in (2015.5,): ax.axvline(y, color="k", ls=":")
ax.set_title("Vintage default rate vs. resolved share (36m loans)  |  dotted = end of fully-matured window"); plt.tight_layout(); plt.show()'''),
M("""**Clean-validation maturity audit (issue month).** Snapshot = 2018-12-31; `fully_matured` is explicit date arithmetic (issue + 36 months <= snapshot). A month enters the clean sample only if fully matured **and** >= 95% of its loans are resolved; failing months (typically late-year ones at the window edge, e.g. Dec-2015) are demoted to the flagged sample and reported. Read this table before reading any 2015 result."""),
C('''data.valid_audit'''),
M("**Censoring diagnostic by issue quarter.** If the resolved default rate jumps as the resolved share falls, that is the bias, not credit quality."),
C('''q = data.df[data.df.vintage_year >= 2013].assign(q=lambda d: pd.to_datetime(d.issue_date).dt.to_period("Q").astype(str))
cq = q.groupby("q").agg(loans=("target","size"), resolved_share=("is_resolved","mean"), default_rate_resolved=("target","mean"),
                        matured=("fully_matured","mean")).reset_index()
cq'''),
M("**Missingness by vintage** (train-vs-validation comparability)."),
C('''from src.data.cleaning import missingness_report
missingness_report(data.df, cfg["features"]["numeric"] + ["emp_length_raw"]).loc[2010:]'''),
M("**Population drift:** FICO and DTI medians, loan size, by vintage (all statuses; used for PSI later)."),
C('''data.df.groupby("vintage_year")[["fico_mid","dti","annual_inc","loan_amnt","revol_util"]].median().loc[2010:]'''),
M("""**Outlier review.** Quantiles before vs. after train-fitted winsorising (rule R4). The point is that tail values (e.g. `annual_inc` in the millions, `dti` sentinels) cannot dominate the quantile bin edges."""),
C('''from src.data.cleaning import basic_clean
from src.data.loader import load_model_frame
before = basic_clean(load_model_frame(cfg)); tr_b = before[before.vintage_year.between(*cfg["split"]["train_years"]) & before.is_resolved.astype(bool)]
rows=[]
for c in ["annual_inc","dti","revol_util","loan_to_income"]:
    rows.append({"feature":c, "p0.5_raw":tr_b[c].quantile(.005), "p99.5_raw":tr_b[c].quantile(.995), "max_raw":tr_b[c].max(),
                 "max_after":data.train[c].max()})
pd.DataFrame(rows)'''),
M("**Univariate risk ordering on train** (sanity check: does default rate move the way credit logic says?)"),
C('''tr = data.train.assign(fico_band=pd.cut(data.train.fico_mid,[0,660,680,700,720,740,900]), dti_band=pd.qcut(data.train.dti,5,duplicates="drop"))
for c in ["fico_band","dti_band","emp_length","home_ownership","purpose"]:
    print(tr.groupby(c, observed=True).target.agg(["size","mean"]).rename(columns={"mean":"default_rate"}).round(4), "\\n")'''),
])

# ---------------------------------------------------------------- 03
nb("03_woe_iv_binning.ipynb", [
M("""# 03 · WoE / IV binning
**Method.** From-scratch binner (`src/features/woe.py`): 20 quantile fine bins, then adjacent bins are merged until (a) every bin has >=5% of train and (b) bad-rate is **monotone**. NaN gets its own bin. `WoE = ln(%bad/%good)` with +0.5 smoothing, so a correct logistic model has **positive** coefficients.

### Why this way
- **Monotone coarse classing** makes each coefficient business-explainable and stops the champion from learning noise; a model reviewer can read a score card.
- **Own implementation + cross-check**: `scorecardpy` has had pandas/numpy compatibility issues, so the production path does not depend on it; it is used here only as an optional cross-check of IV.
- **Binning fitted on train only.** Validation is only ever *transformed*. The stability table below applies train bins to 2015 and checks that the risk ordering holds out-of-time.
- **IV thresholds:** drop `IV < 0.02` (no signal); flag `IV > 0.5` (suspiciously strong; investigate leakage). IV is a screening heuristic, not a significance test."""),
C('''from src.pipeline import prepare
from src.features.woe import WoEBinner, bin_stability
from src.features.guards import feature_sets
data = prepare(cfg)
num, cat = feature_sets(cfg)["with_lc"]
y = data.train.target.astype(int)
b = WoEBinner(cfg["binning"]["fine_bins"], cfg["binning"]["min_bin_frac"], num, cat).fit(data.train, y)
iv = b.iv_.to_frame("IV")
iv["decision"] = np.where(iv.IV < cfg["binning"]["iv_min"], "drop (IV too low)", np.where(iv.IV > cfg["binning"]["iv_suspicious"], "FLAG: too strong", "keep"))
iv["circular_LC"] = iv.index.isin(cfg["features"]["lc_numeric"] + cfg["features"]["lc_categorical"])
iv'''),
M("Bin tables for key variables (train)."),
C('''for f in ["fico_mid","dti","revol_util"]:
    print(f); display(b.bin_table(f)[["bin","n","bad_rate","woe","iv"]])'''),
C('''fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, f in zip(axs, ["fico_mid","dti","int_rate"]):
    t = b.bin_table(f); ax.bar(range(len(t)), t.woe, color=np.where(t.woe>0,"#e45756","#4c78a8")); ax.set_xticks(range(len(t))); ax.set_xticklabels(t.bin, rotation=60, ha="right", fontsize=7); ax.set_title(f"WoE: {f}")
plt.tight_layout(); plt.show()'''),
M("**Binning stability out-of-time.** Train bins applied to 2015: does bad-rate keep the same ordering?"),
C('''yv = data.valid.target.astype(int)
for f in ["fico_mid","dti","revol_util"]:
    print(f); display(bin_stability(b, f, data.valid, yv))'''),
M("**Optional cross-check against `scorecardpy`** (skipped automatically if not installed or incompatible)."),
C('''try:
    import scorecardpy as sc
    tmp = data.train[["fico_mid","dti","target"]].dropna().assign(target=lambda d: d.target.astype(int))
    bins = sc.woebin(tmp, y="target", positive="bad|1", print_info=False)
    ref = {k: v["total_iv"].iloc[0] for k, v in bins.items()}
    print(pd.DataFrame({"scorecardpy_IV": ref, "this_repo_IV": {k: b.iv_[k] for k in ref}}))
    print("IVs differ because the binning algorithms differ; the ranking of variables is what should agree.")
except Exception as e:
    print("scorecardpy cross-check skipped:", type(e).__name__, str(e)[:120])'''),
])

# ---------------------------------------------------------------- 04
nb("04_pd_champion_challenger.ipynb", [
M("""# 04 · PD model: champion vs challenger
**Champion:** logistic regression on WoE inputs (core features, **no LC grade/int_rate**).
**Challenger:** LightGBM with monotone constraints (FICO down, DTI/revol_util/delinquencies up, etc.).
**Benchmarks:** LC grade only; core + LC (`with_lc`) for incremental lift.

### Justifications
- **Logit-on-WoE as champion:** interpretable, monotone by construction, standard in credit, easy to validate. GBM is a *challenger*; it replaces the champion only if it earns it (below).
- **Excluding `int_rate`/`grade`/`sub_grade` from the champion:** known at origination (so not temporal leakage) but they encode **LC's own PD model** and pricing; a model that leans on them mostly re-learns LC's score. The `with_lc` row quantifies how much lift they add.
- **Class weights over SMOTE:** class weights rebalance the loss without inventing observations. Either way predicted PD levels are inflated, so we **recalibrate on train out-of-fold scores only** (notebook 05).
- **Monotone constraints in LightGBM:** keeps the challenger economically sensible and comparable to the scorecard; regularised (`min_child_samples=200`, `lambda=5`) to limit overfit to the 2007-14 regime.
- **Banned features** (`funded_amnt`, `out_prncp`, `recoveries`, ...) are blocked by `src/features/guards.py` and unit tests."""),
C('''from src.pipeline import prepare, get_models, evaluate_models
data = prepare(cfg)
models = get_models(data, cfg)       # cached on a config hash; pass refit=True to retrain
champ = models["champion_logit"]
print("champion features:", champ.features)'''),
M("**Champion coefficients** (expect all positive under the `ln(bad/good)` WoE convention) and **VIF**."),
C('''from src.models.pd_models import logit_coef_table, vif_table
X = champ._X(data.train)
logit_coef_table(champ.model, X.columns).merge(vif_table(X), on="feature")'''),
M("**Discrimination across models** (calibrated PD; AUC/Gini/KS are rank-based so identical to raw)."),
C('''perf = evaluate_models(models, data)
p = perf[perf.pd_type=="calibrated"].pivot_table(index="model", columns="sample", values=["auc","gini","ks"])
p'''),
C('''from sklearn.metrics import roc_curve
yv = data.valid.target.astype(int)
fig, ax = plt.subplots(figsize=(5.5,5))
for n in ["grade_only_logit","champion_logit","with_lc_logit","challenger_lgbm"]:
    f,t,_ = roc_curve(yv, models[n].predict(data.valid)); ax.plot(f,t,label=n)
ax.plot([0,1],[0,1],"k:"); ax.legend(); ax.set(title="ROC, 2015 clean OOT validation", xlabel="FPR", ylabel="TPR"); plt.show()'''),
M("""**Decision rule (stated before looking at results):** the challenger replaces the champion only if validation Gini improves by **>= 0.02** *and* it is not materially worse on calibration. Otherwise the scorecard stays: equal power, more transparency."""),
C('''v = perf[(perf.pd_type=="calibrated") & (perf["sample"]=="valid_clean")].set_index("model")
uplift = v.loc["challenger_lgbm","gini"] - v.loc["champion_logit","gini"]
lc_lift = v.loc["with_lc_logit","gini"] - v.loc["champion_logit","gini"]
print(f"Challenger Gini uplift vs champion: {uplift:+.4f}  ->  {'CHALLENGER earns promotion' if uplift>=0.02 else 'KEEP champion'}")
print(f"Incremental Gini from adding LC int_rate/grade: {lc_lift:+.4f}")
tr_gini = perf[(perf.pd_type=="calibrated") & (perf["sample"]=="train (in-sample)")].set_index("model").gini
print("Train->valid Gini decay (larger = more overfit):")
for m_ in ["champion_logit", "challenger_lgbm"]:
    print(f"  {m_}: {tr_gini[m_] - v.loc[m_, 'gini']:.4f}")'''),
M("**Challenger feature importance (gain)**."),
C('''g = models["challenger_lgbm"]
imp = pd.Series(g.model.booster_.feature_importance("gain"), index=g.prep.numeric+g.prep.categorical).sort_values(ascending=False)
(imp/imp.sum()).to_frame("gain_share")'''),
M("""### SMOTE vs class weights (tested once, as a documented negative result)
SMOTE interpolates synthetic defaults in WoE space until defaults are ~50% of the training set. The model then believes defaults are common, so **predicted PDs are inflated relative to the true base rate**. Provision = PD x LGD x EAD is linear in PD, so the bias flows straight into the provision unless corrected. Class weights have the same *level* problem but it is a pure intercept shift that train-only recalibration removes; SMOTE additionally fabricates points, which can distort the slope as well as the level. Compare `over_prediction_ratio` below."""),
C('''from src.models.pd_models import smote_experiment
smote_experiment(champ._X(data.train), data.train.target, champ._X(data.valid), data.valid.target, cfg)'''),
M("""### Sensitivity: does dropping 2007-2011 (small volumes, different underwriting) change the story?"""),
C('''from src.pipeline import PDModel
from src.evaluation.metrics import performance_row
sub = data.train[data.train.vintage_year.between(*cfg["split"]["sensitivity_train_years"])]
s_m = PDModel(cfg, champ.numeric, champ.categorical, "logit").fit(sub)
pd.DataFrame([performance_row(yv, champ.predict(data.valid), "champion | train 2007-14"),
              performance_row(yv, s_m.predict(data.valid), f"champion | train {cfg['split']['sensitivity_train_years']}")])'''),
])

# ---------------------------------------------------------------- 05
nb("05_calibration_and_threshold.ipynb", [
M("""# 05 · Calibration and threshold selection
### Calibration design
- Class weights inflate predicted PD, and the 2015 base rate differs from the training base rate, so **level** matters as much as ranking.
- **Deployable recalibration** = Platt scaling fitted on **train out-of-fold** scores. Fitting it on in-sample scores would be optimistic (especially for GBM); fitting it on validation would be leakage.
- **ORACLE intercept shift** (uses validation outcomes) is shown only as a **diagnostic**: it tells us how much of any remaining gap is pure base-rate shift vs. a shape problem. It is not a deployable model.
- Per-decile **binomial tests** are reported; with large n they flag even small gaps, so read the *size* of the gap (rel_gap), not just the p-value.

### Threshold design
**Not F1.** F1 treats a false positive and a false negative as symmetric and ignores exposure size, LGD and risk appetite. We pick the cut-off from an **expected-loss vs. approval-rate trade-off**: approve the lowest-PD share of applicants; EL = PD x LGD_proxy x CCF_proxy x loan_amnt. Choose the **largest approval rate whose approved-book EL rate stays within an assumed appetite** (config input; not a bank policy)."""),
C('''from src.pipeline import prepare, get_models, calibration_tables
from src.evaluation.metrics import calibration_by_decile
data = prepare(cfg); models = get_models(data, cfg)
cal = calibration_tables(models, data)
cal["champion_logit"]["summary"]'''),
C('''c = cal["champion_logit"]
print("Decile calibration, train-only Platt (deployable):"); display(c["decile_cal"])
print("Decile calibration, raw class-weighted (before recalibration):"); display(c["decile_raw"])'''),
C('''fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, n in zip(axs, ["champion_logit","challenger_lgbm"]):
    for lab, key in [("raw","decile_raw"),("train-only Platt","decile_cal"),("ORACLE (diag.)","decile_oracle")]:
        t = cal[n][key]; ax.plot(t.expected_pd, t.observed_dr, "o-", label=lab)
    m = cal[n]["decile_raw"][["expected_pd","observed_dr"]].values.max(); ax.plot([0,m],[0,m],"k:"); ax.set(title=n, xlabel="expected PD", ylabel="observed default rate"); ax.legend()
plt.tight_layout(); plt.show()
cal["challenger_lgbm"]["summary"]'''),
M("**2016 (flagged, partially censored)** - reported separately; do not read its calibration as clean evidence."),
C('''y16 = data.flagged.target.astype(int); p16 = models["champion_logit"].predict(data.flagged)
from src.evaluation.metrics import calibration_summary
pd.Series(calibration_summary(y16, p16)).to_frame("champion on 2016 (censored)")'''),
M("""### Trade-off table
LGD and CCF come from **train defaults only** (notebook 06 details them) and are **proxies**."""),
C('''from src.models.el_proxy import fit_lgd_ead_proxy
from src.decision.threshold import tradeoff_table, recommend, ks_optimal_cutoff
el = fit_lgd_ead_proxy(data.train[data.train.target==1]); print(el)
v = data.valid.assign(pd_hat=models["champion_logit"].predict(data.valid))
trade = tradeoff_table(v, "pd_hat", el["lgd"], el["ccf"], cfg["decision"]["approval_grid"], cfg["decision"]["el_appetite"])
trade'''),
C('''rec = recommend(trade); ksc = ks_optimal_cutoff(v.target.astype(int), v.pd_hat)
print(f"Assumed EL appetite: {cfg['decision']['el_appetite']:.1%}")
print(f"Recommended: approve {rec.approval_rate:.0%} (PD cutoff {rec.pd_cutoff:.3f}); approved-book model EL rate {rec.el_rate_model:.2%}; defaults rejected {rec.defaults_rejected_share:.0%}; good loans rejected {rec.good_loans_rejected_share:.0%}")
print(f"KS-optimal PD cutoff (reference only, ignores exposure/LGD/appetite): {ksc:.3f}  -> approval rate at that cutoff: {(v.pd_hat<=ksc).mean():.0%}")
fig, ax = plt.subplots(figsize=(6.5,4.2))
ax.plot(trade.approval_rate*100, trade.el_rate_model*100, "o-", label="model EL rate"); ax.plot(trade.approval_rate*100, trade.loss_rate_realised_proxy*100, "s--", label="realised (proxy LGD/EAD)")
ax.axhline(cfg["decision"]["el_appetite"]*100, color="r", ls=":", label="assumed appetite"); ax.set(xlabel="approval rate (%)", ylabel="loss rate on approved exposure (%)"); ax.legend(); plt.show()
(ROOT/cfg["paths"]["reports_dir"]).mkdir(exist_ok=True, parents=True); trade.to_csv(ROOT/cfg["paths"]["reports_dir"]/"threshold_tradeoff.csv", index=False)'''),
M("""**Reading the table.** Each step down in approval rate rejects a lot of defaults but also many good loans; the cost of a rejected good loan (foregone margin) is **not** modelled here, which is why the recommendation is conditional on a stated appetite rather than presented as "optimal". Changing `decision.el_appetite` in `configs/config.yaml` moves the answer."""),
])

# ---------------------------------------------------------------- 06
nb("06_el_proxy_and_psi.ipynb", [
M("""# 06 · Illustrative EL decomposition and PSI
## Part A - EL = PD x LGD x EAD (**proxies, not production models**)
- **EAD proxy:** balance at default `= funded_amnt - total_rec_prncp` on charged-off loans; CCF_proxy = mean(EAD / funded_amnt). Term loans are fully drawn at origination, so this "CCF" is really amortisation.
- **LGD proxy:** exposure-weighted `1 - recoveries / EAD`, clipped to [0,1], on **train** defaults.
- These use `funded_amnt`, `recoveries`, `total_rec_prncp`, which are **banned from the PD feature set** (post-origination / outcome-linked). They are used only here.
- Limits: recoveries are undiscounted, no downturn LGD, no workout-cost data, no segmentation. A bank would model LGD with beta/fractional regression or a two-stage model and discount recoveries at the contract rate."""),
C('''from src.pipeline import prepare, get_models
from src.models.el_proxy import fit_lgd_ead_proxy
data = prepare(cfg); models = get_models(data, cfg); champ = models["champion_logit"]
td = data.train[data.train.target==1]
el = fit_lgd_ead_proxy(td); print(el)
d = td.assign(ead=(td.funded_amnt-td.total_rec_prncp).clip(lower=0))
d["lgd_i"] = (1 - d.recoveries/d.ead.where(d.ead>0)).clip(0,1)
print("LGD distribution on train defaults:"); print(d.lgd_i.describe().round(3).to_string())'''),
C('''v = data.valid.assign(pd_hat=champ.predict(data.valid))
v["EL"] = v.pd_hat * el["lgd"] * el["ccf"] * v.loan_amnt
bands = pd.qcut(v.pd_hat, 5, labels=[f"Q{i}" for i in range(1,6)])
g = v.groupby(bands, observed=True).agg(loans=("pd_hat","size"), mean_pd=("pd_hat","mean"), observed_dr=("target","mean"), exposure=("loan_amnt","sum"), EL=("EL","sum"))
g["EL_rate"] = g.EL/g.exposure; g'''),
M("""## Part B - PSI (distribution shift, **not** performance)
PSI compares the distribution of a score or feature in a *reference* sample (train) to another sample. It uses **no outcomes**, so:
- low PSI does **not** mean the model performs well; high PSI does **not** mean it performs badly;
- it is a *monitoring trigger* ("has the population moved?"), to be read next to the performance/calibration results, never instead of them.

Convention (not a theorem): <0.10 stable, 0.10-0.25 moderate shift, >0.25 major shift. 2017-2018 loans are included here (any status) because PSI needs features, not outcomes."""),
C('''from src.pipeline import psi_tables
pt = psi_tables(models, data)
pt["score"]'''),
C('''pt["features"].style.format("{:.3f}").background_gradient(cmap="Reds", vmin=0, vmax=0.25)'''),
C('''s = pt["score"].set_index("sample").score_psi
fig, ax = plt.subplots(figsize=(6,3.6)); ax.bar(s.index, s.values, color="#72b7b2"); ax.axhline(.1,color="orange",ls=":"); ax.axhline(.25,color="r",ls=":")
ax.set_title("Score PSI vs train (distribution shift only)"); plt.xticks(rotation=30); plt.show()'''),
])

# ---------------------------------------------------------------- 99
nb("99_end_to_end.ipynb", [
M("""# 99 · End-to-end orchestrator
Thin wrapper: raw CSV -> DuckDB -> clean -> bin -> train -> calibrate -> evaluate -> threshold -> reports. All logic lives in `src/`; this notebook only calls `run_all` and displays the headline outputs. It also writes `reports/model_card.md`, `reports/validation_memo.md`, `reports/threshold_tradeoff.csv` and the figures."""),
C('''from src.pipeline import run_all
res = run_all(cfg, rebuild=False, make_reports=True)'''),
M("## Results table (calibrated PD)"),
C('''p = res["perf"]; p[(p.pd_type=="calibrated") & (p["sample"]!="train (in-sample)")][["model","sample","n","default_rate","mean_pd","auc","gini","ks","brier"]]'''),
M("## Champion calibration (2015)"),
C('''res["cal"]["champion_logit"]["summary"]'''),
M("## Threshold recommendation (illustrative; assumed appetite)"),
C('''r = res["rec"]; print(f"approve {r.approval_rate:.0%} | PD cutoff {r.pd_cutoff:.3f} | approved-book EL rate {r.el_rate_model:.2%} | appetite {cfg['decision']['el_appetite']:.1%}")
res["trade"]'''),
M("## Score PSI (distribution shift, not performance)"),
C('''res["psi"]["score"]'''),
C('''print(sorted(p.name for p in (ROOT/cfg["paths"]["reports_dir"]).glob("*")), sorted(p.name for p in (ROOT/cfg["paths"]["figures_dir"]).glob("*.png")))'''),
])
print("built", sorted(p.name for p in OUT.glob("*.ipynb")))
