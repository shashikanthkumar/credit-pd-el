# credit-pd-el: probability-of-default model with out-of-time validation

## Scope statement
Develop and evaluate a **probability-of-default (PD) model** on a public consumer-lending dataset (Lending Club accepted loans, 2007-2018Q4) with out-of-time validation, calibration and threshold selection. LGD and EAD are used **only as illustrative proxies** for an expected-loss decomposition.

> **This is a PD model, not a Basel/IRB-compliant credit risk framework.** LGD and EAD are proxy extensions, not production models. Results describe a public dataset of *accepted* loans, not a bank book.

## Status: read this first
- The full pipeline has run on the real Lending Club file: **2,260,701 raw rows parsed; 1,609,754 36-month loans** in `model_frame`. Results are in the table below and in `reports/validation_memo.md`.
- The maturity audit passed for **all 12 months of 2015** (resolved share >= 0.9968; threshold 0.95).
- The Kaggle dataset slug in `scripts/download_data.py` is as supplied in the brief and was not verified; the script fails loudly if it cannot find the file.


Engineering improvements

During development and code review, I identified and fixed several issues involving loan maturity checks, category pooling, data-loading integrity, PSI calculation and missing-value handling. I documented these changes in the decision log and added regression tests to prevent the issues from recurring.

## Design in one table
| Element | Choice | Why |
|---|---|---|
| Population | 36-month loans | uniform maturity window; snapshot is right-censored |
| Target | Charged Off/Default=1, Fully Paid=0; others excluded | unknown outcomes are not assumed |
| Train | issue year 2007-2014 (sensitivity: 2012-14) | |
| **Clean validation** | **2015** (only fully matured OOT vintage), month-by-month audited | headline performance + calibration |
| Flagged validation | 2016, labelled *partially censored* | reported separately |
| Shift only | 2017-2018 | PSI only, never performance |
| Champion | logistic regression on monotone WoE bins, **no LC grade/int_rate** | interpretable; avoids circularity |
| Challenger | LightGBM, monotone constraints | promoted only if Gini uplift >= 0.02 and calibration not worse |
| Imbalance | class weights; SMOTE tested once as negative result | SMOTE inflates PD level -> provisioning bias |
| Calibration | Platt on train out-of-fold scores; oracle intercept shift as diagnostic only | no validation leakage |
| Threshold | expected-loss vs approval-rate trade-off under an *assumed* appetite | not F1 |
| PSI | labelled a distribution-shift metric | uses no outcomes |

Every choice has a written justification in [`docs/decision_log.md`](docs/decision_log.md) and in the notebook markdown.

## Results (real Lending Club data, run 2026-11-19)

| Model | Sample | n | Default rate | AUC | Gini | KS | Brier | Mean PD |
|---|---|---|---|---|---|---|---|---|
| champion_logit | valid 2015 | 283,026 | 0.1489 | 0.6363 | 0.2725 | 0.1980 | 0.1232 | 0.1364 |
| challenger_lgbm | valid 2015 | 283,026 | 0.1489 | 0.6492 | 0.2983 | 0.2155 | 0.1225 | 0.1368 |
| grade_only_logit (benchmark) | valid 2015 | 283,026 | 0.1489 | 0.6675 | 0.3349 | 0.2616 | 0.1215 | 0.1280 |
| with_lc_logit (incremental lift) | valid 2015 | 283,026 | 0.1489 | 0.6834 | 0.3669 | 0.2670 | 0.1208 | 0.1234 |
| with_lc_lgbm (ceiling) | valid 2015 | 283,026 | 0.1489 | 0.6871 | 0.3741 | 0.2725 | 0.1204 | 0.1239 |
| champion_logit | 2016 (censored) | 232,361 | 0.1985 | 0.6450 | 0.2899 | 0.2054 | 0.1576 | 0.1321 |

**Key finding.** 
Lending Club's own grade achieved a Gini of 0.3349, compared with 0.2725 for my borrower-characteristics-only model. This comparison illustrates how much predictive information may already be captured by an existing underwriting system. It also helped me understand why including a lender's own risk rating in a model can change what the model is actually measuring.
Calibration by decile, SMOTE comparison, PSI tables and the trade-off table are generated into `reports/validation_memo.md` and `reports/threshold_tradeoff.csv`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
make install       # or: pip install -r requirements.lock for the exact resolved set
# put Kaggle credentials in ~/.kaggle/kaggle.json, then:
make data        # downloads data/raw/accepted_2007_to_2018Q4.csv
make test
make report      # runs everything; writes reports/ and figures/
jupyter lab      # then run notebooks 01 -> 06, or 99 for the one-shot version
make smoke       # synthetic-data plumbing check (numbers meaningless)
```

## Layout
```
configs/config.yaml        seeds, split years, term filter, features, appetite (single source of truth)
sql/                       01 schema, 02 load raw (DuckDB), 03 model_frame (typing, target, vintage, maturity flags)
src/data                   DuckDB loader, cleaning + documented missing-value rules
src/features               WoE/IV binner, leakage guards
src/models                 PD logit + LightGBM, calibration, LGD/EAD proxies
src/evaluation             AUC/Gini/KS/Brier, decile calibration, PSI, plots
src/decision               EL-vs-approval-rate threshold table
src/pipeline.py            end-to-end run; src/reporting.py writes reports from actual results
notebooks/                 01 data+SQL, 02 EDA+vintage, 03 WoE/IV, 04 champion/challenger,
                           05 calibration+threshold, 06 EL proxy+PSI, 99 orchestrator
tests/                     binning monotonicity, WoE/IV identities, metrics vs known cases, leakage guard, split bounds
reports/                   model_card.md, validation_memo.md, threshold_tradeoff.csv (generated)
```

## Limitations (do not overclaim)
- **Accepted loans only**: no reject inference; the model describes approved applicants.
- **Single snapshot, not a performance panel**; 2015 is a single clean validation vintage (one macro regime).
- **Right-censoring** handled by design (36m, 2015-only headline), not eliminated; 2016 is shown but flagged.
- LC `grade`/`int_rate` are LC's own risk output; excluded from the champion, shown as an incremental-lift variant.
- **LGD/EAD are proxies**: undiscounted recoveries, no downturn LGD, no segmentation. EL appetite is an assumed input, not bank policy. Cost of rejected good loans (foregone margin) is not modelled.
- Recalibration restores the training base rate; it cannot anticipate a shifted one. Monitor observed/expected.
- PSI is distribution shift, **not** performance. IV and PSI thresholds are conventions, not statistical tests.
- No Basel/IRB elements: no PD floors, margin of conservatism, downturn calibration or regulatory capital.
