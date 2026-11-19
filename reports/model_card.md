# Model card: PD model, Lending Club 36-month loans

_Generated 2026-11-19 by `src/reporting.py` from the run's actual outputs. Scope: **PD model, not a Basel/IRB framework. LGD/EAD are illustrative proxies.**_

## Intended use
Rank-order and estimate 36-month default probability for accepted consumer loans on a public dataset, to demonstrate PD development, out-of-time validation, calibration and an EL-vs-approval-rate decision. **Not** for production lending or regulatory capital.

## Data and population
36-month loans only (uniform maturity window). Target: Charged Off/Default = 1, Fully Paid = 0; Current/Late/In Grace excluded.
Train 2007-2014 (n=335,558); clean validation 2015 (n=283,026); flagged-censored sample = 2016 plus any demoted 2015 months (n=232,361); 2017-2018 used for PSI only. All issue months of the clean year passed the maturity/resolution audit.

## Models
- **Champion**: logistic regression on monotone WoE bins, features: fico_mid, annual_inc, dti, loan_to_income, home_ownership, revol_util; class_weight=`balanced` then train-only out-of-fold Platt recalibration.
- **Challenger**: LightGBM with monotone constraints, same recalibration.
- **Benchmarks**: LC grade only; core + LC int_rate/grade ("with_lc") to measure incremental lift over LC's own pricing.

## Performance (calibrated PD, out-of-time)
| model | sample | n | default_rate | mean_pd | auc | gini | ks | brier |
|---|---|---|---|---|---|---|---|---|
| grade_only_logit | valid_clean | 283026 | 0.1489 | 0.1280 | 0.6675 | 0.3349 | 0.2616 | 0.1215 |
| grade_only_logit | flagged_censored | 232361 | 0.1985 | 0.1323 | 0.6656 | 0.3312 | 0.2559 | 0.1558 |
| champion_logit | valid_clean | 283026 | 0.1489 | 0.1364 | 0.6363 | 0.2725 | 0.1980 | 0.1232 |
| champion_logit | flagged_censored | 232361 | 0.1985 | 0.1321 | 0.6450 | 0.2899 | 0.2054 | 0.1576 |
| with_lc_logit | valid_clean | 283026 | 0.1489 | 0.1234 | 0.6834 | 0.3669 | 0.2670 | 0.1208 |
| with_lc_logit | flagged_censored | 232361 | 0.1985 | 0.1275 | 0.6874 | 0.3747 | 0.2718 | 0.1543 |
| challenger_lgbm | valid_clean | 283026 | 0.1489 | 0.1368 | 0.6492 | 0.2983 | 0.2155 | 0.1225 |
| challenger_lgbm | flagged_censored | 232361 | 0.1985 | 0.1345 | 0.6578 | 0.3156 | 0.2264 | 0.1560 |
| with_lc_lgbm | valid_clean | 283026 | 0.1489 | 0.1239 | 0.6871 | 0.3741 | 0.2725 | 0.1204 |
| with_lc_lgbm | flagged_censored | 232361 | 0.1985 | 0.1283 | 0.6929 | 0.3859 | 0.2787 | 0.1533 |

Calibration summary, champion, 2015:

| variant | mean_pred | observed | cal_ratio | brier |
|---|---|---|---|---|
| raw (class-weighted) | 0.4911 | 0.1489 | 0.3031 | 0.2441 |
| train-only Platt | 0.1364 | 0.1489 | 1.0910 | 0.1232 |
| ORACLE intercept shift (diagnostic) | 0.1489 | 0.1489 | 1.0000 | 0.1230 |

## Threshold (illustrative)
Assumed EL appetite 6.0% (an input, not a bank policy). Largest approval rate within appetite: **80%** (PD cutoff 0.183, model EL rate 5.61%).
Proxies: LGD=0.87, CCF=0.58. KS-optimal PD cutoff for reference: 0.136.

## Limitations
See `validation_memo.md` section 6 and the README.
