
# Consumer Lending Risk: Predictive Modelling & Decision Analysis

## Scope

This project develops and evaluates a **probability-of-default (PD) model** using the public Lending Club accepted-loan dataset (2007–2018 Q4). It covers out-of-time validation, probability calibration and approval-threshold analysis.

LGD and EAD are included as illustrative proxies to demonstrate expected-loss decomposition. The analysis is based on Lending Club's accepted loans and is not intended to represent a bank-level credit risk model.

## Project status

The complete pipeline has been executed on the real Lending Club dataset.

- **Raw records parsed:** 2,260,701
- **36-month loans in model_frame:** 1,609,754
- **Clean validation sample:** 283,026 loans from 2015
- **Maturity audit:** Passed for all 12 months of 2015, with a minimum resolved share of 0.9968 against a 0.95 threshold.

The results are available below and in `reports/validation_memo.md`.

The Kaggle dataset slug configured in `scripts/download_data.py` follows the project setup. The download script checks for the expected file and reports an error if it cannot be located.

### Engineering improvements

During development and code review, I identified and fixed several issues involving loan maturity checks, category pooling, data-loading integrity, PSI calculation and missing-value handling.

These changes are documented in `docs/decision_log.md` and covered by regression tests in `tests/test_review_fixes.py`.

## Modelling approach

| Element | Choice | Reason |
|---|---|---|
| Population | 36-month loans | Maintains a consistent maturity window |
| Target | Charged Off/Default = 1, Fully Paid = 0 | Excludes loans with unresolved outcomes |
| Training period | Issue years 2007–2014 | Historical training sample |
| Sensitivity analysis | Issue years 2012–2014 | Checks sensitivity to the training period |
| Clean validation | 2015 | Fully matured out-of-time validation vintage |
| Flagged validation | 2016 | Reported separately because of partial censoring |
| Distribution monitoring | 2017–2018 | Used for PSI analysis, not model performance |
| Champion model | Logistic Regression with monotonic WoE bins | Interpretability and exclusion of Lending Club's own risk ratings |
| Challenger model | LightGBM with monotonic constraints | Tests whether a more flexible model improves predictive performance |
| Imbalance handling | Class weights | SMOTE was also tested as a comparison |
| Calibration | Platt scaling using out-of-fold training predictions | Avoids fitting calibration on the validation sample |
| Threshold selection | Expected-loss versus approval-rate analysis | Connects model predictions to lending decisions |
| Stability monitoring | Population Stability Index (PSI) | Monitors changes in feature distributions |

The challenger has a predefined promotion criterion: a Gini improvement of at least 0.02, provided calibration is not worse.

Each major modelling decision is explained in [`docs/decision_log.md`](docs/decision_log.md) and the notebook documentation.

## Results

The following results were obtained from the real Lending Club dataset. The clean validation sample is the 2015 out-of-time vintage.

| Model | Sample | Observations | Default rate | AUC | Gini | KS | Brier | Mean PD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| champion_logit | Valid 2015 | 283,026 | 0.1489 | 0.6363 | 0.2725 | 0.1980 | 0.1232 | 0.1364 |
| challenger_lgbm | Valid 2015 | 283,026 | 0.1489 | 0.6492 | 0.2983 | 0.2155 | 0.1225 | 0.1368 |
| grade_only_logit (benchmark) | Valid 2015 | 283,026 | 0.1489 | 0.6675 | 0.3349 | 0.2616 | 0.1215 | 0.1280 |
| with_lc_logit (incremental lift) | Valid 2015 | 283,026 | 0.1489 | 0.6834 | 0.3669 | 0.2670 | 0.1208 | 0.1234 |
| with_lc_lgbm (ceiling) | Valid 2015 | 283,026 | 0.1489 | 0.6871 | 0.3741 | 0.2725 | 0.1204 | 0.1239 |
| champion_logit | 2016 (censored) | 232,361 | 0.1985 | 0.6450 | 0.2899 | 0.2054 | 0.1576 | 0.1321 |

### Key finding

Lending Club's own grade achieved a Gini of 0.3349, compared with 0.2725 for my borrower-characteristics-only model.

This comparison illustrates the predictive information contained in an existing underwriting assessment and highlights why including a lender's own risk rating can change what a model is actually measuring.

The challenger LightGBM model achieved a Gini of 0.2983, compared with 0.2725 for the logistic regression champion. Its Brier score was also marginally lower (0.1225 versus 0.1232).

Calibration by decile, SMOTE comparison, PSI tables and approval-rate trade-off analysis are available in:

- `reports/validation_memo.md`
- `reports/threshold_tradeoff.csv`

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/shashikanthkumar/credit-pd-el.git
cd credit-pd-el
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
make install
```

Alternatively, install the exact resolved dependencies:

```bash
pip install -r requirements.lock
```

### 4. Download the dataset

Configure Kaggle credentials in `~/.kaggle/kaggle.json`, then run:

```bash
make data
```

The expected dataset file is:

`data/raw/accepted_2007_to_2018Q4.csv`

### 5. Run tests and generate reports

```bash
make test
make report
```

The pipeline generates reports and figures in their respective directories.

### 6. Launch Jupyter

```bash
jupyter lab
```

Run notebooks `01` through `06` in sequence, or use notebook `99` to execute the complete workflow.

### 7. Run a synthetic-data check

```bash
make smoke
```

This checks pipeline functionality using synthetic data. Its results should not be interpreted as model performance.

## Repository structure

```text
configs/
    config.yaml              Model settings, split years and features

data/
    raw/                     Downloaded source data

sql/
    01_schema
    02_load_raw              Raw data ingestion using DuckDB
    03_model_frame           Data typing, target and maturity flags

src/
    data/                    Data loading and cleaning
    features/                WoE/IV binning and leakage checks
    models/                  Logistic Regression, LightGBM and calibration
    evaluation/              Performance metrics, PSI and plots
    decision/                Expected-loss and approval-rate analysis
    pipeline.py              End-to-end pipeline
    reporting.py             Report generation

notebooks/
    01–06                     Step-by-step analysis
    99                        Complete workflow

tests/
    WoE/IV, metrics, leakage,
    split boundaries and regression tests

reports/
    model_card.md
    validation_memo.md
    threshold_tradeoff.csv

figures/
    Generated evaluation figures

docs/
    decision_log.md
```

## Limitations and considerations

- **Accepted loans only:** The dataset contains approved loans. Reject inference is not performed, so the model does not describe all loan applicants.
- **Single snapshot:** The analysis uses a single dataset snapshot rather than a complete performance panel. The 2015 validation vintage represents one macroeconomic period.
- **Right-censoring:** The 36-month loan filter and maturity audit reduce the impact of incomplete outcomes. The 2016 sample remains flagged for partial censoring.
- **Lending Club risk variables:** Grade and interest rate reflect Lending Club's own underwriting process. They are excluded from the champion model and included separately for comparison.
- **LGD and EAD:** These are illustrative proxies based on available recovery information. They do not represent production-grade estimates, downturn LGD or portfolio segmentation.
- **Expected loss:** The approval appetite is an assumed modelling input, not an actual bank policy. The cost of rejected good loans is not included.
- **Calibration:** Recalibration can adjust predicted probabilities to a reference base rate, but it cannot anticipate future changes in default behaviour.
- **PSI:** Measures changes in feature distributions, not predictive performance. PSI and IV thresholds are conventions rather than statistical tests.
- **Regulatory scope:** The project does not implement Basel or IRB requirements such as regulatory PD floors, margin of conservatism, downturn calibration or regulatory capital calculations.

## Technologies

- **Programming:** Python
- **Data processing:** SQL, DuckDB, Pandas
- **Modelling:** Scikit-learn, LightGBM
- **Evaluation:** AUC, Gini, KS, Brier score, calibration and PSI
- **Development:** Jupyter Notebook, Git, automated testing

## Author

**Shashikanth Kumar Sole**

MSc Data Science and Business Analytics  
University of Warsaw

[GitHub](https://github.com/shashikanthkumar)
