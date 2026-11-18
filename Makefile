PY ?= python
.PHONY: lock install data sql train evaluate report test smoke notebooks clean

lock:           ## regenerate requirements.lock from the current env
	$(PY) scripts/make_lock.py

install:
	$(PY) -m pip install -r requirements.txt   # exact resolved set: pip install -r requirements.lock

data:            ## download raw data (needs Kaggle credentials)
	$(PY) scripts/download_data.py

sql:             ## build DuckDB model_frame from the raw CSV
	$(PY) -c "from src.utils.config import load_config; from src.data.loader import build_database; print(build_database(load_config(), rebuild=True))"

train:           ## train all models (cached on config hash)
	$(PY) -c "from src.utils.config import load_config; from src.pipeline import prepare, get_models; c=load_config(); get_models(prepare(c), c, refit=True)"

evaluate:        ## full run incl. metrics, calibration, threshold, PSI
	$(PY) -m src.pipeline

report:          ## model_card.md, validation_memo.md, threshold_tradeoff.csv, figures (+ PDF if pandoc exists)
	$(PY) -m src.pipeline
	@command -v pandoc >/dev/null && pandoc reports/validation_memo.md --resource-path=reports:figures -o reports/analytical_report.pdf || echo "pandoc not found: analytical_report.pdf skipped (validation_memo.md is the report)"

test:
	$(PY) -m pytest -q tests

smoke:           ## pipeline on SYNTHETIC data to verify the plumbing (numbers are meaningless)
	$(PY) scripts/make_synthetic.py --n 150000
	CREDIT_PD_CONFIG=configs/config_synthetic.yaml $(PY) -m src.pipeline

notebooks:       ## regenerate notebooks/*.ipynb from scripts/build_notebooks.py
	$(PY) scripts/build_notebooks.py

clean:
	rm -rf data/processed/* .pytest_cache
