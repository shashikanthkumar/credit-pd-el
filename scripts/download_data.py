"""Download Lending Club accepted loans via the Kaggle API (no raw data is committed).

Prereqs: `pip install kaggle`, and ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY).
Dataset slug is as supplied in the brief; it is verified at run time, not assumed.
"""
import glob, gzip, os, shutil, sys, zipfile
from pathlib import Path

SLUG = "wordsforthewise/lending-club"
RAW = Path("data/raw")
TARGET = RAW / "accepted_2007_to_2018Q4.csv"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"{TARGET} already exists; nothing to do.")
        return
    try:
        import kaggle  # noqa: F401  (authenticates on import)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.dataset_download_files(SLUG, path=str(RAW), unzip=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Kaggle download failed ({exc}).\nDownload {SLUG} manually, put the accepted-loans CSV at {TARGET}.")
    # The dataset ships accepted_2007_to_2018Q4.csv.gz (and sometimes a nested folder)
    for gz in glob.glob(str(RAW / "**" / "accepted_2007_to_2018Q4.csv.gz"), recursive=True):
        with gzip.open(gz, "rb") as f_in, open(TARGET, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    if not TARGET.exists():
        found = glob.glob(str(RAW / "**" / "accepted*.csv"), recursive=True)
        if found:
            shutil.copy(found[0], TARGET)
    if not TARGET.exists():
        sys.exit(f"Could not locate accepted-loans file under {RAW}. Place it at {TARGET}.")
    print("Ready:", TARGET)


if __name__ == "__main__":
    main()
