import os
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    """Load config. Override with CREDIT_PD_CONFIG env var (used for the synthetic smoke test)."""
    path = path or os.environ.get("CREDIT_PD_CONFIG") or ROOT / "configs" / "config.yaml"
    path = Path(path)
    path = path if path.is_absolute() else (ROOT / path if (ROOT / path).exists() else path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def abs_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
