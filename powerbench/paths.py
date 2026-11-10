from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Walk upward until pyproject.toml + powerbench/ are found."""
    here = (start or Path(__file__)).resolve()
    for cand in [here, *here.parents]:
        if (cand / "pyproject.toml").is_file() and (cand / "powerbench").is_dir():
            return cand
    # Fallback: package lives at <root>/powerbench/paths.py
    return Path(__file__).resolve().parents[1]


def data_dir(start: Path | None = None) -> Path:
    return repo_root(start) / "data"


def taxonomy_dir(start: Path | None = None) -> Path:
    return data_dir(start) / "taxonomy"


def demo_models_dir(start: Path | None = None) -> Path:
    """Demo / working device-type directories: data/demo/models/."""
    return data_dir(start) / "demo" / "models"


def demo_raw_dir(start: Path | None = None) -> Path:
    """Frozen corpus export for tasks: data/demo/raw/."""
    return data_dir(start) / "demo" / "raw"


# Backward-compatible aliases for the previous types/corpus layout names.
demo_types_dir = demo_models_dir
demo_corpus_dir = demo_raw_dir


def configs_dir(start: Path | None = None) -> Path:
    return repo_root(start) / "configs"


DEFAULT_DEVICE_MODELS_XLSX = "device_models.xlsx"
DEFAULT_DEMO_TYPE = "0002_pm800kV_high-end_converter_transformer"
