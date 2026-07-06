"""Path helpers so every notebook resolves data the same way,
regardless of the directory it was launched from.

Usage from a notebook (after adding the repo root to sys.path):
    from src.paths import RAW_DIR, PROCESSED_DIR, repo_root
    df = pd.read_csv(RAW_DIR / "dropout.csv", sep=";")
"""

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Walk upward from `start` until the folder containing data/raw is found.

    Anchoring to a known marker (data/raw) rather than the current working
    directory means paths resolve correctly whether a notebook runs from
    notebooks/ or from the repo root.
    """
    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if (p / "data" / "raw").is_dir():
            return p
    raise FileNotFoundError(
        "Repo root not found (no data/raw directory above the current path). "
        "If the dataset is missing, run:  bash data/download_data.sh"
    )


ROOT = repo_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
