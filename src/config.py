"""Shared repository constants and paths."""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root when called from the root or ``notebooks/``."""
    candidate = (start or Path.cwd()).resolve()
    for path in (candidate, *candidate.parents):
        if (path / "environment.yml").exists() and (path / "notebooks").is_dir():
            return path
    raise FileNotFoundError("Could not locate the MindTheGap project root.")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "datasets"
DATA_PATH = DATA_DIR / "base.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
REVIEW_ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
CLASSICAL_ANALYSIS_FEATURES = REVIEW_ARTIFACT_DIR / "classical_analysis_features.npz"
CLASSICAL_ANALYSIS_MORGAN = REVIEW_ARTIFACT_DIR / "classical_analysis_morgan.npz"
TABLE_DIR = OUTPUT_DIR / "tables"
EDA_FIGURE_DIR = OUTPUT_DIR / "figures" / "eda"
CLASSICAL_FIGURE_DIR = RESULTS_DIR / "figures"
FINAL_FIGURE_DIR = RESULTS_DIR / "final_figures"
ADDITIONAL_DIR = OUTPUT_DIR / "additional_analysis"
MODEL_DIR = RESULTS_DIR / "models"
SPLIT_ARTIFACT = RESULTS_DIR / "split_indices.npz"
CLASSICAL_RANDOM_PREDICTIONS = RESULTS_DIR / "classical_predictions_random.csv"
CLASSICAL_SCAFFOLD_PREDICTIONS = RESULTS_DIR / "classical_predictions_scaffold.csv"
SCHNET_RANDOM_PREDICTIONS = RESULTS_DIR / "schnet_predictions_random.csv"
SCHNET_SCAFFOLD_PREDICTIONS = RESULTS_DIR / "schnet_predictions_scaffold.csv"
CLASSICAL_METRICS = RESULTS_DIR / "classical_baselines.csv"
SCHNET_METRICS = RESULTS_DIR / "schnet_metrics.csv"
FINAL_RANDOM_PREDICTIONS = RESULTS_DIR / "final_predictions_random_test.csv"
FINAL_SCAFFOLD_PREDICTIONS = RESULTS_DIR / "final_predictions_scaffold_test.csv"
FINAL_VALIDATION_PREDICTIONS = RESULTS_DIR / "final_predictions_validation.csv"

ID_COLUMN = "mol_id"
RAW_INPUT_COLUMN = "json_conformer"
TARGET_COLUMN = "gap"
RANDOM_SEED = 42
SPLIT_NAMES = ("train", "validation", "test")
NOTEBOOK_ORDER = "01 → 02 → 03 → 04 → 05"
