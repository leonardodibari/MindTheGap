"""Dataset loading and artifact-alignment checks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, ID_COLUMN, NOTEBOOK_ORDER, RAW_INPUT_COLUMN, TARGET_COLUMN


def require_artifact(path: Path, generated_by: str) -> Path:
    """Return an artifact path or raise an actionable workflow error."""
    if path.exists():
        return path
    committed_review_input = "outputs/artifacts" in path.as_posix()
    expectation = (
        "This compact review artifact should normally be present in a fresh clone. "
        "Restore it from Git; for full reproduction, regenerate it by running"
        if committed_review_input else
        "This generated workflow artifact is required. Create it by running"
    )
    raise FileNotFoundError(
        f"Required artifact not found:\n{path}\n\n"
        f"Why it is required: notebooks 04/05 consume saved analysis inputs without training.\n"
        f"{expectation}:\n{generated_by}\n\n"
        f"Recommended full-reproduction order:\n{NOTEBOOK_ORDER}"
    )


def load_required_csv(path: Path, generated_by: str, **kwargs) -> pd.DataFrame:
    """Load a required CSV with an actionable missing-artifact message."""
    return pd.read_csv(require_artifact(path, generated_by), **kwargs)


def load_required_numpy(path: Path, generated_by: str, **kwargs):
    """Load a required NumPy artifact with an actionable missing-artifact message."""
    return np.load(require_artifact(path, generated_by), **kwargs)


def ordered_id_hash(ids: Iterable[object]) -> str:
    """Return a stable, boundary-safe SHA-256 hash for ordered molecule IDs."""
    digest = hashlib.sha256()
    for value in ids:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_analysis_features(artifact, labeled: pd.DataFrame) -> None:
    """Validate the committed classical-analysis artifact against the supplied dataset."""
    required = {"X_size", "X_geometry", "molecular_weight", "scaffolds", "row_count", "ordered_mol_id_sha256"}
    missing = required.difference(artifact.files)
    if missing:
        raise ValueError(f"Analysis feature artifact is missing keys: {sorted(missing)}")
    n_rows = len(labeled)
    stored_rows = int(np.asarray(artifact["row_count"]).item())
    stored_hash = str(np.asarray(artifact["ordered_mol_id_sha256"]).item())
    observed_hash = ordered_id_hash(labeled[ID_COLUMN])
    if stored_rows != n_rows or stored_hash != observed_hash:
        raise ValueError(
            "The committed analysis artifact is incompatible with datasets/base.csv: "
            f"expected {stored_rows} labeled rows and ordered-ID hash {stored_hash}, "
            f"found {n_rows} rows and {observed_hash}. Use the original assignment dataset "
            "or regenerate the artifact with notebooks/02_classical_baselines.ipynb."
        )
    expected_shapes = {
        "X_size": (n_rows, 13), "X_geometry": (n_rows, 7),
        "molecular_weight": (n_rows,), "scaffolds": (n_rows,),
    }
    for key, shape in expected_shapes.items():
        if artifact[key].shape != shape:
            raise ValueError(f"Analysis feature {key!r} has shape {artifact[key].shape}; expected {shape}")
    for key in ("X_size", "X_geometry", "molecular_weight"):
        if not np.isfinite(artifact[key]).all():
            raise ValueError(f"Analysis feature {key!r} contains NaN or infinite values")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the molecular dataset and normalize its identifier column."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required assignment dataset not found: {path}\n"
            "Copy the original supplied file to datasets/base.csv before running the notebooks."
        )
    frame = pd.read_csv(path)
    if ID_COLUMN not in frame and str(frame.columns[0]).startswith("Unnamed"):
        frame = frame.rename(columns={frame.columns[0]: ID_COLUMN})
    required = {ID_COLUMN, RAW_INPUT_COLUMN, TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def labeled_molecules(frame: pd.DataFrame) -> pd.DataFrame:
    """Return labeled IDs, conformers and targets without changing row order."""
    return frame.loc[
        frame[TARGET_COLUMN].notna(), [ID_COLUMN, RAW_INPUT_COLUMN, TARGET_COLUMN]
    ].reset_index(drop=True)


def finite_complete_rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Select rows complete and finite for the requested columns."""
    return frame.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna()


def validate_split_indices(
    train: Sequence[int], validation: Sequence[int], test: Sequence[int], n_rows: int
) -> None:
    """Validate uniqueness, range, and disjointness of three index splits."""
    arrays = [np.asarray(values, dtype=int) for values in (train, validation, test)]
    names = ("train", "validation", "test")
    for name, values in zip(names, arrays):
        if len(values) != len(np.unique(values)):
            raise ValueError(f"{name} split contains duplicate indices")
        if len(values) and (values.min() < 0 or values.max() >= n_rows):
            raise ValueError(f"{name} split contains an out-of-range index")
    for left in range(3):
        for right in range(left + 1, 3):
            overlap = np.intersect1d(arrays[left], arrays[right])
            if len(overlap):
                raise ValueError(f"{names[left]} and {names[right]} overlap")


def validate_prediction_alignment(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    id_column: str = ID_COLUMN,
    target_column: str = "target",
) -> None:
    """Fail when prediction IDs are duplicated, missing, reordered, or mis-targeted."""
    if predictions[id_column].duplicated().any():
        raise ValueError("Prediction IDs are not unique")
    expected = reference.set_index(id_column)
    observed_ids = predictions[id_column]
    missing = observed_ids[~observed_ids.isin(expected.index)]
    if len(missing):
        raise ValueError(f"Unknown prediction IDs: {missing.head().tolist()}")
    aligned = expected.loc[observed_ids]
    if target_column in predictions and TARGET_COLUMN in aligned:
        np.testing.assert_allclose(
            predictions[target_column].to_numpy(float),
            aligned[TARGET_COLUMN].to_numpy(float),
            rtol=0,
            atol=0,
        )
    prediction_columns = [column for column in predictions if "prediction" in column]
    if prediction_columns:
        values = predictions[prediction_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Prediction artifact contains NaN or infinite predictions")


def validate_prediction_split(
    predictions: pd.DataFrame,
    expected_indices,
    *,
    split: str | None = None,
) -> None:
    """Validate row count, index membership, order, and optional split label."""
    expected = np.asarray(expected_indices, dtype=int)
    observed = predictions["dataset_index"].to_numpy(dtype=int)
    if not np.array_equal(observed, expected):
        raise ValueError("Prediction dataset indices do not match the expected split order")
    if split is not None and (
        "split" not in predictions or not predictions["split"].eq(split).all()
    ):
        raise ValueError(f"Prediction artifact is not consistently labeled as {split!r}")


def subset_by_ids(
    frame: pd.DataFrame, ids: Iterable[object], id_column: str = ID_COLUMN
) -> pd.DataFrame:
    """Return rows in requested ID order; reject missing or duplicate IDs."""
    if frame[id_column].duplicated().any():
        raise ValueError(f"{id_column} is not unique in source frame")
    requested = pd.Index(ids)
    indexed = frame.set_index(id_column)
    missing = requested.difference(indexed.index)
    if len(missing):
        raise KeyError(f"Missing IDs: {missing[:5].tolist()}")
    return indexed.loc[requested].reset_index()
