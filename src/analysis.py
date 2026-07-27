"""Project-specific exploratory and post-hoc analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem

from .loaders import mol_from_json, remove_hydrogens


def missing_patterns(frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Count the most frequent row-wise missingness patterns."""
    patterns = frame.isna().apply(
        lambda row: ", ".join(frame.columns[row.to_numpy()]) or "none", axis=1
    )
    return patterns.value_counts().head(top_n).rename_axis("missing_columns").to_frame("rows")


def empirical_ks(x, y, max_n: int = 20_000, seed: int = 42, rng=None) -> float:
    """Compute the empirical KS distance with deterministic large-sample subsampling."""
    rng = rng or np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    if len(x) > max_n:
        x = rng.choice(x, max_n, replace=False)
    if len(y) > max_n:
        y = rng.choice(y, max_n, replace=False)
    grid = np.sort(np.concatenate([x, y]))
    return float(np.max(np.abs(
        np.searchsorted(np.sort(x), grid, side="right") / len(x)
        - np.searchsorted(np.sort(y), grid, side="right") / len(y)
    )))


def residualize(values, control):
    """Remove the least-squares linear association with a control variable."""
    design = np.column_stack([np.ones(len(control)), control])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    """Return Spearman correlation, or NaN if either variable is constant."""
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return np.nan
    return float(left.corr(right, method="spearman"))


def error_associations(frame: pd.DataFrame, geometry_names: list[str]) -> pd.DataFrame:
    """Associate model errors and SchNet improvement with molecular properties."""
    numeric = ["target", "n_atoms", "n_heavy_atoms", "scaffold_train_frequency", *geometry_names]
    return pd.DataFrame({
        "lightgbm_error_spearman": [safe_spearman(frame[c], frame["lightgbm_absolute_error"]) for c in numeric],
        "schnet_error_spearman": [safe_spearman(frame[c], frame["schnet_absolute_error"]) for c in numeric],
        "schnet_improvement_spearman": [safe_spearman(frame[c], frame["schnet_improvement"]) for c in numeric],
    }, index=numeric)


def difficult_tables(frame: pd.DataFrame, n: int = 12) -> dict[str, pd.DataFrame]:
    """Select difficult and strongly model-favoring molecules without interpretation."""
    scored = frame.copy()
    scored["both_models_error"] = scored[
        ["lightgbm_absolute_error", "schnet_absolute_error"]
    ].min(axis=1)
    columns = ["mol_id", "target", "lightgbm_prediction", "schnet_prediction",
        "lightgbm_absolute_error", "schnet_absolute_error", "scaffold_train_frequency",
        "n_atoms", "n_heavy_atoms", "smiles"]
    return {
        "Both models predict poorly": scored.nlargest(n, "both_models_error")[columns],
        "Strongly improved by SchNet": scored.nlargest(n, "schnet_improvement")[columns],
        "Strongly improved by LightGBM": scored.nsmallest(n, "schnet_improvement")[columns],
    }


def build_prediction_frame(
    *,
    split: str,
    train_indices,
    test_indices,
    targets,
    lightgbm_predictions,
    schnet_predictions,
    molecule_table: pd.DataFrame,
    scaffolds,
    size_matrix,
    size_names,
    geometry_matrix,
    geometry_names,
) -> pd.DataFrame:
    """Build the aligned wide prediction artifact used by notebooks 04 and 05."""
    train_frequency = pd.Series(scaffolds[train_indices]).value_counts()
    frame = pd.DataFrame({
        "dataset_index": test_indices,
        "mol_id": molecule_table.iloc[test_indices]["mol_id"].to_numpy(),
        "target": targets,
        "lightgbm_prediction": lightgbm_predictions,
        "schnet_prediction": schnet_predictions,
        "lightgbm_absolute_error": np.abs(targets - lightgbm_predictions),
        "schnet_absolute_error": np.abs(targets - schnet_predictions),
        "scaffold": scaffolds[test_indices],
        "scaffold_train_frequency": [
            int(train_frequency.get(scaffold, 0)) for scaffold in scaffolds[test_indices]
        ],
    })
    frame["schnet_improvement"] = (
        frame["lightgbm_absolute_error"] - frame["schnet_absolute_error"]
    )
    for column, values in zip(size_names, size_matrix[test_indices].T):
        frame[column] = values
    for column, values in zip(geometry_names, geometry_matrix[test_indices].T):
        frame[column] = values
    frame["smiles"] = [
        Chem.MolToSmiles(
            remove_hydrogens(mol_from_json(value), context="final-analysis molecule"),
            canonical=True,
        )
        for value in molecule_table.iloc[test_indices]["json_conformer"]
    ]
    frame["split"] = split
    return frame
