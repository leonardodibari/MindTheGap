"""Reusable constructors for the exact classical baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import RANDOM_SEED

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


def to_dense(matrix):
    """Convert a sparse matrix to dense, leaving dense inputs as arrays."""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def make_booster(params: dict[str, Any], random_state: int = RANDOM_SEED):
    """Construct the project's LightGBM model or its documented sklearn fallback."""
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="regression_l1", n_estimators=30, learning_rate=.1,
            num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
            subsample=.9, colsample_bytree=.5, max_bin=63, reg_lambda=1.,
            random_state=random_state, n_jobs=-1, verbosity=-1,
        )
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=.05, max_iter=200,
        max_leaf_nodes=params["num_leaves"], min_samples_leaf=params["min_child_samples"],
        l2_regularization=1., random_state=random_state,
    )


def build_pipeline(
    kind: str,
    params: dict[str, Any],
    *,
    n_fingerprint_features: int | None = None,
    n_total_features: int | None = None,
):
    """Build an existing baseline pipeline while keeping choices explicit at the call site."""
    if kind == "mean":
        return Pipeline([("regressor", DummyRegressor(strategy="mean"))])
    if kind == "dense_ridge":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("regressor", Ridge(alpha=params["alpha"], solver="lsqr"))])
    if kind == "sparse_ridge":
        return Pipeline([("scaler", StandardScaler(with_mean=False)),
                         ("regressor", Ridge(alpha=params["alpha"], solver="lsqr"))])
    if kind == "dense_boost":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("regressor", make_booster(params))])
    if kind == "sparse_boost":
        steps = []
        if LGBMRegressor is None:
            steps.append(("densify", FunctionTransformer(to_dense, accept_sparse=True)))
        return Pipeline([*steps, ("regressor", make_booster(params))])
    if kind == "combined_boost":
        if n_fingerprint_features is None or n_total_features is None:
            raise ValueError("Combined pipelines require fingerprint and total feature counts")
        preprocess = ColumnTransformer([
            ("fingerprint", "passthrough", slice(0, n_fingerprint_features)),
            ("descriptors", SimpleImputer(strategy="median"),
             slice(n_fingerprint_features, n_total_features)),
        ], sparse_threshold=1.)
        steps = [("preprocess", preprocess)]
        if LGBMRegressor is None:
            steps.append(("densify", FunctionTransformer(to_dense, accept_sparse=True)))
        return Pipeline([*steps, ("regressor", make_booster(params))])
    raise ValueError(f"Unknown pipeline kind: {kind}")


def matrix_rows(matrix, indices: np.ndarray) -> np.ndarray:
    """Select rows, providing the dummy predictor with a valid placeholder matrix."""
    if matrix is None:
        return np.zeros((len(indices), 1), dtype=np.float32)
    return matrix[indices]

