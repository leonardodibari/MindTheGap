"""Regression metrics, residual summaries, and deterministic bootstrap helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE, and R² using the project's metric definitions."""
    true, predicted = np.asarray(y_true), np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(true, predicted)),
        "rmse": float(mean_squared_error(true, predicted) ** .5),
        "r2": float(r2_score(true, predicted)),
    }


def metric_row(
    split: str, model: str, y_true, y_pred, validation_mae: float,
    training_time: float = np.nan, parameters: float = np.nan,
) -> dict[str, object]:
    """Build one row in the existing consolidated metric schema."""
    metrics = regression_metrics(y_true, y_pred)
    return {
        "split": split, "model": model, "validation_mae": validation_mae,
        "test_mae": metrics["mae"], "test_rmse": metrics["rmse"], "test_r2": metrics["r2"],
        "training_time_seconds": training_time, "trainable_parameters": parameters,
    }


def residual_summary(
    frame: pd.DataFrame, split: str, model_columns: Mapping[str, str]
) -> pd.DataFrame:
    """Summarize signed and absolute residuals for multiple prediction columns."""
    rows = []
    for model, column in model_columns.items():
        residual = frame[column] - frame["target"]
        absolute = residual.abs()
        rows.append({
            "split": split, "model": model, "samples": len(frame),
            "mean_residual": residual.mean(), "mae": absolute.mean(),
            "rmse": np.sqrt(np.mean(residual ** 2)),
            "median_absolute_error": absolute.median(),
            "absolute_error_p90": absolute.quantile(.90),
        })
    return pd.DataFrame(rows)


def bootstrap_interval(
    values,
    metric: Callable[[np.ndarray], float] = np.mean,
    *,
    repetitions: int = 500,
    confidence: float = .95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a scalar metric with the existing resampling convention."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for index in range(repetitions):
        samples[index] = metric(rng.choice(values, size=len(values), replace=True))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return float(low), float(high)


def bootstrap_model_comparison(
    frame: pd.DataFrame,
    split: str,
    model_columns: Mapping[str, str],
    *,
    repetitions: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the paired MAE/RMSE bootstrap used in notebook 05."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(frame), size=(repetitions, len(frame)))
    target = frame["target"].to_numpy()
    errors = {model: frame[column].to_numpy() - target for model, column in model_columns.items()}
    metric_rows, boot_mae = [], {}
    for model, residual in errors.items():
        absolute = np.abs(residual)
        mae_samples = absolute[indices].mean(axis=1)
        rmse_samples = np.sqrt((residual[indices] ** 2).mean(axis=1))
        boot_mae[model] = mae_samples
        for metric, estimate, samples in [
            ("MAE", absolute.mean(), mae_samples),
            ("RMSE", np.sqrt(np.mean(residual ** 2)), rmse_samples),
        ]:
            metric_rows.append({"split": split, "model": model, "metric": metric,
                "estimate": estimate, "ci_low": np.quantile(samples, .025),
                "ci_high": np.quantile(samples, .975)})
    comparisons = [
        ("SchNet minus LightGBM", "SchNet", "LightGBM"),
        ("Ensemble minus LightGBM", "Ensemble", "LightGBM"),
        ("Ensemble minus SchNet", "Ensemble", "SchNet"),
    ]
    paired_rows = []
    for label, first, second in comparisons:
        difference = boot_mae[first] - boot_mae[second]
        paired_rows.append({"split": split, "comparison": label, "first_model": first,
            "second_model": second,
            "mean_paired_difference": np.abs(errors[first]).mean() - np.abs(errors[second]).mean(),
            "ci_low": np.quantile(difference, .025), "ci_high": np.quantile(difference, .975),
            "bootstrap_proportion_favoring_second": (difference > 0).mean()})
    return pd.DataFrame(metric_rows), pd.DataFrame(paired_rows)
