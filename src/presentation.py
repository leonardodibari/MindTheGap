"""Presentation-specific figures generated from saved experiment artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import save_figure

SPLIT_COLORS = {"random": "#4C78A8", "scaffold": "#F58518"}
MODEL_COLORS = {
    "Mean predictor": "#9E9E9E",
    "Ridge": "#D55E00",
    "LightGBM": "#009E73",
    "Gradient boosting": "#CC79A7",
}

GEOMETRIC_DESCRIPTORS = [
    "radius_of_gyration", "mean_bond_length", "std_bond_length",
    "max_atomic_distance", "coordinate_cov_eigenvalue_1",
    "coordinate_cov_eigenvalue_2", "coordinate_cov_eigenvalue_3",
]


def plot_target_distribution(target: pd.Series, output_dir: Path) -> Path:
    """Plot the full and central target distributions used in the presentation."""
    gap = target.dropna()
    zoom_low, zoom_high = gap.quantile([0.0005, 0.9995])
    central_gap = gap[gap.between(zoom_low, zoom_high)]
    with plt.rc_context({
        "font.size": 24, "axes.titlesize": 30, "axes.labelsize": 30,
        "xtick.labelsize": 30, "ytick.labelsize": 30,
    }):
        fig, axes = plt.subplots(2, 1, figsize=(13, 12), constrained_layout=True)
        axes[0].hist(gap, bins=70, color="#4C78A8", edgecolor="white")
        axes[0].set_yscale("log")
        axes[0].set(xlabel="HOMO–LUMO gap", ylabel="Molecules (log scale)")
        axes[1].hist(central_gap, bins=55, color="#4C78A8", edgecolor="white")
        axes[1].set(
            xlabel="HOMO–LUMO gap (central 99.9 %)", ylabel="Molecules",
        )
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    return save_figure("target_distribution_vertical", output_dir, figure=fig)


def _residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(control)), control])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def plot_geometry_partial_correlations(
    metadata: pd.DataFrame, targets: pd.DataFrame, output_dir: Path,
) -> Path:
    """Plot geometry–target rank associations after controlling for molecular size."""
    frame = metadata.set_index("mol_id")[GEOMETRIC_DESCRIPTORS + ["n_heavy_atoms"]].join(
        targets.set_index("mol_id")["gap"]
    )
    correlations = {}
    for descriptor in GEOMETRIC_DESCRIPTORS:
        pair = frame[[descriptor, "n_heavy_atoms", "gap"]].dropna()
        correlations[descriptor] = np.corrcoef(
            _residualize(pair[descriptor].rank().to_numpy(), pair["n_heavy_atoms"].rank().to_numpy()),
            _residualize(pair["gap"].rank().to_numpy(), pair["n_heavy_atoms"].rank().to_numpy()),
        )[0, 1]
    partial_corr = pd.Series(correlations).sort_values()
    label_map = {
        "radius_of_gyration": "Radius of gyration",
        "mean_bond_length": "Mean bond length",
        "std_bond_length": "Bond-length variability",
        "max_atomic_distance": "Maximum atomic distance",
        "coordinate_cov_eigenvalue_1": "Coordinate covariance eig. 1",
        "coordinate_cov_eigenvalue_2": "Coordinate covariance eig. 2",
        "coordinate_cov_eigenvalue_3": "Coordinate covariance eig. 3",
    }
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    bars = ax.barh(
        [label_map[name] for name in partial_corr.index], partial_corr.values,
        color="#4472C4", edgecolor="white",
    )
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel(
        "Partial Spearman correlation \n with HOMO–LUMO gap\n"
        "(controlling for heavy-atom count)", fontsize=18,
    )
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    offset = 0.008 * max(1, partial_corr.abs().max())
    for bar, value in zip(bars, partial_corr.values):
        ax.text(
            value + offset if value >= 0 else value - offset,
            bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center",
            ha="left" if value >= 0 else "right", fontsize=15, fontweight="bold",
        )
    ax.set_xlim(-0.2, partial_corr.abs().max() * 1.25)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return save_figure("geometry_partial_spearman_ppt", output_dir, figure=fig)


def _feature_name(model: str) -> str:
    """Return a concise feature-family label from a stored classical model name."""
    prefixes = {
        "Mean predictor": "No features",
        "Size/composition": "Size and composition",
        "RDKit 2D": "RDKit 2D descriptors",
        "Morgan + RDKit": "Morgan + RDKit + simple 3D",
        "Morgan": "Morgan fingerprints",
    }
    return next((label for prefix, label in prefixes.items() if model.startswith(prefix)), model)


def _estimator_name(model: str) -> str:
    """Return the estimator family encoded in a stored classical model name."""
    if model == "Mean predictor":
        return model
    for token, label in (
        ("Ridge", "Ridge"),
        ("LightGBM", "LightGBM"),
        ("HistGradientBoosting", "Gradient boosting"),
    ):
        if token in model:
            return label
    return model


def _classical_plot_table(results: pd.DataFrame) -> pd.DataFrame:
    table = results[["split_strategy", "model", "test_mae"]].copy()
    table["features"] = table["model"].map(_feature_name)
    table["model_type"] = table["model"].map(_estimator_name)
    table["display_label"] = np.where(
        table["model_type"].eq("Mean predictor"),
        "Mean predictor",
        table["features"] + "\n" + table["model_type"],
    )
    return table


def _style_horizontal_mae(ax, labels: Sequence[str], x_max: float) -> None:
    positions = np.arange(len(labels))
    ax.set_yticks(positions, labels, fontsize=19, linespacing=1.1)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Test MAE", fontsize=22, fontweight="semibold", labelpad=12)
    ax.tick_params(axis="x", labelsize=17, length=0)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", linewidth=0.8, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)


def plot_classical_random_models(results: pd.DataFrame, output_dir: Path) -> Path:
    """Plot random-test MAE for every evaluated classical baseline."""
    table = _classical_plot_table(results)
    feature_order = [
        "No features", "Size and composition", "RDKit 2D descriptors",
        "Morgan fingerprints", "Morgan + RDKit + simple 3D",
    ]
    model_order = ["Mean predictor", "Ridge", "LightGBM", "Gradient boosting"]
    table = table.query("split_strategy == 'random'").copy()
    table["feature_rank"] = table["features"].map({v: i for i, v in enumerate(feature_order)})
    table["model_rank"] = table["model_type"].map({v: i for i, v in enumerate(model_order)})
    table = table.sort_values(["feature_rank", "model_rank"]).reset_index(drop=True)

    x_max = table["test_mae"].max() * 1.23
    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    bars = ax.barh(
        np.arange(len(table)), table["test_mae"], height=0.58,
        color=table["model_type"].map(MODEL_COLORS), edgecolor="none", zorder=3,
    )
    for bar, value in zip(bars, table["test_mae"]):
        ax.text(value + x_max * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=17, fontweight="semibold")
    _style_horizontal_mae(ax, table["display_label"], x_max)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[name], label=name)
               for name in model_order if name in table["model_type"].values]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=len(handles), frameon=False, fontsize=16)
    return save_figure("random_split_model_comparison", output_dir, figure=fig)


def plot_classical_split_comparison(results: pd.DataFrame, output_dir: Path) -> Path:
    """Compare random and scaffold MAE for classical models evaluated on both."""
    table = _classical_plot_table(results)
    paired = table.pivot_table(
        index=["model", "display_label"], columns="split_strategy",
        values="test_mae", aggfunc="first",
    ).dropna(subset=["random", "scaffold"]).reset_index()
    paired = paired.sort_values("scaffold").reset_index(drop=True)
    x_max = paired[["random", "scaffold"]].to_numpy().max() * 1.23
    y = np.arange(len(paired))
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    for split, offset in (("random", -0.17), ("scaffold", 0.17)):
        bars = ax.barh(y + offset, paired[split], height=0.32,
                       color=SPLIT_COLORS[split], label=f"{split.title()} test", zorder=3)
        for bar, value in zip(bars, paired[split]):
            ax.text(value + x_max * 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}", va="center", fontsize=17, fontweight="semibold")
    _style_horizontal_mae(ax, paired["display_label"], x_max)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize=17)
    return save_figure("random_vs_scaffold_matched_models", output_dir, figure=fig)


def plot_lightgbm_vs_schnet(results: pd.DataFrame, output_dir: Path) -> Path:
    """Compare LightGBM and SchNet test MAE across both split strategies."""
    table = results.copy()
    table["display_model"] = np.select(
        [table["model"].str.contains("LightGBM"), table["model"].str.contains("Ensemble")],
        ["LightGBM", "Ensemble"], default="SchNet",
    )
    values = table.pivot(index="display_model", columns="split", values="test_mae")
    models = ["LightGBM", "SchNet"]
    x = np.arange(len(models))
    width = 0.34
    y_max = values.loc[models, ["random", "scaffold"]].to_numpy().max() * 1.35
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for split, offset in (("random", -width / 2), ("scaffold", width / 2)):
        split_values = values.loc[models, split].to_numpy()
        bars = ax.bar(x + offset, split_values, width, label=f"{split.title()} test",
                      color=SPLIT_COLORS[split], edgecolor="none", zorder=3)
        for bar, value in zip(bars, split_values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + y_max * 0.018,
                    f"{value:.4f}", ha="center", fontsize=18, fontweight="semibold")
    ax.set_xticks(x, models, fontsize=21)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Test MAE", fontsize=23, fontweight="semibold")
    ax.tick_params(axis="y", labelsize=18, length=0)
    ax.grid(axis="y", alpha=0.18, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize=18)
    return save_figure("01_lightgbm_vs_schnet_mae", output_dir, figure=fig)


def plot_ensemble_mae(results: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the final ensemble MAE for random and scaffold tests."""
    ensemble = (
        results[results["model"].str.contains("Ensemble")]
        .set_index("split").loc[["random", "scaffold"], "test_mae"]
    )
    labels = ["Random test", "Scaffold test"]
    values = ensemble.to_numpy()
    y_max = values.max() * 1.3
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    bars = ax.bar(labels, values, width=0.58,
                  color=[SPLIT_COLORS[split] for split in ensemble.index], zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + y_max * 0.018,
                f"{value:.4f}", ha="center", fontsize=19, fontweight="semibold")
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Ensemble test MAE", fontsize=23, fontweight="semibold")
    ax.tick_params(axis="x", labelsize=20, length=0)
    ax.tick_params(axis="y", labelsize=18, length=0)
    ax.grid(axis="y", alpha=0.18, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return save_figure("02_ensemble_only_mae", output_dir, figure=fig)


def plot_model_complementarity(results: pd.DataFrame, output_dir: Path) -> Path:
    """Summarize per-molecule wins and error-rank correlation by split."""
    table = results.set_index("split").loc[["random", "scaffold"]]
    labels = ["Random test", "Scaffold test"]
    y = np.arange(len(labels))
    lightgbm = table["lightgbm_lower_error_proportion"].to_numpy()
    schnet = table["schnet_lower_error_proportion"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax = axes[0]
    ax.barh(y, lightgbm, color="#2E8B57", height=0.55, label="LightGBM lower error")
    ax.barh(y, schnet, left=lightgbm, color="#7B3294", height=0.55,
            label="SchNet lower error")
    for index in range(len(y)):
        ax.text(lightgbm[index] / 2, y[index], f"{lightgbm[index]:.1%}",
                ha="center", va="center", color="white", fontsize=18, fontweight="semibold")
        ax.text(lightgbm[index] + schnet[index] / 2, y[index], f"{schnet[index]:.1%}",
                ha="center", va="center", color="white", fontsize=18, fontweight="semibold")
    ax.set_yticks(y, labels, fontsize=19)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks, [f"{value:.0%}" for value in ticks], fontsize=17)
    ax.set_xlabel("Fraction of molecules", fontsize=21)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize=15)

    ax = axes[1]
    x = np.arange(len(labels))
    width = 0.32
    signed = table["signed_residual_spearman"].to_numpy()
    absolute = table["absolute_error_spearman"].to_numpy()
    for values, offset, label, color in (
        (signed, -width / 2, "Signed residuals", "#5B8FF9"),
        (absolute, width / 2, "Absolute errors", "#61DDAA"),
    ):
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}",
                    ha="center", fontsize=17, fontweight="semibold")
    ax.set_xticks(x, labels, fontsize=19)
    ax.set_ylim(0, max(signed.max(), absolute.max()) * 1.3)
    ax.set_ylabel("Spearman correlation", fontsize=21)
    ax.tick_params(axis="y", labelsize=17, length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize=15)
    for axis in axes:
        axis.grid(axis="x" if axis is axes[0] else "y", alpha=0.18)
        axis.spines[["top", "right", "left"]].set_visible(False)
    return save_figure("model_complementarity_summary_spearman", output_dir, figure=fig)


def _second_model_probability(row: pd.Series, first: str, second: str) -> float:
    probability = float(row["bootstrap_proportion_favoring_second"])
    if row["first_model"] == first and row["second_model"] == second:
        return probability
    if row["first_model"] == second and row["second_model"] == first:
        return 1 - probability
    raise ValueError(f"Row does not compare {first!r} and {second!r}")


def _comparison_row(frame: pd.DataFrame, split: str, first: str, second: str) -> pd.Series:
    rows = frame[
        frame["split"].eq(split)
        & (((frame["first_model"] == first) & (frame["second_model"] == second))
           | ((frame["first_model"] == second) & (frame["second_model"] == first)))
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one {first}/{second} comparison for {split}; found {len(rows)}")
    return rows.iloc[0]


def plot_bootstrap_win_probabilities(results: pd.DataFrame, output_dir: Path) -> Path:
    """Plot bootstrap win probabilities for the individual and ensemble comparisons."""
    rows = []
    for split in ("random", "scaffold"):
        individual = _comparison_row(results, split, "LightGBM", "SchNet")
        p_schnet = _second_model_probability(individual, "LightGBM", "SchNet")
        best = "SchNet" if p_schnet >= 0.5 else "LightGBM"
        ensemble = _comparison_row(results, split, best, "Ensemble")
        rows.extend([
            {"split": split, "comparison": "LightGBM → SchNet", "probability": 100 * p_schnet},
            {"split": split, "comparison": f"{best} → Ensemble",
             "probability": 100 * _second_model_probability(ensemble, best, "Ensemble")},
        ])
    plot_data = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(11, 11), sharey=True)
    for ax, split in zip(axes, ("random", "scaffold")):
        subset = plot_data.query("split == @split").reset_index(drop=True)
        bars = ax.bar(np.arange(len(subset)), subset["probability"], width=0.68,
                      color=SPLIT_COLORS[split], edgecolor="none")
        ax.axhline(50, linestyle="--", linewidth=1.8, color="#555555")
        for bar, probability in zip(bars, subset["probability"]):
            ax.text(bar.get_x() + bar.get_width() / 2, min(probability + 2.5, 104),
                    f"{probability:.0f}%", ha="center", fontsize=22, fontweight="semibold")
        ax.set_title(f"{split.title()} test", fontsize=25, fontweight="semibold")
        ax.set_xticks(np.arange(len(subset)), subset["comparison"], fontsize=21)
        ax.set_ylim(0, 108)
        ax.tick_params(axis="y", labelsize=18, length=0)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.supylabel("Bootstrap probability that\nsecond model performs better (%)",
                  fontsize=22, x=0.02)
    return save_figure("bootstrap_probability_random_scaffold", output_dir, figure=fig)
