from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["LogisticRegression", "RandomForest", "NeuralNet"]
VARIANT_ORDER = ["base", "mitigated"]
VARIANT_LABEL = {"base": "Baseline", "mitigated": "Mitigated"}
VARIANT_COLOR = {"base": "#7A8DA8", "mitigated": "#2E8B57"}


def _error_size(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _bar_positions(n_models: int, width: float = 0.36) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    x = np.arange(n_models)
    positions = {
        "base": x - width / 2,
        "mitigated": x + width / 2,
    }
    return x, positions, width


def _significance_stars(p_value: float | None) -> str:
    if p_value is None or np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plot_fairness(per_run: pd.DataFrame, ptests: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["DPD", "EOD"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x, positions, width = _bar_positions(len(MODEL_ORDER))

    for axis, metric in zip(axes, metrics):
        y_max = 0.0
        for variant in VARIANT_ORDER:
            means = []
            errors = []
            for model_name in MODEL_ORDER:
                values = per_run[
                    (per_run["model"] == model_name) & (per_run["variant"] == variant)
                ][metric]
                means.append(values.mean())
                errors.append(_error_size(values))
            y_max = max(y_max, max(np.array(means) + np.array(errors)))
            axis.bar(
                positions[variant],
                means,
                width=width,
                yerr=errors,
                capsize=4,
                color=VARIANT_COLOR[variant],
                label=VARIANT_LABEL[variant],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.95,
            )

        for index, model_name in enumerate(MODEL_ORDER):
            ptest_row = ptests[(ptests["model"] == model_name) & (ptests["metric"] == metric)]
            p_value = float(ptest_row["p_value"].iloc[0]) if len(ptest_row) else np.nan
            stars = _significance_stars(p_value)
            if stars:
                axis.text(
                    x[index],
                    y_max * 1.06 if y_max > 0 else 0.005,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=14,
                )

        axis.set_title(metric)
        axis.set_xticks(x)
        axis.set_xticklabels(MODEL_ORDER, rotation=10)
        axis.set_ylabel(metric)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Fairness Metrics by Model: Baseline vs Mitigated", fontsize=13, y=1.03)
    fig.savefig(output_dir / "fairness_main.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(per_run: pd.DataFrame, ptests: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    x, positions, width = _bar_positions(len(MODEL_ORDER))

    y_max = 0.0
    for variant in VARIANT_ORDER:
        means = []
        errors = []
        for model_name in MODEL_ORDER:
            values = per_run[
                (per_run["model"] == model_name) & (per_run["variant"] == variant)
            ]["accuracy"]
            means.append(values.mean())
            errors.append(_error_size(values))
        y_max = max(y_max, max(np.array(means) + np.array(errors)))
        axis.bar(
            positions[variant],
            means,
            width=width,
            yerr=errors,
            capsize=4,
            color=VARIANT_COLOR[variant],
            label=VARIANT_LABEL[variant],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
        )

    for index, model_name in enumerate(MODEL_ORDER):
        ptest_row = ptests[(ptests["model"] == model_name) & (ptests["metric"] == "accuracy")]
        p_value = float(ptest_row["p_value"].iloc[0]) if len(ptest_row) else np.nan
        stars = _significance_stars(p_value)
        if stars:
            axis.text(x[index], y_max * 1.015, stars, ha="center", va="bottom", fontsize=14)

    axis.set_xticks(x)
    axis.set_xticklabels(MODEL_ORDER, rotation=10)
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy by Model: Baseline vs Mitigated")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right")
    fig.savefig(output_dir / "accuracy_secondary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
