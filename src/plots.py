from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


MODEL_ORDER = ["LogisticRegression", "NeuralNet", "RandomForest"]
VARIANT_ORDER = ["pre", "in_process", "post"]
VARIANT_LABEL = {"pre": "Pre-mitigation", "in_process": "In-process", "post": "Post-process"}
VARIANT_COLOR = {"pre": "#C0392B", "in_process": "#7A8DA8", "post": "#2E8B57"}


def _error_size(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _bar_positions(n_models: int, n_variants: int = 3, width: float = 0.25) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    x = np.arange(n_models)
    offsets = np.linspace(-(n_variants - 1) / 2, (n_variants - 1) / 2, n_variants) * width
    positions = {v: x + offsets[i] for i, v in enumerate(VARIANT_ORDER)}
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x, positions, width = _bar_positions(len(MODEL_ORDER))

    for axis, metric in zip(axes, metrics):
        y_max = 0.0
        for variant in VARIANT_ORDER:
            means, errors = [], []
            for model_name in MODEL_ORDER:
                values = per_run[
                    (per_run["model"] == model_name) & (per_run["variant"] == variant)
                ][metric]
                means.append(float(np.nanmean(values)))
                errors.append(_error_size(values.dropna()))
            y_max = max(y_max, max(np.array(means) + np.array(errors)))
            axis.bar(
                positions[variant], means, width=width,
                yerr=errors, capsize=3,
                color=VARIANT_COLOR[variant],
                label=VARIANT_LABEL[variant],
                edgecolor="black", linewidth=0.6, alpha=0.92,
            )

        # Stars for pre vs post comparison
        for idx, model_name in enumerate(MODEL_ORDER):
            row = ptests[
                (ptests["model"] == model_name) &
                (ptests["comparison"] == "pre_vs_post") &
                (ptests["metric"] == metric)
            ]
            p = float(row["p_value"].iloc[0]) if len(row) else np.nan
            stars = _significance_stars(p)
            if stars:
                axis.text(x[idx], y_max * 1.07 if y_max > 0 else 0.005,
                          stars, ha="center", va="bottom", fontsize=13)

        axis.set_title(metric)
        axis.set_xticks(x)
        axis.set_xticklabels(MODEL_ORDER, rotation=10)
        axis.set_ylabel(metric)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Fairness Metrics Across Mitigation Phases", fontsize=13, y=1.03)
    fig.savefig(output_dir / "fairness_main.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(per_run: pd.DataFrame, ptests: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x, positions, width = _bar_positions(len(MODEL_ORDER))

    y_max = 0.0
    for variant in VARIANT_ORDER:
        means, errors = [], []
        for model_name in MODEL_ORDER:
            values = per_run[
                (per_run["model"] == model_name) & (per_run["variant"] == variant)
            ]["accuracy"]
            means.append(float(np.nanmean(values)))
            errors.append(_error_size(values.dropna()))
        y_max = max(y_max, max(np.array(means) + np.array(errors)))
        axis.bar(
            positions[variant], means, width=width,
            yerr=errors, capsize=3,
            color=VARIANT_COLOR[variant],
            label=VARIANT_LABEL[variant],
            edgecolor="black", linewidth=0.6, alpha=0.92,
        )

    for idx, model_name in enumerate(MODEL_ORDER):
        row = ptests[
            (ptests["model"] == model_name) &
            (ptests["comparison"] == "pre_vs_post") &
            (ptests["metric"] == "accuracy")
        ]
        p = float(row["p_value"].iloc[0]) if len(row) else np.nan
        stars = _significance_stars(p)
        if stars:
            axis.text(x[idx], y_max * 1.015, stars, ha="center", va="bottom", fontsize=13)

    axis.set_xticks(x)
    axis.set_xticklabels(MODEL_ORDER, rotation=10)
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy Across Mitigation Phases")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right")
    fig.savefig(output_dir / "accuracy_secondary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


GROUP_LABELS = {0: "Female", 1: "Male"}


def _draw_cm(ax: plt.Axes, cm: np.ndarray, title: str) -> None:
    """Draw a single annotated confusion matrix on ax."""
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    tick_labels = ["Not Selected", "Selected"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8, rotation=90, va="center")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > thresh else "black",
            )


def plot_baseline_confusion_matrices(baseline_preds: dict, output_dir: Path) -> None:
    """
    One confusion-matrix PNG per baseline model — no retraining required.
    baseline_preds: {"y_test": Series, "models": {model_name: y_pred ndarray}}
    Output files: confusion_matrix_<ModelName>_baseline.png
    """
    y_true = np.asarray(baseline_preds["y_test"])
    for model_name in MODEL_ORDER:
        y_pred = np.asarray(baseline_preds["models"][model_name])
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
        _draw_cm(ax, cm, f"{model_name} — Baseline")
        fig.savefig(
            output_dir / f"confusion_matrix_{model_name}_baseline.png",
            dpi=220, bbox_inches="tight",
        )
        plt.close(fig)


def plot_post_confusion_matrices(snap_preds: dict, output_dir: Path) -> None:
    """
    One confusion-matrix PNG per post-mitigated model.
    snap_preds: {"y_test": Series, "models": {model_name: {"pre": ndarray, "in_process": ndarray, "post": ndarray}}}
    Output files: confusion_matrix_<ModelName>_post.png
    """
    y_true = np.asarray(snap_preds["y_test"])
    for model_name in MODEL_ORDER:
        y_pred = np.asarray(snap_preds["models"][model_name]["post"])
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
        _draw_cm(ax, cm, f"{model_name} — Post-mitigated")
        fig.savefig(
            output_dir / f"confusion_matrix_{model_name}_post.png",
            dpi=220, bbox_inches="tight",
        )
        plt.close(fig)


def plot_confusion_matrices(
    baseline_preds: dict,
    post_preds: dict,
    output_dir: Path,
) -> None:
    """
    2 × 3 grid: row 0 = baseline, row 1 = post-mitigated.
    baseline_preds / post_preds: {"y_test": Series, "models": {name: y_pred ndarray}}
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

    for col, model_name in enumerate(MODEL_ORDER):
        for row, (label, preds) in enumerate(
            [("Baseline", baseline_preds), ("Post-mitigated", post_preds)]
        ):
            y_true = np.asarray(preds["y_test"])
            y_pred = np.asarray(preds["models"][model_name])
            cm = confusion_matrix(y_true, y_pred)
            _draw_cm(axes[row, col], cm, f"{model_name}\n({label})")

    fig.suptitle("Confusion Matrices — Baseline vs Post-mitigated", fontsize=13, fontweight="bold")
    fig.savefig(output_dir / "confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


AGE_GROUP_LABELS = {0: "Under 30", 1: "30-45", 2: "45-60", 3: "Over 60"}
MODEL_COLORS = ["#4878CF", "#6ACC65", "#D65F5F"]  # LR / NeuralNet / RF


def _plot_selection_rate_by_attribute(
    groups: list,
    group_labels: list[str],
    sensitive: np.ndarray,
    preds_by_model: dict,
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Bar chart: x = demographic groups, grouped bars = one per model."""
    n_groups = len(groups)
    n_models = len(MODEL_ORDER)
    width = 0.22
    x = np.arange(n_groups)
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.8), 5), constrained_layout=True)

    for m_idx, model_name in enumerate(MODEL_ORDER):
        y_pred = np.asarray(preds_by_model[model_name])
        rates = []
        for g in groups:
            mask = sensitive == g
            rate = float(y_pred[mask].mean()) * 100 if mask.any() else 0.0
            rates.append(rate)

        bars = ax.bar(
            x + offsets[m_idx], rates, width=width,
            color=MODEL_COLORS[m_idx], edgecolor="black", linewidth=0.6,
            alpha=0.9, label=model_name,
        )
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2, rate + 0.1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Selection rate (%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selection_rates(baseline_preds: dict, output_dir: Path) -> None:
    """
    Three separate PNGs — one per sensitive attribute — showing baseline selection
    rate (%) per demographic group, with one bar per model.
    Output files:
      selection_rate_gender.png
      selection_rate_race.png
      selection_rate_age_group.png
    """
    preds = baseline_preds["models"]

    # Gender
    gender_arr = np.asarray(baseline_preds["sensitive_test"])
    gender_groups = sorted(GROUP_LABELS.keys())
    _plot_selection_rate_by_attribute(
        groups=gender_groups,
        group_labels=[GROUP_LABELS[g] for g in gender_groups],
        sensitive=gender_arr,
        preds_by_model=preds,
        title="Baseline Selection Rate by Gender",
        filename="selection_rate_gender.png",
        output_dir=output_dir,
    )

    # Race
    race_arr = np.asarray(baseline_preds["race_test"])
    race_groups = sorted(np.unique(race_arr).tolist())
    _plot_selection_rate_by_attribute(
        groups=race_groups,
        group_labels=race_groups,
        sensitive=race_arr,
        preds_by_model=preds,
        title="Baseline Selection Rate by Race",
        filename="selection_rate_race.png",
        output_dir=output_dir,
    )

    # Age Group
    age_arr = np.asarray(baseline_preds["age_group_test"])
    age_groups = sorted(AGE_GROUP_LABELS.keys())
    _plot_selection_rate_by_attribute(
        groups=age_groups,
        group_labels=[AGE_GROUP_LABELS[g] for g in age_groups],
        sensitive=age_arr,
        preds_by_model=preds,
        title="Baseline Selection Rate by Age Group",
        filename="selection_rate_age_group.png",
        output_dir=output_dir,
    )
