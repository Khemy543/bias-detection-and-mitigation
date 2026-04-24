from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── colour palettes ───────────────────────────────────────────────────────────
GENDER_COLORS = ["#C0392B", "#2980B9"]          # Female / Male
RACE_COLORS   = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
AGE_COLORS    = ["#3D9970", "#2ECC71", "#F39C12", "#E74C3C"]


def _add_value_labels(ax: plt.Axes, bars, total: int) -> None:
    """Add count + percentage labels on top of each bar."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + total * 0.005,
            f"{int(h):,}\n({h / total * 100:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )


def plot_gender_distribution(csv_path: Path, output_dir: Path) -> None:
    """Bar chart — count and % of Female vs Male in the raw dataset."""
    df = pd.read_csv(csv_path)
    counts = df["Gender"].value_counts().reindex(["Female", "Male"]).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    bars = ax.bar(counts.index, counts.values, color=GENDER_COLORS,
                  edgecolor="black", linewidth=0.7, alpha=0.9)
    _add_value_labels(ax, bars, len(df))

    ax.set_title("Gender Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Gender")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.savefig(output_dir / "dist_gender.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_race_distribution(csv_path: Path, output_dir: Path) -> None:
    """Bar chart — count and % per race group in the raw dataset."""
    df = pd.read_csv(csv_path)
    order  = ["White", "Black", "Asian", "Other"]
    counts = df["Race"].value_counts().reindex(order).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    bars = ax.bar(counts.index, counts.values, color=RACE_COLORS,
                  edgecolor="black", linewidth=0.7, alpha=0.9)
    _add_value_labels(ax, bars, len(df))

    ax.set_title("Race Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Race")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.savefig(output_dir / "dist_race.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_age_distribution(csv_path: Path, output_dir: Path) -> None:
    """Bar chart — count and % per age group in the raw dataset."""
    df = pd.read_csv(csv_path)
    order  = ["Under_30", "30_to_45", "45_to_60", "Over_60"]
    labels = ["Under 30", "30–45", "45–60", "Over 60"]
    counts = df["Age_Group"].value_counts().reindex(order).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    bars = ax.bar(labels, counts.values, color=AGE_COLORS,
                  edgecolor="black", linewidth=0.7, alpha=0.9)
    _add_value_labels(ax, bars, len(df))

    ax.set_title("Age Group Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Age Group")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.savefig(output_dir / "dist_age_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of Pearson correlations across all numeric features (encoded df)."""
    # Drop non-numeric / label columns
    drop_cols = [c for c in ["Race_label", "Candidate_ID"] if c in df.columns]
    numeric_df = df.drop(columns=drop_cols).select_dtypes(include=[np.number])

    # Friendly column name mapping
    rename = {
        "Gender": "Gender",
        "Age_Group": "Age Group",
        "Education": "Education",
        "Experience_Years": "Experience",
        "Certifications": "Certifications",
        "Screening_Score": "Screening Score",
        "Selected": "Selected",
        "Race_Asian": "Race: Asian",
        "Race_Black": "Race: Black",
        "Race_Other": "Race: Other",
        "Race_White": "Race: White",
    }
    numeric_df = numeric_df.rename(columns=rename)

    corr = numeric_df.corr()
    n = len(corr)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(7, n * 0.7)), constrained_layout=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix (Pre-Mitigation)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    fig.savefig(output_dir / "correlation_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
