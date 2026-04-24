from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

# AIF360
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

# FairLearn
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

TARGET_POS_RATE = 0.25
MODELS = ["LogisticRegression", "NeuralNet", "RandomForest"]


# ── Fairness metrics (same definitions as our pipeline) ───────────────────────

def _di(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    rates = [float(np.mean(y_pred[sensitive == g])) for g in np.unique(sensitive)]
    return float(min(rates) / max(rates)) if max(rates) > 0 else float("nan")


def _dpd(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    rates = [float(np.mean(y_pred[sensitive == g])) for g in np.unique(sensitive)]
    return float(max(rates) - min(rates))


def _eod(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    tprs, fprs = [], []
    for g in np.unique(sensitive):
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        tprs.append(float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1)))
        fprs.append(float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1)))
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))


def _metrics(y_true, y_pred, sensitive) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "DI":  _di(y_pred, sensitive),
        "DPD": _dpd(y_pred, sensitive),
        "EOD": _eod(y_true, y_pred, sensitive),
    }


def _make_model(name: str, seed: int):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, random_state=seed,
                                      class_weight="balanced_subsample")
    return MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _split(df: pd.DataFrame, seed: int = 42):
    features = [c for c in df.columns if c not in ["Candidate_ID", "Selected", "Race_label"]]
    strat = df[["Selected", "Gender"]].astype(str).agg("_".join, axis=1)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat)
    fit_df, _ = train_test_split(train_df, test_size=0.25,
                                 random_state=seed + 1, stratify=train_df["Selected"])

    x_train, x_test = train_df[features], test_df[features]
    y_train, y_test = train_df["Selected"], test_df["Selected"]
    x_fit, y_fit = fit_df[features], fit_df["Selected"]
    sensitive_train = train_df["Gender"].reset_index(drop=True)
    sensitive_test  = test_df["Gender"].reset_index(drop=True)
    return x_train, x_test, y_train, y_test, x_fit, y_fit, sensitive_train, sensitive_test, features


def _threshold_pred(model, x, cal_proba) -> np.ndarray:
    threshold = float(np.quantile(cal_proba, 1.0 - TARGET_POS_RATE))
    return (model.predict_proba(x)[:, 1] >= threshold).astype(int)


# ── AIF360 Reweighing ─────────────────────────────────────────────────────────

def _run_aif360(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    AIF360 Reweighing (pre-processing) + Calibrated Equalized Odds (post-processing).
    Returns one row per model with accuracy, DI, DPD, EOD.
    """
    x_train, x_test, y_train, y_test, x_fit, y_fit, s_train, s_test, features = _split(df, seed)

    rows = []
    for model_name in MODELS:
        model = _make_model(model_name, seed)

        # Build AIF360 BinaryLabelDataset for the training set
        train_aif = pd.concat([x_train.reset_index(drop=True),
                                s_train.reset_index(drop=True).rename("Gender"),
                                y_train.reset_index(drop=True)], axis=1)
        aif_ds = BinaryLabelDataset(
            df=train_aif,
            label_names=["Selected"],
            protected_attribute_names=["Gender"],
            favorable_label=1,
            unfavorable_label=0,
        )

        # Reweighing: compute sample weights to reduce disparate impact
        rw = Reweighing(
            unprivileged_groups=[{"Gender": 0}],   # Female
            privileged_groups=[{"Gender": 1}],      # Male
        )
        rw.fit(aif_ds)
        aif_ds_rw = rw.transform(aif_ds)
        sample_weights = aif_ds_rw.instance_weights

        # Train with reweighted samples
        if model_name == "NeuralNet":
            model.fit(x_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(x_train, y_train, sample_weight=sample_weights)

        # Threshold prediction (same rate as our pipeline)
        cal_proba = model.predict_proba(x_fit)[:, 1]
        threshold = float(np.quantile(cal_proba, 1.0 - TARGET_POS_RATE))
        y_pred = (model.predict_proba(x_test)[:, 1] >= threshold).astype(int)

        m = _metrics(y_test, y_pred, s_test)
        rows.append({"model": model_name, "toolkit": "AIF360 (Reweighing)", **m})

    return pd.DataFrame(rows)


# ── FairLearn ThresholdOptimizer ──────────────────────────────────────────────

def _run_fairlearn(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    FairLearn ThresholdOptimizer (post-processing, demographic parity constraint).
    Returns one row per model with accuracy, DI, DPD, EOD.
    """
    x_train, x_test, y_train, y_test, x_fit, y_fit, s_train, s_test, features = _split(df, seed)

    rows = []
    for model_name in MODELS:
        model = _make_model(model_name, seed)

        if model_name == "NeuralNet":
            sw = compute_sample_weight("balanced", y_train)
            model.fit(x_train, y_train, sample_weight=sw)
        else:
            model.fit(x_train, y_train)

        # ThresholdOptimizer wraps the trained model and finds group thresholds
        to = ThresholdOptimizer(
            estimator=model,
            constraints="demographic_parity",
            objective="accuracy_score",
            predict_method="predict_proba",
        )
        to.fit(x_train, y_train, sensitive_features=s_train)
        y_pred = to.predict(x_test, sensitive_features=s_test)

        m = _metrics(y_test, y_pred, s_test)
        rows.append({"model": model_name, "toolkit": "FairLearn (ThresholdOptimizer)", **m})

    return pd.DataFrame(rows)


# ── Our pipeline results (from summary_stats.csv) ────────────────────────────

def _load_our_post(output_dir: Path) -> pd.DataFrame:
    snap = pd.read_csv(output_dir / "summary_stats.csv")
    post = snap[snap["variant"] == "post"][["model", "accuracy_mean", "DI_mean", "DPD_mean", "EOD_mean"]].copy()
    post = post.rename(columns={
        "accuracy_mean": "accuracy",
        "DI_mean": "DI",
        "DPD_mean": "DPD",
        "EOD_mean": "EOD",
    })
    post["toolkit"] = "Ours (Post-process)"
    return post[["model", "toolkit", "accuracy", "DI", "DPD", "EOD"]]


# ── Comparison plot ───────────────────────────────────────────────────────────

def _plot_comparison(combined: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["accuracy", "DI", "DPD", "EOD"]
    titles  = ["Accuracy", "Disparate Impact (DI)", "Demographic Parity Diff. (DPD)",
               "Equalized Odds Diff. (EOD)"]

    toolkits = combined["toolkit"].unique().tolist()
    colors   = ["#2E8B57", "#2980B9", "#E67E22"]
    width    = 0.25
    x        = np.arange(len(MODELS))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for i, (toolkit, color) in enumerate(zip(toolkits, colors)):
            vals = [
                combined[(combined["model"] == m) & (combined["toolkit"] == toolkit)][metric].values
                for m in MODELS
            ]
            vals = [float(v[0]) if len(v) else 0.0 for v in vals]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width=width, color=color,
                          edgecolor="black", linewidth=0.6, alpha=0.9, label=toolkit)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        if metric == "DI":
            ax.axhline(0.80, color="red", linestyle="--", linewidth=1.2, label="Legal threshold (0.80)")

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=10, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(toolkits) + 1,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Fairness Toolkit Comparison — Post-Mitigation Results",
                 fontsize=13, fontweight="bold")

    fig.savefig(output_dir / "toolkit_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_toolkit_comparison(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Run AIF360 and FairLearn mitigation, compare to our post-process results,
    save toolkit_comparison.png and toolkit_comparison.csv.
    """
    print("  Running AIF360 Reweighing...")
    aif_results = _run_aif360(df)

    print("  Running FairLearn ThresholdOptimizer...")
    fl_results  = _run_fairlearn(df)

    print("  Loading our post-process results...")
    our_results = _load_our_post(output_dir)

    combined = pd.concat([our_results, aif_results, fl_results], ignore_index=True)
    combined = combined.sort_values(["model", "toolkit"]).reset_index(drop=True)

    combined.to_csv(output_dir / "toolkit_comparison.csv", index=False)
    _plot_comparison(combined, output_dir)

    return combined
