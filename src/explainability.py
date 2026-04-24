from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

TARGET_POS_RATE = 0.25
N_EXPLAIN = 500   # rows to explain (bar + beeswarm)
N_BACKGROUND = 100  # background rows for NN PermutationExplainer


def _train_models(df: pd.DataFrame, seed: int = 42):
    features = [c for c in df.columns if c not in ["Candidate_ID", "Selected", "Race_label"]]
    strat_key = df[["Selected", "Gender"]].astype(str).agg("_".join, axis=1)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=strat_key
    )
    fit_df, cal_df = train_test_split(
        train_df, test_size=0.25, random_state=seed + 1, stratify=train_df["Selected"]
    )

    x_fit = fit_df[features]
    y_fit = fit_df["Selected"]
    x_train = train_df[features]
    y_train = train_df["Selected"]
    x_test = test_df[features]
    y_test = test_df["Selected"]

    sw_fit = compute_sample_weight("balanced", y_fit)
    sw_train = compute_sample_weight("balanced", y_train)

    models = {}

    lr = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    lr.fit(x_train, y_train)
    models["LogisticRegression"] = lr

    rf = RandomForestClassifier(
        n_estimators=200, random_state=seed, class_weight="balanced_subsample"
    )
    rf.fit(x_train, y_train)
    models["RandomForest"] = rf

    nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)
    nn.fit(x_train, y_train, sample_weight=sw_train)
    models["NeuralNet"] = nn

    return models, x_train, x_test, y_test, features


def _shap_bar(shap_values: shap.Explanation, model_name: str, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title(f"{model_name} — SHAP Feature Importance", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_bar_{model_name}.png", dpi=220, bbox_inches="tight")
    plt.close()


def _shap_beeswarm(shap_values: shap.Explanation, model_name: str, output_dir: Path) -> None:
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values, plot_size=None, show=False)
    plt.title(f"{model_name} — SHAP Summary (Beeswarm)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_beeswarm_{model_name}.png", dpi=220, bbox_inches="tight")
    plt.close()


def run_shap_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Train baseline models on seed=42, compute SHAP values, and save:
      shap_bar_<ModelName>.png        — mean |SHAP| feature importance bar
      shap_beeswarm_<ModelName>.png   — beeswarm summary plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    models, x_train, x_test, y_test, features = _train_models(df)

    # Sample explanation set (same for all models for comparability)
    rng = np.random.default_rng(42)
    explain_idx = rng.choice(len(x_test), size=min(N_EXPLAIN, len(x_test)), replace=False)
    x_explain = x_test.iloc[explain_idx]

    background_idx = rng.choice(len(x_train), size=min(N_BACKGROUND, len(x_train)), replace=False)
    x_background = x_train.iloc[background_idx]

    for model_name, model in models.items():
        print(f"  Computing SHAP for {model_name}...")

        if model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, x_background)
            shap_vals = explainer(x_explain)

        elif model_name == "RandomForest":
            explainer = shap.TreeExplainer(model)
            # New API returns Explanation with shape (n_samples, n_features, n_classes)
            full_explanation = explainer(x_explain)
            # Slice class 1 (Selected = 1)
            shap_vals = full_explanation[:, :, 1]

        else:  # NeuralNet
            explainer = shap.PermutationExplainer(
                lambda x: model.predict_proba(x)[:, 1],
                x_background,
            )
            raw = explainer(x_explain)
            shap_vals = shap.Explanation(
                values=raw.values,
                base_values=raw.base_values,
                data=x_explain.values,
                feature_names=features,
            )

        _shap_bar(shap_vals, model_name, output_dir)
        _shap_beeswarm(shap_vals, model_name, output_dir)
        print(f"  Saved shap_bar_{model_name}.png and shap_beeswarm_{model_name}.png")
