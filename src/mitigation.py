from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


SEEDS = [42, 43, 44, 45, 46]


def build_models(seed: int) -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=seed),
    }


def demographic_parity_difference(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    rates = []
    for group_value in sorted(pd.Series(sensitive).unique()):
        mask = sensitive == group_value
        rates.append(float(np.mean(y_pred[mask])))
    return float(max(rates) - min(rates))


def equalized_odds_difference(y_true: pd.Series, y_pred: np.ndarray, sensitive: pd.Series) -> float:
    true_positive_rates = []
    false_positive_rates = []
    for group_value in sorted(pd.Series(sensitive).unique()):
        mask = sensitive == group_value
        y_true_group = np.asarray(y_true[mask])
        y_pred_group = np.asarray(y_pred[mask])
        positives = max(int((y_true_group == 1).sum()), 1)
        negatives = max(int((y_true_group == 0).sum()), 1)
        true_positive_rates.append(float(((y_pred_group == 1) & (y_true_group == 1)).sum() / positives))
        false_positive_rates.append(float(((y_pred_group == 1) & (y_true_group == 0)).sum() / negatives))
    return float(
        max(
            max(true_positive_rates) - min(true_positive_rates),
            max(false_positive_rates) - min(false_positive_rates),
        )
    )


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, sensitive: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "DPD": demographic_parity_difference(y_pred, sensitive),
        "EOD": equalized_odds_difference(y_true, y_pred, sensitive),
    }


def optimize_group_thresholds(
    scores: np.ndarray,
    y_true: pd.Series,
    sensitive: pd.Series,
    grid: np.ndarray,
) -> dict[int, float]:
    groups = sorted(pd.Series(sensitive).unique().tolist())
    sensitive_array = np.asarray(sensitive)

    best_score = float("inf")
    best_thresholds = {int(group_value): 0.5 for group_value in groups}

    for thresholds in product(grid, repeat=len(groups)):
        threshold_map = {int(group_value): float(threshold) for group_value, threshold in zip(groups, thresholds)}
        threshold_vector = np.vectorize(lambda value: threshold_map[int(value)], otypes=[float])(sensitive_array)
        y_pred = (scores >= threshold_vector).astype(int)

        metrics = compute_metrics(y_true, y_pred, sensitive)
        objective = metrics["EOD"] + 0.5 * metrics["DPD"] - 0.01 * metrics["accuracy"]
        if objective < best_score:
            best_score = objective
            best_thresholds = threshold_map

    return best_thresholds


def apply_group_thresholds(
    scores: np.ndarray,
    sensitive: pd.Series,
    threshold_map: dict[int, float],
) -> np.ndarray:
    sensitive_array = np.asarray(sensitive)
    threshold_vector = np.vectorize(lambda value: threshold_map[int(value)], otypes=[float])(sensitive_array)
    return (scores >= threshold_vector).astype(int)


def run_once(df: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    features = [column for column in df.columns if column not in ["Candidate_ID", "Selected"]]
    strat_key = df[["Selected", "Gender"]].astype(str).agg("_".join, axis=1)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=strat_key
    )
    x_train, x_test = train_df[features], test_df[features]
    y_train, y_test = train_df["Selected"], test_df["Selected"]
    sensitive_train, sensitive_test = train_df["Gender"], test_df["Gender"]

    rows: list[dict[str, object]] = []
    for model_name, model in build_models(seed).items():
        model.fit(x_train, y_train)
        train_scores = model.predict_proba(x_train)[:, 1]
        test_scores = model.predict_proba(x_test)[:, 1]

        baseline_pred = (test_scores >= 0.5).astype(int)
        baseline_metrics = compute_metrics(y_test, baseline_pred, sensitive_test)
        rows.append({"seed": seed, "model": model_name, "variant": "base", **baseline_metrics})

        threshold_grid = np.linspace(0.1, 0.9, 9)
        best_thresholds = optimize_group_thresholds(
            train_scores,
            y_train,
            sensitive_train,
            threshold_grid,
        )
        mitigated_pred = apply_group_thresholds(test_scores, sensitive_test, best_thresholds)
        mitigated_metrics = compute_metrics(y_test, mitigated_pred, sensitive_test)
        rows.append(
            {
                "seed": seed,
                "model": model_name,
                "variant": "mitigated",
                **mitigated_metrics,
            }
        )

    return rows


def paired_tests(df_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in sorted(df_runs["model"].unique()):
        model_runs = df_runs[df_runs["model"] == model_name]
        baseline = model_runs[model_runs["variant"] == "base"].sort_values("seed")
        mitigated = model_runs[model_runs["variant"] == "mitigated"].sort_values("seed")
        for metric in ["accuracy", "DPD", "EOD"]:
            t_stat, p_value = ttest_rel(baseline[metric].to_numpy(), mitigated[metric].to_numpy())
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "base_mean": float(baseline[metric].mean()),
                    "mitigated_mean": float(mitigated[metric].mean()),
                    "mean_diff_mitigated_minus_base": float(
                        mitigated[metric].mean() - baseline[metric].mean()
                    ),
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def run_mitigation(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        run_rows.extend(run_once(df, seed))

    per_run = pd.DataFrame(run_rows).sort_values(["model", "variant", "seed"])
    summary = (
        per_run.groupby(["model", "variant"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            DPD_mean=("DPD", "mean"),
            DPD_std=("DPD", "std"),
            EOD_mean=("EOD", "mean"),
            EOD_std=("EOD", "std"),
        )
        .sort_values(["model", "variant"])
    )
    ttests = paired_tests(per_run).sort_values(["model", "metric"])

    per_run.to_csv(output_dir / "per_run_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_stats.csv", index=False)
    ttests.to_csv(output_dir / "paired_ttests.csv", index=False)
    return per_run, summary, ttests
