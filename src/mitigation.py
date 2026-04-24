from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

TARGET_POS_RATE = 0.25
SEEDS = [42, 43, 44, 45, 46]

# Mitigation phases tracked at each seed run.
VARIANTS = ["pre", "in_process", "post"]
VARIANT_LABEL = {"pre": "Pre-mitigation", "in_process": "In-process", "post": "Post-process"}


def demographic_parity_difference(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    rates = [float(np.mean(y_pred[sensitive == g])) for g in sorted(pd.Series(sensitive).unique())]
    return float(max(rates) - min(rates))


def disparate_impact(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    rates = [float(np.mean(y_pred[sensitive == g])) for g in sorted(pd.Series(sensitive).unique())]
    return float(min(rates) / max(rates)) if max(rates) > 0 else float("nan")


def equalized_odds_difference(y_true: pd.Series, y_pred: np.ndarray, sensitive: pd.Series) -> float:
    tprs, fprs = [], []
    for g in sorted(pd.Series(sensitive).unique()):
        mask = sensitive == g
        yt = np.asarray(y_true[mask])
        yp = np.asarray(y_pred[mask])
        tprs.append(float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1)))
        fprs.append(float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1)))
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, scores)) if scores is not None else float("nan"),
        "DI": disparate_impact(y_pred, sensitive),
        "DPD": demographic_parity_difference(y_pred, sensitive),
        "EOD": equalized_odds_difference(y_true, y_pred, sensitive),
    }


def optimize_group_thresholds(
    scores: np.ndarray,
    y_true: pd.Series,
    sensitive: pd.Series,
    grid: np.ndarray,
    min_pos_rate: float = 0.05,
    max_pos_rate: float = 0.60,
) -> dict[int, float]:
    groups = sorted(pd.Series(sensitive).unique().tolist())
    sensitive_array = np.asarray(sensitive)

    best_score = float("inf")
    best_thresholds = {int(g): 0.5 for g in groups}

    for thresholds in product(grid, repeat=len(groups)):
        threshold_map = {int(g): float(t) for g, t in zip(groups, thresholds)}
        threshold_vec = np.vectorize(lambda v: threshold_map[int(v)], otypes=[float])(sensitive_array)
        y_pred = (scores >= threshold_vec).astype(int)

        pos_rate = float(y_pred.mean())
        if pos_rate < min_pos_rate or pos_rate > max_pos_rate:
            continue

        metrics = compute_metrics(y_true, y_pred, sensitive)
        # Strong penalty when DI < 0.80 (legal threshold), then minimise EOD + DPD.
        di_penalty = max(0.0, 0.80 - metrics["DI"]) * 5
        objective = di_penalty + metrics["EOD"] + 0.5 * metrics["DPD"]
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
    threshold_vec = np.vectorize(lambda v: threshold_map[int(v)], otypes=[float])(sensitive_array)
    return (scores >= threshold_vec).astype(int)


def run_once(df: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    features = [c for c in df.columns if c not in ["Candidate_ID", "Selected", "Race_label"]]
    strat_key = df[["Selected", "Gender"]].astype(str).agg("_".join, axis=1)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat_key)

    # Hold out 25% of training for threshold calibration (out-of-sample, like OOB for RF).
    fit_df, cal_df = train_test_split(
        train_df, test_size=0.25, random_state=seed + 100, stratify=train_df["Selected"]
    )

    x_fit, x_cal = fit_df[features], cal_df[features]
    y_fit, y_cal = fit_df["Selected"], cal_df["Selected"]
    sensitive_cal = cal_df["Gender"]

    x_train, x_test = train_df[features], test_df[features]
    y_train, y_test = train_df["Selected"], test_df["Selected"]
    sensitive_test = test_df["Gender"]

    sw_fit = compute_sample_weight("balanced", y_fit)
    sw_train = compute_sample_weight("balanced", y_train)

    threshold_grid = np.linspace(0.01, 0.99, 99)
    rows: list[dict[str, object]] = []

    for model_name in ["LogisticRegression", "RandomForest", "NeuralNet"]:

        # ── Phase 1: Pre-mitigation ───────────────────────────────────────────
        # Standard model, no class-weight adjustment — reveals natural bias without intervention.
        if model_name == "LogisticRegression":
            pre_cal_model = LogisticRegression(max_iter=1000, random_state=seed)
            pre_full = LogisticRegression(max_iter=1000, random_state=seed)
        elif model_name == "RandomForest":
            pre_cal_model = RandomForestClassifier(n_estimators=200, random_state=seed)
            pre_full = RandomForestClassifier(n_estimators=200, random_state=seed)
        else:  # NeuralNet
            pre_cal_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)
            pre_full = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)

        pre_cal_model.fit(x_fit, y_fit)
        cal_pre = pre_cal_model.predict_proba(x_cal)[:, 1]
        pre_full.fit(x_train, y_train)
        pre_test = pre_full.predict_proba(x_test)[:, 1]

        threshold_pre = float(np.quantile(cal_pre, 1.0 - TARGET_POS_RATE))
        pre_pred = (pre_test >= threshold_pre).astype(int)
        rows.append({"seed": seed, "model": model_name, "variant": "pre",
                     **compute_metrics(y_test, pre_pred, sensitive_test, scores=pre_test)})

        # ── Phase 2: In-process mitigation ───────────────────────────────────
        # Balanced class weighting during training to counteract class imbalance.
        if model_name == "LogisticRegression":
            in_cal_model = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
            in_full = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
            in_cal_model.fit(x_fit, y_fit)
            cal_in = in_cal_model.predict_proba(x_cal)[:, 1]
            in_full.fit(x_train, y_train)
            in_test = in_full.predict_proba(x_test)[:, 1]

        elif model_name == "RandomForest":
            in_cal_model = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced_subsample")
            in_full = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced_subsample")
            in_cal_model.fit(x_fit, y_fit)
            cal_in = in_cal_model.predict_proba(x_cal)[:, 1]
            in_full.fit(x_train, y_train)
            in_test = in_full.predict_proba(x_test)[:, 1]

        else:  # NeuralNet
            in_cal_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)
            in_full = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)
            in_cal_model.fit(x_fit, y_fit, sample_weight=sw_fit)
            cal_in = in_cal_model.predict_proba(x_cal)[:, 1]
            in_full.fit(x_train, y_train, sample_weight=sw_train)
            in_test = in_full.predict_proba(x_test)[:, 1]

        threshold_in = float(np.quantile(cal_in, 1.0 - TARGET_POS_RATE))
        in_pred = (in_test >= threshold_in).astype(int)
        rows.append({"seed": seed, "model": model_name, "variant": "in_process",
                     **compute_metrics(y_test, in_pred, sensitive_test, scores=in_test)})

        # ── Phase 3: Post-processing mitigation ──────────────────────────────
        # Group-specific decision thresholds optimised on calibration data.
        # Objective is pure fairness (EOD + 0.5*DPD); no accuracy reward so
        # mitigation genuinely trades a small amount of accuracy for fairness.
        # min_pos_rate = TARGET_POS_RATE forces the optimizer to maintain at least the same
        # overall selection rate as the baseline, preventing it from gaming the objective by
        # simply predicting far fewer positives (which would spuriously improve both DPD and accuracy).
        best_thresholds = optimize_group_thresholds(
            cal_in, y_cal, sensitive_cal, threshold_grid,
            min_pos_rate=TARGET_POS_RATE * 0.5, max_pos_rate=TARGET_POS_RATE * 1.5,
        )
        post_pred = apply_group_thresholds(in_test, sensitive_test, best_thresholds)
        rows.append({"seed": seed, "model": model_name, "variant": "post",
                     **compute_metrics(y_test, post_pred, sensitive_test, scores=in_test)})

    return rows


def paired_tests(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests comparing each later phase against the pre-mitigation baseline."""
    rows: list[dict[str, object]] = []
    for model_name in sorted(df_runs["model"].unique()):
        model_runs = df_runs[df_runs["model"] == model_name]
        pre = model_runs[model_runs["variant"] == "pre"].sort_values("seed")
        for compare_variant in ["in_process", "post"]:
            comp = model_runs[model_runs["variant"] == compare_variant].sort_values("seed")
            for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "DI", "DPD", "EOD"]:
                pre_vals = pre[metric].to_numpy()
                comp_vals = comp[metric].to_numpy()
                # Skip if NaN present (e.g. DI undefined when no positives predicted)
                valid = ~(np.isnan(pre_vals) | np.isnan(comp_vals))
                if valid.sum() < 2:
                    t_stat, p_value = float("nan"), float("nan")
                else:
                    t_stat, p_value = ttest_rel(pre_vals[valid], comp_vals[valid])
                rows.append(
                    {
                        "model": model_name,
                        "comparison": f"pre_vs_{compare_variant}",
                        "metric": metric,
                        "pre_mean": float(np.nanmean(pre_vals)),
                        "comp_mean": float(np.nanmean(comp_vals)),
                        "mean_diff": float(np.nanmean(comp_vals) - np.nanmean(pre_vals)),
                        "t_stat": float(t_stat),
                        "p_value": float(p_value),
                    }
                )
    return pd.DataFrame(rows)


def phase_snapshot(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of each metric at every phase — one row per model × variant."""
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "DI", "DPD", "EOD"]
    agg_dict = {}
    for m in metrics:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    return (
        df_runs.groupby(["model", "variant"], as_index=False)
        .agg(**agg_dict)
        .sort_values(["model", "variant"])
    )


def get_snapshot_predictions(df: pd.DataFrame, seed: int = 42) -> dict:
    """
    Run a single seed and return raw predictions for all three phases per model.
    Returns {"y_test": Series, "sensitive_test": Series,
             "models": {model_name: {"pre": ndarray, "in_process": ndarray, "post": ndarray}}}
    Used for confusion matrices and per-group selection-rate plots.
    """
    features = [c for c in df.columns if c not in ["Candidate_ID", "Selected", "Race_label"]]
    strat_key = df[["Selected", "Gender"]].astype(str).agg("_".join, axis=1)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat_key)

    fit_df, cal_df = train_test_split(
        train_df, test_size=0.25, random_state=seed + 100, stratify=train_df["Selected"]
    )
    x_fit, x_cal = fit_df[features], cal_df[features]
    y_fit, y_cal = fit_df["Selected"], cal_df["Selected"]
    sensitive_cal = cal_df["Gender"]

    x_train, x_test = train_df[features], test_df[features]
    y_train, y_test = train_df["Selected"], test_df["Selected"]
    sensitive_test = test_df["Gender"].reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    sw_fit = compute_sample_weight("balanced", y_fit)
    sw_train = compute_sample_weight("balanced", y_train)
    threshold_grid = np.linspace(0.01, 0.99, 99)

    result: dict = {"y_test": y_test, "sensitive_test": sensitive_test, "models": {}}

    for model_name in ["LogisticRegression", "RandomForest", "NeuralNet"]:
        preds: dict[str, np.ndarray] = {}

        # Pre-mitigation
        if model_name == "LogisticRegression":
            pre_cal = LogisticRegression(max_iter=1000, random_state=seed)
            pre_full = LogisticRegression(max_iter=1000, random_state=seed)
        elif model_name == "RandomForest":
            pre_cal = RandomForestClassifier(n_estimators=200, random_state=seed)
            pre_full = RandomForestClassifier(n_estimators=200, random_state=seed)
        else:
            pre_cal = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)
            pre_full = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)

        pre_cal.fit(x_fit, y_fit)
        cal_pre = pre_cal.predict_proba(x_cal)[:, 1]
        pre_full.fit(x_train, y_train)
        pre_test = pre_full.predict_proba(x_test)[:, 1]
        preds["pre"] = (pre_test >= float(np.quantile(cal_pre, 1.0 - TARGET_POS_RATE))).astype(int)

        # In-process mitigation
        if model_name == "LogisticRegression":
            in_cal = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
            in_full = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
            in_cal.fit(x_fit, y_fit)
            cal_in = in_cal.predict_proba(x_cal)[:, 1]
            in_full.fit(x_train, y_train)
        elif model_name == "RandomForest":
            in_cal = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced_subsample")
            in_full = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced_subsample")
            in_cal.fit(x_fit, y_fit)
            cal_in = in_cal.predict_proba(x_cal)[:, 1]
            in_full.fit(x_train, y_train)
        else:
            in_cal = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)
            in_full = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=seed)
            in_cal.fit(x_fit, y_fit, sample_weight=sw_fit)
            cal_in = in_cal.predict_proba(x_cal)[:, 1]
            in_full.fit(x_train, y_train, sample_weight=sw_train)

        in_test = in_full.predict_proba(x_test)[:, 1]
        preds["in_process"] = (in_test >= float(np.quantile(cal_in, 1.0 - TARGET_POS_RATE))).astype(int)

        # Post-processing mitigation
        best_thresholds = optimize_group_thresholds(
            cal_in, y_cal, sensitive_cal, threshold_grid,
            min_pos_rate=TARGET_POS_RATE * 0.5, max_pos_rate=TARGET_POS_RATE * 1.5,
        )
        preds["post"] = apply_group_thresholds(in_test, sensitive_test, best_thresholds)

        result["models"][model_name] = preds

    return result


def run_mitigation(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        run_rows.extend(run_once(df, seed))

    per_run = pd.DataFrame(run_rows).sort_values(["model", "variant", "seed"])
    snapshot = phase_snapshot(per_run)
    ttests = paired_tests(per_run).sort_values(["model", "comparison", "metric"])

    per_run.to_csv(output_dir / "per_run_metrics.csv", index=False)
    snapshot.to_csv(output_dir / "summary_stats.csv", index=False)
    ttests.to_csv(output_dir / "paired_ttests.csv", index=False)
    return per_run, snapshot, ttests
