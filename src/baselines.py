from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Top TARGET_POS_RATE fraction of calibration scores sets the decision threshold.
# Calibration uses a held-out validation split so threshold transfer to test is accurate.
TARGET_POS_RATE = 0.25


def _di(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    rates = [float(np.mean(y_pred[sensitive == g])) for g in sorted(sensitive.unique())]
    return float(min(rates) / max(rates)) if max(rates) > 0 else float("nan")


def _dpd(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    rates = [float(np.mean(y_pred[sensitive == g])) for g in sorted(sensitive.unique())]
    return float(max(rates) - min(rates))


def _eod(y_true: pd.Series, y_pred: np.ndarray, sensitive: pd.Series) -> float:
    tprs, fprs = [], []
    for g in sorted(sensitive.unique()):
        mask = sensitive == g
        yt = np.asarray(y_true[mask])
        yp = y_pred[mask]
        tprs.append(float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1)))
        fprs.append(float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1)))
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))


def group_metrics(y_true: pd.Series, y_pred: np.ndarray, group: pd.Series) -> pd.DataFrame:
    rows: list[dict] = []
    for g in sorted(group.unique()):
        mask = group == g
        yp = y_pred[mask]
        yt = y_true[mask]
        rows.append(
            {
                "group": int(g),
                "n": int(mask.sum()),
                "positive_rate": float(np.mean(yp)),
                "tpr": float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1)),
                "fpr": float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1)),
            }
        )
    return pd.DataFrame(rows)


def run_baselines(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    features = [c for c in df.columns if c not in ["Candidate_ID", "Selected", "Race_label"]]
    strat_key = df[["Selected", "Gender"]].astype(str).agg("_".join, axis=1)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat_key
    )

    # Hold out 25% of training for threshold calibration (prevents training-set inflation).
    fit_df, cal_df = train_test_split(
        train_df, test_size=0.25, random_state=random_state + 1, stratify=train_df["Selected"]
    )

    x_fit, x_cal = fit_df[features], cal_df[features]
    y_fit, y_cal = fit_df["Selected"], cal_df["Selected"]
    x_train, x_test = train_df[features], test_df[features]
    y_train, y_test = train_df["Selected"], test_df["Selected"]
    y_test_r = y_test.reset_index(drop=True)
    sensitive_test = test_df["Gender"].reset_index(drop=True)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=random_state,
            class_weight="balanced_subsample", oob_score=True,
        ),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=random_state),
    }

    sw_fit = compute_sample_weight("balanced", y_fit)
    sw_train = compute_sample_weight("balanced", y_train)
    summary_rows: list[dict] = []
    group_rows: list[pd.DataFrame] = []
    race_test = test_df["Race_label"].reset_index(drop=True)
    age_group_test = test_df["Age_Group"].reset_index(drop=True)
    raw_preds: dict = {
        "y_test": y_test_r,
        "sensitive": sensitive_test,
        "race_test": race_test,
        "age_group_test": age_group_test,
        "models": {},
    }

    for model_name, model in models.items():
        if model_name == "RandomForest":
            # RF: fit on full training and use OOB for calibration (no data wasted).
            model.fit(x_train, y_train)
            cal_proba = model.oob_decision_function_[:, 1]
        elif model_name == "NeuralNet":
            model.fit(x_fit, y_fit, sample_weight=sw_fit)
            cal_proba = model.predict_proba(x_cal)[:, 1]
            model.fit(x_train, y_train, sample_weight=sw_train)
        else:
            model.fit(x_fit, y_fit)
            cal_proba = model.predict_proba(x_cal)[:, 1]
            model.fit(x_train, y_train)

        test_proba = model.predict_proba(x_test)[:, 1]
        threshold = float(np.quantile(cal_proba, 1.0 - TARGET_POS_RATE))
        y_pred = (test_proba >= threshold).astype(int)

        summary_rows.append(
            {
                "model": model_name,
                "threshold": round(threshold, 4),
                "accuracy":  float(accuracy_score(y_test_r, y_pred)),
                "precision": float(precision_score(y_test_r, y_pred, zero_division=0)),
                "recall":    float(recall_score(y_test_r, y_pred, zero_division=0)),
                "f1":        float(f1_score(y_test_r, y_pred, zero_division=0)),
                "roc_auc":   float(roc_auc_score(y_test_r, test_proba)),
                "DI": _di(y_pred, sensitive_test),
                "DPD": _dpd(y_pred, sensitive_test),
                "EOD": _eod(y_test_r, y_pred, sensitive_test),
            }
        )

        raw_preds["models"][model_name] = y_pred

        group_df = group_metrics(y_test_r, y_pred, sensitive_test)
        group_df.insert(0, "model", model_name)
        group_rows.append(group_df)

    baseline_summary = pd.DataFrame(summary_rows).sort_values("model")
    baseline_groups = pd.concat(group_rows, ignore_index=True).sort_values(["model", "group"])

    baseline_summary.to_csv(output_dir / "baseline_metrics.csv", index=False)
    baseline_groups.to_csv(output_dir / "baseline_group_metrics.csv", index=False)
    return baseline_summary, baseline_groups, raw_preds
