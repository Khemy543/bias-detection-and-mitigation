from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def group_metrics(y_true: pd.Series, y_pred: np.ndarray, group: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for group_value in sorted(group.unique()):
        mask = group == group_value
        y_pred_group = y_pred[mask]
        y_true_group = y_true[mask]
        rows.append(
            {
                "group": int(group_value),
                "n": int(mask.sum()),
                "positive_rate": float(np.mean(y_pred_group)),
                "tpr": float(
                    ((y_pred_group == 1) & (y_true_group == 1)).sum()
                    / max((y_true_group == 1).sum(), 1)
                ),
                "fpr": float(
                    ((y_pred_group == 1) & (y_true_group == 0)).sum()
                    / max((y_true_group == 0).sum(), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_baselines(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    features = [column for column in df.columns if column not in ["Candidate_ID", "Selected"]]
    strat_key = df[["Selected", "Gender"]].astype(str).agg("_".join, axis=1)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat_key
    )

    x_train, x_test = train_df[features], test_df[features]
    y_train, y_test = train_df["Selected"], test_df["Selected"]
    sensitive_test = test_df["Gender"]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "NeuralNet": MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=500,
            random_state=random_state,
        ),
    }

    summary_rows: list[dict[str, float | str]] = []
    group_rows: list[pd.DataFrame] = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

        summary_rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else float("nan"),
            }
        )

        group_df = group_metrics(
            y_test.reset_index(drop=True),
            np.asarray(y_pred),
            sensitive_test.reset_index(drop=True),
        )
        group_df.insert(0, "model", model_name)
        group_rows.append(group_df)

    baseline_summary = pd.DataFrame(summary_rows).sort_values("model")
    baseline_groups = pd.concat(group_rows, ignore_index=True).sort_values(["model", "group"])

    baseline_summary.to_csv(output_dir / "baseline_metrics.csv", index=False)
    baseline_groups.to_csv(output_dir / "baseline_group_metrics.csv", index=False)
    return baseline_summary, baseline_groups
