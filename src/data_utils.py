from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_and_encode(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop pre-encoded columns written by load_dataset.py — we re-encode here
    df = df.drop(columns=["Gender_encoded", "Race_encoded", "Age_Group_encoded"], errors="ignore")

    # Gender: Female=0, Male=1 — used as sensitive attribute and feature
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    # Age_Group: natural ordinal ordering
    age_map = {"Under_30": 0, "30_to_45": 1, "45_to_60": 2, "Over_60": 3}
    df["Age_Group"] = df["Age_Group"].map(age_map)

    # Preserve Race label before one-hot expansion (used for fairness plots)
    df["Race_label"] = df["Race"]

    # Race: one-hot encode so models don't treat groups as ordered
    df = pd.get_dummies(df, columns=["Race"])

    bool_cols = df.select_dtypes("bool").columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    return df
