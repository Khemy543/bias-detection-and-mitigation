from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def load_and_encode(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[18, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65"],
    )
    df["Selected"] = df["Selected"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["Age"])

    skills_df = df.copy()
    skills_df["Skills"] = skills_df["Skills"].fillna("").str.lower().str.split(",")
    mlb = MultiLabelBinarizer()
    skills_encoded = pd.DataFrame(
        mlb.fit_transform(skills_df["Skills"]),
        columns=[f"skill_{skill.strip().replace(' ', '_')}" for skill in mlb.classes_],
        index=skills_df.index,
    )
    df = pd.concat([skills_df.drop(columns=["Skills"]), skills_encoded], axis=1)

    categorical_cols = [
        "Gender",
        "Race",
        "Education",
        "Age_Group",
        "Certifications",
        "Job_Role_Applied",
    ]
    for col in categorical_cols:
        df[col] = pd.factorize(df[col])[0]

    return df
