"""
Adult Income Dataset Loader & Preprocessor
==========================================
Loads the UCI Adult Income dataset from the raw .data and .test files
and preprocesses it to match the structure of your existing recruitment
bias detection pipeline.

Files needed (place in same folder as this script):
    adult.data   — training split (32,561 rows)
    adult.test   — test split    (16,281 rows)

Output:
    adult_recruitment.csv — cleaned, combined dataset ready for your pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# ── 1. Column names (from adult.names) ────────────────────────────────────────
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

# ── 2. Load both splits ────────────────────────────────────────────────────────
def load_raw(data_path, test_path):
    train = pd.read_csv(
        data_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?"]
    )

    test = pd.read_csv(
        test_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?"],
        skiprows=1          # first row is a comment line in adult.test
    )

    # adult.test labels have a trailing dot e.g. "<=50K." — strip it
    test["income"] = test["income"].str.rstrip(".")

    # Combine into one dataset (you'll do your own train/test split later)
    df = pd.concat([train, test], ignore_index=True)
    print(f"Loaded {len(df):,} rows ({len(train):,} train + {len(test):,} test)")
    return df


# ── 3. Clean ──────────────────────────────────────────────────────────────────
def clean(df):
    before = len(df)
    df = df.dropna()          # drop ~3,600 rows with missing workclass/occupation
    print(f"Dropped {before - len(df):,} rows with missing values → {len(df):,} remaining")
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


# ── 4. Map to your pipeline's column naming convention ────────────────────────
def map_columns(df):
    """
    Maps Adult dataset columns to names that match your existing
    recruitment pipeline as closely as possible.

    Sensitive attributes:
        Gender     ← sex
        Race       ← race  (grouped to 3 categories)
        Age_Group  ← age   (binned)

    Proxy / skill features:
        Education        ← education_num  (0–16 scale)
        Experience_Years ← hours_per_week (best available proxy)
        Certifications   ← occupation     (encoded)
        Screening_Score  ← capital_gain   (scaled 0–1, proxy for merit score)

    Target:
        Selected  ← income  (>50K = 1 = selected/high earner)
    """

    out = pd.DataFrame()

    # ── Sensitive attributes ──────────────────────────────────────────────────

    # Gender: Male / Female (binary in this dataset)
    out["Gender"] = df["sex"].map({"Male": "Male", "Female": "Female"})

    # Race: consolidate 5 categories → 3 to match your pipeline
    race_map = {
        "White":                "White",
        "Black":                "Black",
        "Asian-Pac-Islander":   "Asian",
        "Amer-Indian-Eskimo":   "Other",
        "Other":                "Other"
    }
    out["Race"] = df["race"].map(race_map)

    # Age_Group: bin continuous age into groups matching your pipeline
    out["Age_Group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["Under_30", "30_to_45", "45_to_60", "Over_60"]
    ).astype(str)

    # ── Skill / proxy features ────────────────────────────────────────────────

    # Education: 0–16 numeric scale (already numeric, normalize to 0–1)
    out["Education"] = (df["education_num"] / 16).round(4)

    # Experience_Years: hours_per_week as proxy (normalize to 0–1)
    out["Experience_Years"] = (
        (df["hours_per_week"] - df["hours_per_week"].min()) /
        (df["hours_per_week"].max() - df["hours_per_week"].min())
    ).round(4)

    # Certifications: encode occupation into 0–1 skill tiers
    occupation_tiers = {
        "Prof-specialty":     1.0,
        "Exec-managerial":    0.9,
        "Tech-support":       0.8,
        "Protective-serv":    0.7,
        "Sales":              0.6,
        "Craft-repair":       0.5,
        "Adm-clerical":       0.5,
        "Transport-moving":   0.4,
        "Farming-fishing":    0.3,
        "Machine-op-inspct":  0.3,
        "Handlers-cleaners":  0.2,
        "Other-service":      0.2,
        "Priv-house-serv":    0.1,
        "Armed-Forces":       0.6,
    }
    out["Certifications"] = df["occupation"].map(occupation_tiers).fillna(0.5)

    # Screening_Score: capital_gain as merit proxy (normalize to 0–1)
    max_gain = df["capital_gain"].max()
    out["Screening_Score"] = (df["capital_gain"] / max_gain).round(4)

    # ── Target ───────────────────────────────────────────────────────────────
    # >50K = 1 (Selected / high earner), <=50K = 0
    out["Selected"] = (df["income"] == ">50K").astype(int)

    return out


# ── 5. Encode for modelling ───────────────────────────────────────────────────
def encode(df):
    """Label-encode categorical sensitive attributes (same as your pipeline)."""
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    for col in ["Gender", "Race", "Age_Group"]:
        df[col + "_encoded"] = le.fit_transform(df[col])

    return df


# ── 6. Summary ────────────────────────────────────────────────────────────────
def print_summary(df):
    print("\n── Dataset Summary ──────────────────────────────────────")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution (Selected):")
    vc = df["Selected"].value_counts()
    print(f"  Not Selected (0): {vc[0]:,}  ({vc[0]/len(df)*100:.1f}%)")
    print(f"  Selected     (1): {vc[1]:,}  ({vc[1]/len(df)*100:.1f}%)")

    print(f"\nGender distribution:")
    print(df["Gender"].value_counts().to_string())

    print(f"\nRace distribution:")
    print(df["Race"].value_counts().to_string())

    print(f"\nAge_Group distribution:")
    print(df["Age_Group"].value_counts().to_string())

    print(f"\nSelection rate by Gender:")
    print(df.groupby("Gender")["Selected"].mean().round(3).to_string())

    print(f"\nSelection rate by Race:")
    print(df.groupby("Race")["Selected"].mean().round(3).to_string())

    print(f"\nColumns: {list(df.columns)}")
    print("─────────────────────────────────────────────────────────\n")


# ── 7. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Update these paths to wherever your files are
    DATA_PATH = ROOT_DIR / "dataset" / "adult" / "adult.data"
    TEST_PATH = ROOT_DIR / "dataset" / "adult" / "adult.test"
    OUT_PATH  = ROOT_DIR / "dataset" /  "adult_recruitment.csv"

    df_raw  = load_raw(DATA_PATH, TEST_PATH)
    df_clean = clean(df_raw)
    df_mapped = map_columns(df_clean)
    df_final  = encode(df_mapped)

    print_summary(df_final)

    df_final.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")
    print("You can now load this file in your pipeline exactly like recruitment.csv")