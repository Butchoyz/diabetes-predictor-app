"""
Rebuild ``shap_background.csv`` - the reference patients SHAP uses to explain every model.

How the background is built
---------------------------
1. Start from the ORIGINAL Pima Indians Diabetes dataset (768 rows, raw, zeros intact).
   Note: ``Pimadiabetes.csv`` in this repo is a modified copy (zeros already replaced with
   class-wise medians), so it can't be used to reconstruct the split.
2. Remove the 154 held-out test patients in ``ProposeX_test_raw.csv``. What remains is the
   exact 614-patient training split. This is verified below: its non-zero medians must match
   ``imputation_medians.joblib`` (Glucose 117, BloodPressure 72, SkinThickness 29,
   Insulin 125, BMI 32.4).
3. Apply the same train-median imputation the app applies at prediction time.
4. Take a class-stratified random sample of 100 patients (random_state=42). 100 is SHAP's
   default background size and keeps a live explanation of the stacking ensemble to a few seconds.

Every model (the proposed ensemble and all 8 baselines) is explained against this SAME
background, so their SHAP values are directly comparable.

Usage
-----
    python build_shap_background.py                     # downloads the original dataset
    python build_shap_background.py path/to/diabetes.csv
"""
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ORIGINAL_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
TEST_PATH = "ProposeX_test_raw.csv"
MEDIANS_PATH = "imputation_medians.joblib"
FEATURES_PATH = "feature_names.joblib"
OUTPUT_PATH = "shap_background.csv"
BACKGROUND_SIZE = 100
RANDOM_STATE = 42


def reconstruct_training_split(original: pd.DataFrame, test: pd.DataFrame, features: list) -> pd.DataFrame:
    """Return the original rows that are NOT in the test set (multiset difference)."""
    def row_keys(df):
        return df[features].astype(float).round(6).astype(str).agg("|".join, axis=1)

    test_counts = Counter(row_keys(test))
    original_keys = row_keys(original)

    missing = [k for k, n in test_counts.items() if (original_keys == k).sum() < n]
    if missing:
        raise ValueError(f"{len(missing)} test rows were not found in the original dataset.")

    used = Counter()
    keep = []
    for i, key in enumerate(original_keys):
        if used[key] < test_counts.get(key, 0):
            used[key] += 1          # this row belongs to the test split
        else:
            keep.append(i)
    return original.iloc[keep].reset_index(drop=True)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else ORIGINAL_URL
    features = list(joblib.load(FEATURES_PATH))
    medians = joblib.load(MEDIANS_PATH)

    original = pd.read_csv(source)
    test = pd.read_csv(TEST_PATH)
    assert len(original) == 768, f"Expected the 768-row original dataset, got {len(original)} rows."

    train = reconstruct_training_split(original, test, features)
    assert len(train) == len(original) - len(test), "Training split has the wrong size."

    # Sanity check: the reconstructed split must reproduce the saved training medians.
    for col, saved in medians.items():
        recon = train[col].replace(0, np.nan).median()
        assert np.isclose(recon, saved), f"{col}: reconstructed median {recon} != saved {saved}"

    # Same imputation the app applies before prediction.
    imputed = train.copy()
    for col, median_val in medians.items():
        imputed[col] = imputed[col].replace(0, np.nan).fillna(median_val)

    background, _ = train_test_split(
        imputed,
        train_size=BACKGROUND_SIZE,
        stratify=imputed["Outcome"],
        random_state=RANDOM_STATE,
    )
    background = background[features + ["Outcome"]].reset_index(drop=True)
    background.to_csv(OUTPUT_PATH, index=False)

    print(f"Training split: {len(train)} patients (diabetic rate {train['Outcome'].mean():.1%})")
    print(f"Saved {OUTPUT_PATH}: {len(background)} patients "
          f"({int(background['Outcome'].sum())} diabetic, {int((1 - background['Outcome']).sum())} non-diabetic)")


if __name__ == "__main__":
    main()
