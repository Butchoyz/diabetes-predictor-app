"""
Rebuild ``X_train_imputed.csv`` - the imputed training set (X_train + y_train) that the app
samples its 100 SHAP reference patients from, exactly like the thesis notebook.

Steps (same as the notebook)
----------------------------
1. Load the ORIGINAL Pima Indians Diabetes dataset (768 rows, raw, zeros intact).
   Note: ``Pimadiabetes.csv`` in this repo is a modified copy (zeros already replaced with
   class-wise medians), so it can't be used here.
2. Split it the same way as the notebook:
       train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   Verified below: X_test must equal ``ProposeX_test_raw.csv`` row for row, which proves
   X_train is the notebook's X_train, in the same order.
3. Impute zeros in X_train with the saved training medians (``imputation_medians.joblib``),
   the same imputation the app applies at prediction time.
4. Save X_train (imputed) with y_train as the ``Outcome`` column, keeping the row order.

The app then runs the notebook's sampling on this file at start-up:
       train_test_split(X_train, y_train, train_size=100, stratify=y_train, random_state=42)

Usage
-----
    python build_reference_data.py                     # downloads the original dataset
    python build_reference_data.py path/to/diabetes.csv
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ORIGINAL_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
TEST_PATH = "ProposeX_test_raw.csv"
MEDIANS_PATH = "imputation_medians.joblib"
FEATURES_PATH = "feature_names.joblib"
OUTPUT_PATH = "X_train_imputed.csv"


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else ORIGINAL_URL
    features = list(joblib.load(FEATURES_PATH))
    medians = joblib.load(MEDIANS_PATH)

    data = pd.read_csv(source)
    assert len(data) == 768, f"Expected the 768-row original dataset, got {len(data)} rows."
    X, y = data[features], data["Outcome"]

    # Same split as the notebook.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Proof that this is the notebook's split: the test set matches the saved one row for row.
    saved_test = pd.read_csv(TEST_PATH)[features]
    assert np.allclose(X_test.values.astype(float), saved_test.values.astype(float)), \
        "X_test does not match ProposeX_test_raw.csv - this is not the notebook's split."

    # Sanity check: X_train must reproduce the saved training medians.
    for col, saved in medians.items():
        recon = X_train[col].replace(0, np.nan).median()
        assert np.isclose(recon, saved), f"{col}: X_train median {recon} != saved {saved}"

    # Same imputation as the notebook / the app.
    X_train = X_train.copy()
    for col, median_val in medians.items():
        X_train[col] = X_train[col].replace(0, np.nan).fillna(median_val)

    out = X_train.assign(Outcome=y_train.values).reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"X_test matches {TEST_PATH} row for row ({len(X_test)} patients).")
    print(f"Saved {OUTPUT_PATH}: {len(out)} patients "
          f"({int(out['Outcome'].sum())} diabetic, {int((1 - out['Outcome']).sum())} non-diabetic)")


if __name__ == "__main__":
    main()
