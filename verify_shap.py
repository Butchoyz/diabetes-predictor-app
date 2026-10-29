"""
Check that the SHAP numbers shown in the app are correct.

For each test patient (rows of ``ProposeX_test_raw.csv``, imputed like the app does):

1. Formula 7 computed by hand from the 256 coalition values (``shapley.py``) gives the same SHAP
   values as the shap library's Exact explainer over the same 100 reference patients, for all
   9 models.
2. Formula 6 holds: base risk + sum of SHAP values = the model's predicted probability.
3. The meta-learner chain (7 base learners -> Bagged LR score -> sigmoid calibration -> average of
   the 7 copies) reproduces the Stacking Ensemble's probability, and the meta-learner SHAP values
   add up to it from the same base risk.

Usage
-----
    python verify_shap.py        # first 3 test patients
    python verify_shap.py 10     # first 10 test patients
"""
import sys
import warnings

import numpy as np
import pandas as pd
import shap
from streamlit.logger import set_log_level

set_log_level("error")  # the app's cached functions run here without a Streamlit app, which only warns
warnings.filterwarnings("ignore")

from meta_learner import explain_meta_learner, stacking_internals
from shap_analysis import (FEATURES, PROPOSED_MODEL_NAME, all_model_names, explain_patient, get_predictor,
                           load_reference_patients, patient_tuple)

TOLERANCE = 1e-9


def check(label, ok, detail):
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def main():
    n_patients = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    test = pd.read_csv("ProposeX_test_raw.csv")[FEATURES].head(n_patients)
    background = load_reference_patients().values
    all_ok = True

    for row in range(len(test)):
        patient = patient_tuple(test.iloc[[row]].reset_index(drop=True))
        print(f"\nPatient {row + 1}: " + ", ".join(f"{f}={v:g}" for f, v in zip(FEATURES, patient)))

        for name in all_model_names():
            predictor = get_predictor(name)
            result = explain_patient(name, patient)
            ours = np.array([result["shap_values"][f] for f in FEATURES])

            def predict_fn(X):
                return predictor.predict_proba_batch(pd.DataFrame(X, columns=FEATURES))

            masker = shap.maskers.Independent(background, max_samples=len(background))
            library = shap.explainers.Exact(predict_fn, masker)(np.array([patient]), silent=True)
            prediction = predict_fn(np.array([patient]))[0]
            diff = np.abs(ours - library.values[0]).max()
            gap = abs(result["base_value"] + ours.sum() - prediction)
            all_ok &= check(name, diff < TOLERANCE and gap < TOLERANCE,
                            f"max |ours - shap library| = {diff:.1e}, |base + sum - prediction| = {gap:.1e}")

        internals = stacking_internals(patient)
        meta = explain_meta_learner(patient, load_reference_patients())
        prediction = get_predictor(PROPOSED_MODEL_NAME).predict_proba_batch(
            pd.DataFrame([patient], columns=FEATURES))[0]
        base_risk = explain_patient(PROPOSED_MODEL_NAME, patient)["base_value"]
        chain_gap = abs(internals["probability"] - prediction)
        meta_gap = abs(meta["base_value"] + sum(meta["shap_values"].values()) - prediction)
        all_ok &= check("Meta-learner chain", chain_gap < TOLERANCE,
                        f"|average of 7 calibrated copies - model probability| = {chain_gap:.1e}")
        all_ok &= check("Meta-learner SHAP", meta_gap < TOLERANCE and abs(meta["base_value"] - base_risk) < TOLERANCE,
                        f"|base + sum - prediction| = {meta_gap:.1e}, same base risk as the feature SHAP")

    print("\nAll checks passed." if all_ok else "\nSome checks FAILED.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
