"""
Inside the Stacking Ensemble: what its meta-learner received and returned for one patient, and an
exact SHAP explanation of the final probability in terms of the 7 base learners.

The saved model (``diabetes_model.joblib``) is a ``CalibratedClassifierCV(cv=7)``. It holds 7
calibrated copies of the SMOTE + stacking pipeline, one per calibration fold, and its probability
is the average of the copies:

    copy k   level 0       p_kj = P(diabetic) from base learner j (RF, GBM, XGB, LGB, KNN, SVM, NB)
             level 1       z_k  = b_k + Σ_j w_kj · p_kj   Bagged LR meta-learner: the average
                                                          log-odds of its 15 logistic regressions
             calibration   P_k  = 1 / (1 + exp(A_k · z_k + B_k))   sigmoid (Platt) scaling
    final                  P    = (1/7) Σ_k P_k           = the probability on the result card

The meta-learner SHAP treats the 7 base learners as the players. f(S) is the final probability,
averaged over the 7 copies and the 100 reference patients, when the base learners in S give this
patient's predictions and the others give the reference patient's predictions (each copy with its
own base learners). So f(∅) is the Stacking Ensemble's base risk and f(F) its prediction: the same
two numbers as in its feature-level SHAP.

Note: the thesis notebook's meta-learner summary plot explains ``calibrated_model.estimator``, an
uncalibrated pipeline refit on the full training set that is never used for predictions. This
module explains the 7 calibrated copies, so its numbers add up to the probability the app shows.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from predictor import get_proposed_predictor
from shapley import coalition_values, shapley_values

FEATURES = list(joblib.load("feature_names.joblib"))

BASE_LEARNER_LABELS = {
    "rf": "Random Forest",
    "gbm": "Gradient Boosting",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Machine",
    "nb": "Naive Bayes",
}


def calibrated_copies():
    return get_proposed_predictor().model.calibrated_classifiers_


def base_learner_names() -> list:
    return list(_stack(calibrated_copies()[0]).named_estimators_)


def _stack(copy):
    return copy.estimator.named_steps["stack"]


def _base_predictions(copy, X: pd.DataFrame) -> np.ndarray:
    """Level 0 of one copy: P(diabetic) from each base learner (SMOTE only acts during training)."""
    return _stack(copy).transform(X)


def _final_probability(copy, base_preds: np.ndarray) -> np.ndarray:
    """Level 1 + calibration of one copy, from its base learners' predictions."""
    z = _stack(copy).final_estimator_.decision_function(base_preds)
    return copy.calibrators[0].predict(z)


def _meta_weights(bagging):
    """The Bagged LR as one linear score: the average intercept and weights of its logistic regressions."""
    weights, intercept = np.zeros(bagging.n_features_in_), 0.0
    for lr, inputs in zip(bagging.estimators_, bagging.estimators_features_):
        weights[inputs] += lr.coef_[0] / len(bagging.estimators_)
        intercept += lr.intercept_[0] / len(bagging.estimators_)
    return weights, intercept


@st.cache_data(show_spinner=False, max_entries=64)
def stacking_internals(patient: tuple) -> dict:
    """Every number of the stacking chain for one (imputed) patient, copy by copy."""
    X = pd.DataFrame([patient], columns=FEATURES)
    copies = []
    for copy in calibrated_copies():
        base = _base_predictions(copy, X)[0]
        bagging = _stack(copy).final_estimator_
        weights, intercept = _meta_weights(bagging)
        z = float(bagging.decision_function(base.reshape(1, -1))[0])
        calibrator = copy.calibrators[0]
        copies.append({
            "base": base,
            "weights": weights,
            "intercept": float(intercept),
            "z": z,
            "meta_probability": float(bagging.predict_proba(base.reshape(1, -1))[0, 1]),
            "a": float(calibrator.a_),
            "b": float(calibrator.b_),
            "probability": float(calibrator.predict(np.array([z]))[0]),
        })
    return {
        "learners": base_learner_names(),
        "copies": copies,
        "probability": float(np.mean([c["probability"] for c in copies])),
    }


@st.cache_data(show_spinner=False, max_entries=64)
def explain_meta_learner(patient: tuple, _reference: pd.DataFrame) -> dict:
    """Exact SHAP values of the final probability over the 7 base learners (2^7 = 128 coalitions).

    ``_reference`` holds the reference patients (imputed); the leading underscore keeps it out of
    the cache key, since it never changes.
    """
    X = pd.DataFrame([patient], columns=FEATURES)
    copies = calibrated_copies()
    v, patient_preds, reference_preds = 0.0, 0.0, 0.0
    for copy in copies:
        base_x = _base_predictions(copy, X)[0]
        base_ref = _base_predictions(copy, _reference)
        v = v + coalition_values(lambda B, c=copy: _final_probability(c, B), base_x, base_ref)
        patient_preds = patient_preds + base_x
        reference_preds = reference_preds + base_ref.mean(axis=0)
    v = v / len(copies)

    learners = base_learner_names()
    phi = shapley_values(v)
    return {
        "base_value": float(v[0]),
        "shap_values": {name: float(p) for name, p in zip(learners, phi)},
        "probability": float(v[-1]),
        "coalitions": v,
        # Base learners' predictions, averaged over the 7 copies (and over the reference patients).
        "patient_predictions": dict(zip(learners, (patient_preds / len(copies)).tolist())),
        "reference_predictions": dict(zip(learners, (reference_preds / len(copies)).tolist())),
    }
