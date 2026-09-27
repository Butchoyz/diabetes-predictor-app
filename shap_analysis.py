"""
SHAP (SHapley Additive exPlanations) for every model in the app.

The same method is applied to all 9 models, so their feature contributions can be compared
directly:

* Explainer   - ``shap.explainers.Exact``: exact Shapley values over all 2^8 = 256 feature
                coalitions. No sampling noise, so results are fully reproducible.
* Output      - the predicted probability of diabetes, P(diabetic): the number shown on each
                result card (not log-odds). Contributions are in probability units and add up
                exactly:   base value + sum(SHAP values) = predicted probability.
* Features    - the 8 clinical features AFTER median imputation and BEFORE scaling. Baseline
                models are wrapped together with their StandardScaler, so every model is
                explained in the same, human-readable units (mg/dL, kg/m², years, ...).
* Background  - 100 reference patients sampled from the imputed training set exactly like the
                thesis notebook: train_test_split(X_train, y_train, train_size=100,
                stratify=y_train, random_state=42), from ``X_train_imputed.csv``
                (see ``build_reference_data.py``). Each model's base risk is its average
                predicted probability over these patients (``compute_base_risk``).
* Proposed    - the Stacking Ensemble is explained end to end (SMOTE + stacking + sigmoid
  model       calibration), i.e. the final calibrated probability the user sees.
"""
import html

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.model_selection import train_test_split

from predictor import BASELINE_MODELS, get_baseline_predictor, get_proposed_predictor

PROPOSED_MODEL_NAME = "Stacking Ensemble Model"
TRAIN_DATA_PATH = "X_train_imputed.csv"  # notebook's imputed X_train, with y_train as "Outcome"
REFERENCE_SIZE = 100
RANDOM_STATE = 42
FEATURES = list(joblib.load("feature_names.joblib"))

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree",
    "Age": "Age",
}

# Diverging pair: red = pushes toward diabetic, blue = pushes toward non-diabetic.
COLOR_UP = "#e34948"
COLOR_DOWN = "#2a78d6"


# ============================================
# 1. COMPUTATION
# ============================================
def all_model_names():
    return [PROPOSED_MODEL_NAME] + list(BASELINE_MODELS.keys())


def get_predictor(model_name):
    if model_name == PROPOSED_MODEL_NAME:
        return get_proposed_predictor()
    return get_baseline_predictor(model_name)


@st.cache_resource(show_spinner=False)
def load_reference_sample():
    """Step 1 of the notebook: 100 stratified reference patients from the imputed X_train.

    Returns (raw imputed feature values, their outcomes).
    """
    train = pd.read_csv(TRAIN_DATA_PATH)
    X_train, y_train = train[FEATURES], train["Outcome"]
    sample_data_raw, _, sample_outcomes, _ = train_test_split(
        X_train, y_train, train_size=REFERENCE_SIZE, stratify=y_train, random_state=RANDOM_STATE
    )
    return sample_data_raw.astype(float), sample_outcomes


def load_reference_patients() -> pd.DataFrame:
    """The reference patients' feature values: used for the base risk and as the SHAP background."""
    return load_reference_sample()[0]


@st.cache_data(show_spinner=False)
def compute_base_risk() -> pd.DataFrame:
    """Steps 2-4 of the notebook: each model's average P(diabetic) over the reference patients.

    Raw imputed data for the Stacking Ensemble; data scaled with scaler.pkl for the baseline
    models. Each predictor's preprocess() applies exactly that (the baselines' scaler included).
    """
    reference = load_reference_patients()
    results = []
    for name in all_model_names():
        try:
            preds = get_predictor(name).predict_proba_batch(reference)
            base_risk = float(np.mean(preds) * 100)
        except Exception as e:  # same as the notebook: report the error in the table
            base_risk = f"Error: {e}"
        results.append({
            "Model Name": name,
            "Input Data": "Raw (imputed)" if name == PROPOSED_MODEL_NAME else "Scaled (scaler.pkl)",
            "Base Risk (%)": base_risk,
        })
    return pd.DataFrame(results)


@st.cache_data(show_spinner=False, max_entries=256)
def explain_patient(model_name: str, patient: tuple) -> dict:
    """Exact SHAP values of P(diabetic) for one (already imputed) patient.

    ``patient`` holds the imputed feature values in FEATURES order (a tuple so it can be cached).
    """
    predictor = get_predictor(model_name)
    background = load_reference_patients()

    def predict_fn(X):
        return predictor.predict_proba_batch(pd.DataFrame(X, columns=FEATURES))

    # A fresh explainer per call: maskers keep internal state, so they aren't shared across sessions.
    masker = shap.maskers.Independent(background.values, max_samples=len(background))
    explainer = shap.explainers.Exact(predict_fn, masker, feature_names=FEATURES)
    explanation = explainer(np.array([patient], dtype=float), silent=True)

    values = explanation.values[0]
    base_value = float(explanation.base_values[0])
    return {
        "base_value": base_value,
        "shap_values": {f: float(v) for f, v in zip(FEATURES, values)},
        "probability": base_value + float(values.sum()),
        "threshold": float(predictor.threshold),
    }


def explain_all_models(input_df: pd.DataFrame, progress_callback=None) -> dict:
    """SHAP explanations for the proposed model and every baseline, keyed by model name."""
    imputed = get_proposed_predictor().impute(input_df)[FEATURES].astype(float)
    patient = tuple(imputed.iloc[0].tolist())

    results = {}
    names = all_model_names()
    for i, name in enumerate(names):
        if progress_callback:
            progress_callback(i, len(names), name)
        results[name] = explain_patient(name, patient)
    return results


# ============================================
# 2. FORMATTING HELPERS
# ============================================
def _fmt_value(v: float) -> str:
    return f"{v:.0f}" if float(v).is_integer() else f"{v:g}"


def _fmt_pts(v: float) -> str:
    pts = v * 100
    return "±0.0" if abs(pts) < 0.05 else f"{pts:+.1f}"


def _feature_label(feature, value, imputed_features, with_value=True):
    label = FEATURE_LABELS[feature]
    if not with_value:
        return label
    suffix = " (median)" if feature in imputed_features else ""
    return f"{label} = {_fmt_value(value)}{suffix}"


def _cell_background(v: float, vmax: float) -> str:
    if vmax <= 0 or abs(v) < 0.0005:
        return "transparent"
    alpha = 0.10 + 0.50 * min(abs(v) / vmax, 1.0)
    rgb = "227, 73, 72" if v > 0 else "42, 120, 214"
    return f"rgba({rgb}, {alpha:.2f})"


# ============================================
# 3. COMPARISON TABLE (ALL MODELS SIDE BY SIDE)
# ============================================
def _comparison_table_html(results, patient_values, imputed_features, feature_order):
    vmax = max(abs(v) for r in results.values() for v in r["shap_values"].values())

    header_cells = ""
    for f in feature_order:
        badge = ' <span class="shap-median">median</span>' if f in imputed_features else ""
        header_cells += (
            f'<th class="shap-num">{FEATURE_LABELS[f]}'
            f'<div class="shap-val">{_fmt_value(patient_values[f])}{badge}</div></th>'
        )

    rows = ""
    for name, r in results.items():
        is_proposed = name == PROPOSED_MODEL_NAME
        top_feature = max(r["shap_values"], key=lambda f: abs(r["shap_values"][f]))
        cells = ""
        for f in feature_order:
            v = r["shap_values"][f]
            weight = "800" if f == top_feature else "500"
            cells += (
                f'<td class="shap-num" style="background:{_cell_background(v, vmax)}; font-weight:{weight};" '
                f'title="{html.escape(FEATURE_LABELS[f])}: {_fmt_pts(v)} percentage points">{_fmt_pts(v)}</td>'
            )
        label = f"⭐ {name}" if is_proposed else name
        rows += (
            f'<tr class="{"shap-proposed" if is_proposed else ""}">'
            f'<td class="shap-model">{label}</td>'
            f"{cells}"
            f"</tr>"
        )

    return f"""
<div class="shap-wrapper">
<div class="shap-table-scroll">
<table class="shap-table">
<thead><tr>
<th>Model</th>
{header_cells}
</tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
<div class="shap-legend">
<span><span class="shap-swatch" style="background:{COLOR_UP};"></span>Pushed toward <b>diabetic</b></span>
<span><span class="shap-swatch" style="background:{COLOR_DOWN};"></span>Pushed toward <b>non-diabetic</b></span>
<span>Values in percentage points · <b>Bold</b> = the model's biggest driver · Columns ordered by average impact across all models</span>
</div>
</div>"""


# ============================================
# 4. FEATURE CONTRIBUTION CHART (ONE MODEL)
# ============================================
def _feature_chart(result, patient_values, imputed_features, max_abs):
    """Diverging bars: each feature's push on P(diabetic), in percentage points.

    ``max_abs`` is the largest |contribution| across ALL models, so every tab shares one scale.
    """
    shap_values = result["shap_values"]
    ordered = sorted(FEATURES, key=lambda f: abs(shap_values[f]), reverse=True)
    df = pd.DataFrame([{
        "label": _feature_label(f, patient_values[f], imputed_features),
        "pts": shap_values[f] * 100,
        "kind": "up" if shap_values[f] >= 0 else "down",
        "text": _fmt_pts(shap_values[f]),
    } for f in ordered])

    limit = max(max_abs * 100 * 1.2, 1.0)  # headroom for the value labels
    x_scale = alt.Scale(domain=[-limit, limit], nice=False)
    y = alt.Y("label:N", sort=df["label"].tolist(), title=None,
              axis=alt.Axis(labelLimit=260, labelFontSize=12, labelColor="#0f172a", ticks=False, domain=False,
                            minExtent=185))  # reserve room: Vega under-measures labels in the app's web font

    bars = alt.Chart(df).mark_bar(cornerRadius=3, height=18).encode(
        y=y,
        x=alt.X("pts:Q", scale=x_scale, title="Contribution to predicted risk (percentage points)",
                axis=alt.Axis(format=".0f", gridColor="#eef2f7", labelColor="#475569",
                              titleColor="#475569", domain=False)),
        color=alt.Color("kind:N", legend=None,
                        scale=alt.Scale(domain=["up", "down"], range=[COLOR_UP, COLOR_DOWN])),
        tooltip=[alt.Tooltip("label:N", title="Feature"), alt.Tooltip("text:N", title="Contribution (pts)")],
    )
    text_style = dict(fontSize=12, fontWeight=600, color="#0f172a")
    labels_up = alt.Chart(df).transform_filter("datum.pts >= 0").mark_text(align="left", dx=5, **text_style).encode(
        y=y, x=alt.X("pts:Q", scale=x_scale), text="text:N")
    labels_down = alt.Chart(df).transform_filter("datum.pts < 0").mark_text(align="right", dx=-5, **text_style).encode(
        y=y, x=alt.X("pts:Q", scale=x_scale), text="text:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#94a3b8", strokeWidth=1).encode(
        x=alt.X("x:Q", scale=x_scale))

    return (
        (zero + bars + labels_up + labels_down)
        .properties(height=34 * len(df))
        .configure_view(stroke=None)
    )


def _summary_markdown(result, patient_values, imputed_features):
    shap_values = result["shap_values"]
    up = sorted([f for f in FEATURES if shap_values[f] >= 0.0005], key=lambda f: -shap_values[f])[:3]
    down = sorted([f for f in FEATURES if shap_values[f] <= -0.0005], key=lambda f: shap_values[f])[:3]

    def describe(features):
        if not features:
            return "none"
        return ", ".join(
            f"{_feature_label(f, patient_values[f], imputed_features)} ({_fmt_pts(shap_values[f])} pts)"
            for f in features
        )

    return (
        f"🔺 **Raised risk most:** {describe(up)}  \n"
        f"🔻 **Lowered risk most:** {describe(down)}"
    )


# ============================================
# 5. STREAMLIT SECTION
# ============================================
SHAP_CSS = """
<style>
.shap-wrapper {
    width: 92vw; max-width: 1150px; position: relative; left: 50%; transform: translateX(-50%);
    z-index: 10; margin: 0.5rem 0 1.5rem;
}
.shap-table-scroll {
    overflow-x: auto; border-radius: 16px; background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(12px); border: 1px solid rgba(226, 232, 240, 0.9);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
}
.shap-table { width: 100%; min-width: 720px; border-collapse: collapse; font-size: 0.85rem; }
.shap-table th {
    background: rgba(248, 250, 252, 0.95); color: #475569; font-weight: 800; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.04em; padding: 0.75rem 0.6rem;
    border-bottom: 2px solid #e2e8f0; vertical-align: bottom; text-align: left;
}
.shap-table td { padding: 0.6rem; color: #0f172a; border-bottom: 1px solid #f1f5f9; }
.shap-table .shap-num { text-align: right; font-variant-numeric: tabular-nums; }
.shap-table td.shap-num { white-space: nowrap; }
.shap-table th.shap-num { white-space: normal; min-width: 58px; }
.shap-table .shap-model { font-weight: 700; white-space: nowrap; }
.shap-table tr.shap-proposed td { border-bottom: 2px solid rgba(79, 70, 229, 0.25); }
.shap-table tr.shap-proposed td.shap-model { color: #3730a3; border-left: 4px solid #4f46e5; }
.shap-val { font-size: 0.8rem; color: #0f172a; text-transform: none; letter-spacing: 0; margin-top: 2px; }
.shap-median {
    font-size: 0.62rem; font-weight: 800; color: #0284c7; background: #f0f9ff;
    border: 1px solid #bae6fd; border-radius: 4px; padding: 0 4px;
}
.shap-legend {
    display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 0.78rem; color: #475569;
    margin-top: 0.6rem; padding: 0 0.25rem;
}
.shap-swatch {
    display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px;
}
.shap-method {
    background: rgba(255, 255, 255, 0.85); border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 0.75rem 1rem; margin: 1rem 0 2rem; position: relative; z-index: 10;
}
.shap-method summary { cursor: pointer; font-weight: 700; color: #0f172a; font-size: 0.95rem; }
.shap-method ul { margin: 0.75rem 0 0.25rem; padding-left: 1.2rem; color: #334155; font-size: 0.88rem; line-height: 1.6; }
.shap-method code { font-size: 0.8rem; }
@media (max-width: 1200px) { .shap-wrapper { width: 95vw; } }
@media (max-width: 768px) { .shap-wrapper { width: 100%; left: 0; transform: none; } }
</style>
"""


def render_shap_section(input_df: pd.DataFrame):
    """Explain the current prediction with SHAP for every model and render the results."""
    st.markdown(SHAP_CSS, unsafe_allow_html=True)
    st.markdown("""
<div style="margin: 3rem 0 0.75rem; position: relative; z-index: 10;">
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
<div style="background: #4f46e5; color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; box-shadow: 0 4px 10px rgba(79,70,229,0.3);">🔍</div>
<h3 style="margin: 0; color: #0f172a; font-weight: 800; font-size: 1.4rem; letter-spacing: -0.5px;">Why These Predictions? (SHAP)</h3>
</div>
<p style="margin: 0; color: #475569; font-size: 0.9rem; line-height: 1.6;">How much each patient feature pushed every model's predicted probability of diabetes <b>up</b> (red) or <b>down</b> (blue), in percentage points. All nine models are explained with the same method and the same reference patients, so their feature contributions can be compared directly.</p>
</div>
""", unsafe_allow_html=True)

    status = st.empty()
    progress = st.progress(0)

    def on_progress(i, total, name):
        status.caption(f"Computing exact SHAP values · {name} ({i + 1}/{total})")
        progress.progress(i / total)

    try:
        results = explain_all_models(input_df, progress_callback=on_progress)
    except Exception as e:  # keep the predictions on screen even if SHAP fails
        status.empty()
        progress.empty()
        st.warning(f"⚠️ SHAP explanation could not be computed: {e}")
        return
    status.empty()
    progress.empty()

    imputed = get_proposed_predictor().impute(input_df)[FEATURES].astype(float).iloc[0]
    patient_values = imputed.to_dict()
    imputed_features = {c for c in get_proposed_predictor().medians if float(input_df[c].iloc[0]) == 0}
    feature_order = sorted(
        FEATURES, key=lambda f: -np.mean([abs(r["shap_values"][f]) for r in results.values()])
    )

    # --- A. Side-by-side comparison of all models ---
    st.markdown('<h4 style="color: #0f172a; font-weight: 800; margin: 1rem 0 0.25rem;">📊 Feature Contributions Across All Models</h4>', unsafe_allow_html=True)
    st.markdown(_comparison_table_html(results, patient_values, imputed_features, feature_order), unsafe_allow_html=True)

    # --- B. Probability breakdown per model ---
    st.markdown('<h4 style="color: #0f172a; font-weight: 800; margin: 1.5rem 0 0.25rem;">🧮 Feature Impact by Model</h4>', unsafe_allow_html=True)
    st.markdown('<p style="color: #475569; font-size: 0.9rem; margin: 0 0 0.5rem;">Pick a model to see which features pushed its prediction up or down. Every tab uses the same scale.</p>', unsafe_allow_html=True)

    max_abs = max(abs(v) for r in results.values() for v in r["shap_values"].values())
    names = list(results.keys())
    tab_labels = ["⭐ Stacking Ensemble" if n == PROPOSED_MODEL_NAME else n for n in names]
    for tab, name in zip(st.tabs(tab_labels), names):
        with tab:
            st.altair_chart(_feature_chart(results[name], patient_values, imputed_features, max_abs), width="stretch")
            st.markdown(_summary_markdown(results[name], patient_values, imputed_features), unsafe_allow_html=True)

    # Plain HTML <details> instead of st.expander: the app's global font override breaks
    # Streamlit's icon font, which would show the expander chevron as text.
    st.markdown("""
<details class="shap-method">
<summary>ℹ️ How these SHAP values are calculated</summary>
<ul>
<li><b>Method:</b> exact Shapley values (<code>shap.explainers.Exact</code>), evaluating all 2⁸ = 256 combinations of the 8 features, so there is no sampling noise.</li>
<li><b>What is explained:</b> the predicted probability of diabetes, so each contribution is in percentage points of risk, measured against the model's average prediction over the reference patients.</li>
<li><b>Reference patients (background):</b> the same 100 patients as the notebook, sampled from the imputed training set with <code>train_test_split(X_train, y_train, train_size=100, stratify=y_train, random_state=42)</code>. Their average prediction is each model's base risk (see <i>Base Risk per Model</i> below). Rebuild the training file with <code>build_reference_data.py</code>.</li>
<li><b>Same inputs for every model:</b> features are explained after median imputation and before scaling; baseline models are explained together with their scaler, and the Stacking Ensemble end to end (including calibration).</li>
<li><b>Imputed fields:</b> values marked <i>median</i> were left blank or zero and replaced with the training median before prediction; their contribution reflects that median value.</li>
</ul>
</details>
""", unsafe_allow_html=True)


# ============================================
# 6. BASE RISK PER MODEL (NOTEBOOK "CONFIRM BASE RISK" CELL)
# ============================================
BASE_RISK_CSS = """
<style>
.base-risk-wrapper {
    width: 100%; overflow-x: auto; margin: 1rem 0 0.5rem; border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05); background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(16px); border: 1px solid rgba(226, 232, 240, 0.8);
}
.base-risk-table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.95rem; }
.base-risk-table th {
    background: rgba(248, 250, 252, 0.8); color: #475569; font-weight: 800; padding: 1rem 1.25rem;
    border-bottom: 2px solid #e2e8f0; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;
}
.base-risk-table td { padding: 0.8rem 1.25rem; color: #334155; border-bottom: 1px solid #f1f5f9; font-weight: 500; }
.base-risk-table .num { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; color: #0f172a; }
.base-risk-table .muted { color: #64748b; font-size: 0.85rem; }
.base-risk-table tr.proposed td {
    background: linear-gradient(90deg, rgba(79, 70, 229, 0.08) 0%, rgba(124, 58, 237, 0.03) 100%);
    font-weight: 700; color: #3730a3; border-bottom: 2px solid rgba(79, 70, 229, 0.2);
}
.base-risk-table tr.proposed td:first-child { border-left: 4px solid #4f46e5; }
.base-risk-badge {
    background: linear-gradient(135deg, #4f46e5 0%, #db2777 100%); color: white; padding: 3px 8px;
    border-radius: 12px; font-size: 0.65rem; font-weight: 800; margin-left: 8px; letter-spacing: 0.5px;
}
.base-risk-note { font-size: 0.8rem; color: #475569; line-height: 1.6; margin: 0.5rem 0.25rem 2rem; }
.base-risk-note code { font-size: 0.75rem; }
</style>
"""


def render_base_risk_section():
    """Show every model's base risk, computed exactly like the notebook cell."""
    st.markdown(BASE_RISK_CSS, unsafe_allow_html=True)
    st.markdown("""
<div style="margin-top: 2.5rem; position: relative; z-index: 10;">
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
<div style="background: #0284c7; color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; box-shadow: 0 4px 10px rgba(2, 132, 199, 0.3);">📌</div>
<h3 style="margin: 0; color: #0f172a; font-weight: 800; font-size: 1.4rem; letter-spacing: -0.5px;">Base Risk per Model</h3>
</div>
<p style="margin: 0; color: #475569; font-size: 0.9rem; line-height: 1.6;">Before looking at any patient, each model has a starting point: its <b>average predicted risk for 100 reference patients</b> from the training data. SHAP measures every feature contribution from this point. It is fixed, so it stays the same whatever patient data you enter; only the feature contributions change.</p>
</div>
""", unsafe_allow_html=True)

    try:
        with st.spinner("Calculating base risk for all models..."):
            df = compute_base_risk()
            _, outcomes = load_reference_sample()
    except Exception as e:
        st.warning(f"⚠️ Base risk could not be calculated: {e}")
        return

    rows = ""
    for _, r in df.iterrows():
        is_proposed = r["Model Name"] == PROPOSED_MODEL_NAME
        value = r["Base Risk (%)"]
        shown = f"{value:.1f}%" if isinstance(value, float) else html.escape(str(value))
        name = html.escape(r["Model Name"])
        badge = '<span class="base-risk-badge">PROPOSED</span>' if is_proposed else ""
        rows += (
            f'<tr class="{"proposed" if is_proposed else ""}">'
            f"<td>{name}{badge}</td>"
            f'<td class="muted">{r["Input Data"]}</td>'
            f'<td class="num">{shown}</td>'
            f"</tr>"
        )

    n_pos = int(outcomes.sum())
    st.markdown(f"""
<div class="base-risk-wrapper">
<table class="base-risk-table">
<thead><tr><th>Model Architecture</th><th>Input Data</th><th style="text-align: right;">Base Risk</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
<div class="base-risk-note">Reference patients: {len(outcomes)} sampled from the imputed training set with
<code>train_test_split(X_train, y_train, train_size={REFERENCE_SIZE}, stratify=y_train, random_state={RANDOM_STATE})</code>
({n_pos} diabetic, {len(outcomes) - n_pos} non-diabetic). The Stacking Ensemble receives the raw imputed values;
baseline models receive the same values scaled with <code>scaler.pkl</code>.</div>
""", unsafe_allow_html=True)
