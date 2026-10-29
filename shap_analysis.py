"""
SHAP (SHapley Additive exPlanations) for every model in the app.

The same method is applied to all 9 models, so their feature contributions can be compared
directly:

* Explainer   - exact Shapley values from Formula 7 of the thesis, computed directly over all
                2^8 = 256 feature coalitions (``shapley.py``). No sampling noise, so results are
                fully reproducible. The values are identical to ``shap.explainers.Exact`` with an
                ``Independent`` masker (checked by ``verify_shap.py``); computing them by hand lets
                the app show every intermediate number of the formula.
* Output      - the predicted probability of diabetes, P(diabetic): the number shown on each
                result card (not log-odds). Contributions are in probability units and add up
                exactly:   base value + sum(SHAP values) = predicted probability   (Formula 6).
* Features    - the 8 clinical features AFTER median imputation and BEFORE scaling. Baseline
                models are wrapped together with their StandardScaler, so every model is
                explained in the same, human-readable units (mg/dL, kg/m², years, ...).
* Background  - 100 reference patients sampled from the imputed training set exactly like the
                thesis notebook: train_test_split(X_train, y_train, train_size=100,
                stratify=y_train, random_state=42), from ``X_train_imputed.csv``
                (see ``build_reference_data.py``). Each model's base risk is its average
                predicted probability over these patients (Formula 8, ``compute_base_risk``).
* Proposed    - the Stacking Ensemble is explained end to end (SMOTE + stacking + sigmoid
  model       calibration), i.e. the final calibrated probability the user sees. Its
              meta-learner is also explained in terms of its 7 base learners (``meta_learner.py``).
* Text        - every model's result is also written out in plain English (``interpretation.py``).
"""
import html
from fractions import Fraction
from math import comb

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from interpretation import FEATURE_INFO, meta_learner_interpretation, model_interpretation, overall_summary
from meta_learner import BASE_LEARNER_LABELS, explain_meta_learner, stacking_internals
from predictor import BASELINE_MODELS, get_baseline_predictor, get_proposed_predictor
from shapley import coalition_values, shapley_breakdown, shapley_values, subset_contributions, subset_weight

PROPOSED_MODEL_NAME = "Stacking Ensemble Model"
TRAIN_DATA_PATH = "X_train_imputed.csv"  # notebook's imputed X_train, with y_train as "Outcome"
REFERENCE_SIZE = 100
RANDOM_STATE = 42
FEATURES = list(joblib.load("feature_names.joblib"))

FEATURE_LABELS = {f: info["label"] for f, info in FEATURE_INFO.items()}

# Short names used under the terms of the substituted formulas.
SHORT_LABELS = {
    "Pregnancies": "Preg",
    "Glucose": "Glucose",
    "BloodPressure": "BP",
    "SkinThickness": "Skin",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "DPF",
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


def patient_tuple(input_df: pd.DataFrame) -> tuple:
    """The patient's imputed feature values in FEATURES order (a tuple so it can be a cache key)."""
    imputed = get_proposed_predictor().impute(input_df)[FEATURES].astype(float)
    return tuple(imputed.iloc[0].tolist())


@st.cache_data(show_spinner=False, max_entries=256)
def explain_patient(model_name: str, patient: tuple) -> dict:
    """Exact SHAP values of P(diabetic) for one (already imputed) patient, from all 256 coalitions."""
    predictor = get_predictor(model_name)
    background = load_reference_patients()

    def predict_fn(X):
        return predictor.predict_proba_batch(pd.DataFrame(X, columns=FEATURES))

    v = coalition_values(predict_fn, np.array(patient, dtype=float), background.values)
    phi = shapley_values(v)
    return {
        "base_value": float(v[0]),                  # f(∅): the model's base risk
        "shap_values": {f: float(p) for f, p in zip(FEATURES, phi)},
        "probability": float(v[-1]),                # f(all 8 features): this patient's prediction
        "threshold": float(predictor.threshold),
        "coalitions": v,                            # f(S) for every subset S, for the step-by-step view
    }


def explain_all_models(input_df: pd.DataFrame, progress_callback=None) -> dict:
    """SHAP explanations for the proposed model and every baseline, keyed by model name."""
    patient = patient_tuple(input_df)
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


def _fraction(w: Fraction) -> str:
    return f"{w.numerator}/{w.denominator}"


def _section_title(text: str, margin_top: str = "1.5rem"):
    st.markdown(f'<h4 style="color: #0f172a; font-weight: 800; margin: {margin_top} 0 0.25rem; position: relative; '
                f'z-index: 10;">{text}</h4>', unsafe_allow_html=True)


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
# 4. CONTRIBUTION CHART (ONE MODEL)
# ============================================
def _contribution_df(contributions: dict, labels: dict) -> pd.DataFrame:
    ordered = sorted(contributions, key=lambda k: abs(contributions[k]), reverse=True)
    return pd.DataFrame([{
        "label": labels[k],
        "pts": contributions[k] * 100,
        "kind": "up" if contributions[k] >= 0 else "down",
        "text": _fmt_pts(contributions[k]),
    } for k in ordered])


def _contribution_chart(df: pd.DataFrame, max_abs: float):
    """Diverging bars: each input's push on P(diabetic), in percentage points.

    ``max_abs`` sets the scale; for the model tabs it is the largest |contribution| across ALL
    models, so every tab shares one scale.
    """
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
        tooltip=[alt.Tooltip("label:N", title="Input"), alt.Tooltip("text:N", title="Contribution (pts)")],
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


# ============================================
# 5. THE CALCULATION STEP BY STEP (FORMULAS 8, 7 AND 6 OF THE THESIS)
# ============================================
def _step_html(number: int, title: str, formula: str) -> str:
    return (f'<div class="calc-step"><span class="calc-step-num">{number}</span>{title}'
            f'<span class="calc-formula">{formula} in the thesis</span></div>')


def _breakdown_table_html(groups: list, label: str, n_players: int) -> str:
    """Formula 7 for one player, one row per subset size: the layout of the thesis' worked example."""
    rows = ""
    for g in groups:
        k = g["size"]
        weight = f"{k}!·{n_players - k - 1}!/{n_players}! = {_fraction(g['weight'])}"
        if g["count"] == 1:
            difference = f"{g['f_S_i']:.4f} − {g['f_S']:.4f} = {g['mean_marginal']:.4f}"
        else:
            difference = f"{g['mean_marginal']:.4f} <span class='calc-muted'>(average of {g['count']})</span>"
        total = f"{g['count']} × {_fraction(g['weight'])} × {g['mean_marginal']:.4f} = <b>{g['total']:.5f}</b>"
        rows += (f"<tr><td class='shap-num'>{k}</td><td class='shap-num'>{g['count']}</td>"
                 f"<td>{weight}</td><td>{difference}</td><td>{total}</td></tr>")
    phi = sum(g["total"] for g in groups)
    return f"""
<div class="calc-table-scroll"><table class="shap-table calc-table">
<thead><tr><th class="shap-num">|S|</th><th class="shap-num">Subsets</th><th>Weight of one subset</th>
<th>f(S ∪ {{{label}}}) − f(S)</th><th>Group total = subsets × weight × difference</th></tr></thead>
<tbody>{rows}
<tr class="calc-sum"><td colspan="4">ϕ<sub>{label}</sub> = sum of the group totals</td>
<td><b>{phi:.4f}</b> ({_fmt_pts(phi)} percentage points)</td></tr>
</tbody></table></div>"""


@st.fragment
def _shapley_explorer(key: str, players: list, labels: dict, coalitions, player_word: str):
    """Formula 7 worked out for one chosen player: grouped by subset size, then every subset of one size.

    A fragment, so changing the selection reruns only this part instead of the whole page.
    """
    v = np.asarray(coalitions, dtype=float)
    n = len(players)
    default = int(np.argmax(np.abs(shapley_values(v))))
    i = st.selectbox(f"Show Formula 7 for this {player_word}", range(n), index=default,
                     format_func=lambda j: labels[players[j]], key=f"{key}_player")
    label = labels[players[i]]
    st.markdown(_breakdown_table_html(shapley_breakdown(v, i), html.escape(label), n), unsafe_allow_html=True)

    size = st.selectbox(
        "List every subset S of size |S|", range(n), index=1, key=f"{key}_size",
        format_func=lambda k: f"|S| = {k}  ({comb(n - 1, k)} subset{'' if comb(n - 1, k) == 1 else 's'})",
    )
    df = pd.DataFrame([{
        "S (the other inputs present)": ", ".join(labels[players[j]] for j in r["members"]) or "∅ (none)",
        "f(S)": r["f_S"],
        f"f(S ∪ {{{label}}})": r["f_S_i"],
        "Difference": r["marginal"],
        "× weight": r["weighted"],
    } for r in subset_contributions(v, i, size)])
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.5f" if c == "× weight" else "%.4f") for c in df.columns[1:]
    })
    st.caption(f"Each subset of size {size} has weight {_fraction(subset_weight(size, n))}. "
               f"The '× weight' column adds up to the |S| = {size} group total in the table above.")


def _additive_latex(base: float, contributions: dict, short_labels: dict, probability: float) -> str:
    """Formula 6 with this patient's numbers, ordered by impact like the chart."""
    order = sorted(contributions, key=lambda k: abs(contributions[k]), reverse=True)
    terms = [rf"\underset{{\text{{{short_labels[k]}}}}}{{({contributions[k]:.4f})(1)}}" for k in order]
    lines = [rf"g(z') &= {base:.4f}" + "".join(f" + {t}" for t in terms[:4])]
    for start in range(4, len(terms), 4):
        lines.append(r"&\quad" + "".join(f" + {t}" for t in terms[start:start + 4]))
    lines.append(rf"&= {probability:.4f} \;\; ({probability * 100:.2f}\%)")
    return r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}"


def _render_calculation(key, players, labels, short_labels, result, player_word, f_s_text, presence_note):
    """Base risk (Formula 8), Shapley values (Formula 7) and the additive formula (Formula 6) with real numbers."""
    base, p, n = result["base_value"], result["probability"], len(players)

    st.markdown(_step_html(1, "Base risk ϕ₀", "Formula 8"), unsafe_allow_html=True)
    st.latex(rf"\phi_0 = \frac{{1}}{{N}} \sum_{{j=1}}^{{N}} \hat{{P}}(y = 1 \mid x_j) "
             rf"= \frac{{{base * REFERENCE_SIZE:.4f}}}{{{REFERENCE_SIZE}}} = {base:.4f} \;\; ({base * 100:.2f}\%)")
    st.markdown(f"<p class='calc-text'>N = {REFERENCE_SIZE} reference patients, and P̂ is the model's predicted "
                f"probability of diabetes for reference patient j. Every contribution below is measured from this "
                f"starting point.</p>", unsafe_allow_html=True)

    st.markdown(_step_html(2, f"Shapley value ϕᵢ of each {player_word}", "Formula 7"), unsafe_allow_html=True)
    st.latex(r"\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!}"
             r"\,\bigl[ f(S \cup \{i\}) - f(S) \bigr]")
    st.markdown(f"<p class='calc-text'>|F| = {n} {player_word}s, so each {player_word} is compared over all "
                f"2<sup>{n - 1}</sup> = {2 ** (n - 1)} subsets S of the other {n - 1}. {f_s_text} "
                f"Pick a {player_word} to see its value worked out like the example in the thesis.</p>",
                unsafe_allow_html=True)
    _shapley_explorer(key, players, labels, result["coalitions"], player_word)

    st.markdown(_step_html(3, "Additive Feature Attribution", "Formula 6"), unsafe_allow_html=True)
    st.latex(r"g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i")
    st.latex(_additive_latex(base, result["shap_values"], short_labels, p))
    st.markdown(f"<p class='calc-text'>M = {n}. {presence_note} So g(z') = {p:.4f}: exactly the model's predicted "
                f"probability, shown as {int(round(p * 100, 0))}% on the result card.</p>", unsafe_allow_html=True)


FEATURE_F_S_TEXT = (
    "f(S) is the model's average predicted probability over the 100 reference patients when the features "
    "in S are set to this patient's values (the other features keep each reference patient's own values). "
    "f(∅) is the base risk ϕ₀ and f(all 8 features) is this patient's prediction."
)


# ============================================
# 6. INSIDE THE STACKING ENSEMBLE (META-LEARNER RESULT AND ITS SHAP)
# ============================================
META_SHORT_LABELS = {"rf": "RF", "gbm": "GBM", "xgb": "XGB", "lgb": "LGB", "knn": "KNN", "svm": "SVM", "nb": "NB"}

META_F_S_TEXT = (
    "f(S) is the final probability, averaged over the 7 calibrated copies and the 100 reference patients, "
    "when the base learners in S give this patient's estimates and the others give the reference patient's "
    "estimates. f(∅) is the Stacking Ensemble's base risk (the same ϕ₀ as in its feature tab) and f(all 7) "
    "is this patient's final probability."
)


def _meta_flow_html(meta: dict, internals: dict) -> str:
    preds = meta["patient_predictions"]
    copies = internals["copies"]
    raw = np.mean([c["meta_probability"] for c in copies])
    z = np.mean([c["z"] for c in copies])
    boxes = [
        ("Level 0 · 7 base learners", f"{min(preds.values()) * 100:.1f}–{max(preds.values()) * 100:.1f}%",
         "each learner's P(diabetic)"),
        ("Level 1 · Bagged LR meta-learner", f"{raw * 100:.1f}%", f"score z = {z:.3f} (log-odds)"),
        ("Sigmoid calibration", f"{internals['probability'] * 100:.1f}%", "average of the 7 copies = result card"),
    ]
    html_boxes = "".join(
        f'<div class="meta-box"><div class="meta-box-title">{t}</div><div class="meta-box-value">{v}</div>'
        f'<div class="meta-box-note">{n}</div></div>' for t, v, n in boxes
    )
    return f'<div class="meta-flow">{html_boxes}</div>'


def _meta_table_html(meta: dict, internals: dict) -> str:
    phi = meta["shap_values"]
    learners = internals["learners"]
    weights = {n: np.mean([c["weights"][j] for c in internals["copies"]]) for j, n in enumerate(learners)}
    vmax = max(abs(v) for v in phi.values())
    rows = ""
    for n in sorted(learners, key=lambda n: abs(phi[n]), reverse=True):
        rows += (
            f"<tr><td class='shap-model'>{BASE_LEARNER_LABELS[n]} <span class='calc-muted'>base_{n}</span></td>"
            f"<td class='shap-num'>{meta['patient_predictions'][n] * 100:.1f}%</td>"
            f"<td class='shap-num'>{meta['reference_predictions'][n] * 100:.1f}%</td>"
            f"<td class='shap-num'>{weights[n]:+.3f}</td>"
            f"<td class='shap-num' style='background:{_cell_background(phi[n], vmax)}; font-weight:700;'>"
            f"{_fmt_pts(phi[n])}</td></tr>"
        )
    return f"""
<div class="calc-table-scroll"><table class="shap-table">
<thead><tr><th>Base learner</th><th class="shap-num">Estimate for this patient</th>
<th class="shap-num">Average reference patient</th><th class="shap-num">Meta-learner weight</th>
<th class="shap-num">SHAP contribution (pts)</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<div class="shap-legend"><span>Estimates and weights are averaged over the 7 calibrated copies.
A positive weight means a higher estimate from that learner raises the meta-learner's score.</span></div>"""


def _copy_latex(k: int, copy: dict, learners: list) -> str:
    terms = [rf"\underset{{\text{{{META_SHORT_LABELS.get(n, n)}}}}}{{({w:.4f})({p:.4f})}}"
             for n, w, p in zip(learners, copy["weights"], copy["base"])]
    lines = [rf"z_{k} &= {copy['intercept']:.4f}" + "".join(f" + {t}" for t in terms[:3])]
    for start in range(3, len(terms), 3):
        lines.append(r"&\quad" + "".join(f" + {t}" for t in terms[start:start + 3]))
    lines.append(rf"&= {copy['z']:.4f}")
    return r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}"


def _copies_table_html(internals: dict) -> str:
    rows = ""
    for k, c in enumerate(internals["copies"], 1):
        rows += (f"<tr><td>Copy {k}</td><td class='shap-num'>{c['z']:.4f}</td>"
                 f"<td class='shap-num'>{c['meta_probability']:.4f}</td><td class='shap-num'>{c['a']:.4f}</td>"
                 f"<td class='shap-num'>{c['b']:.4f}</td><td class='shap-num'><b>{c['probability']:.4f}</b></td></tr>")
    return f"""
<div class="calc-table-scroll"><table class="shap-table calc-table">
<thead><tr><th>Calibrated copy</th><th class="shap-num">Score z<sub>k</sub></th>
<th class="shap-num">Meta-learner probability</th><th class="shap-num">A<sub>k</sub></th>
<th class="shap-num">B<sub>k</sub></th><th class="shap-num">Calibrated P<sub>k</sub></th></tr></thead>
<tbody>{rows}
<tr class="calc-sum"><td colspan="5">P = average of the 7 calibrated copies (the probability on the result card)</td>
<td class="shap-num"><b>{internals['probability']:.4f}</b></td></tr>
</tbody></table></div>"""


def _render_meta_learner_section(patient: tuple):
    st.markdown("""
<div style="margin: 3rem 0 0.75rem; position: relative; z-index: 10;">
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
<div style="background: #7c3aed; color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; box-shadow: 0 4px 10px rgba(124,58,237,0.3);">🧠</div>
<h3 style="margin: 0; color: #0f172a; font-weight: 800; font-size: 1.4rem; letter-spacing: -0.5px;">Inside the Stacking Ensemble: Meta-Learner Result</h3>
</div>
<p style="margin: 0; color: #475569; font-size: 0.9rem; line-height: 1.6;">The Stacking Ensemble decides in two levels. Seven <b>base learners</b> first estimate the risk from the 8 features; a <b>Bagged Logistic Regression meta-learner</b> combines their estimates; <b>sigmoid calibration</b> then turns the result into a reliable probability. The saved model holds 7 calibrated copies of this pipeline (one per calibration fold), and the result card shows their average. These 7 base learners are part of the ensemble (trained with SMOTE and tuned settings); they are not the 8 baseline models above.</p>
</div>
""", unsafe_allow_html=True)

    try:
        with st.spinner("Explaining the meta-learner..."):
            internals = stacking_internals(patient)
            meta = explain_meta_learner(patient, load_reference_patients())
    except Exception as e:
        st.warning(f"⚠️ The meta-learner explanation could not be computed: {e}")
        return

    learners = internals["learners"]
    labels = {n: BASE_LEARNER_LABELS.get(n, n) for n in learners}

    st.markdown(_meta_flow_html(meta, internals), unsafe_allow_html=True)

    _section_title("📋 What the Meta-Learner Received and How Much Each Base Learner Counted", "1.25rem")
    st.markdown(_meta_table_html(meta, internals), unsafe_allow_html=True)

    _section_title("🧮 Base Learner Contributions to the Final Probability (Meta-Learner SHAP)")
    chart_labels = {n: f"{labels[n]} = {meta['patient_predictions'][n] * 100:.1f}%" for n in learners}
    max_abs = max(abs(v) for v in meta["shap_values"].values())
    st.altair_chart(_contribution_chart(_contribution_df(meta["shap_values"], chart_labels), max_abs), width="stretch")
    st.markdown(meta_learner_interpretation(meta, internals, labels), unsafe_allow_html=True)

    with st.expander("🧮 Show the calculation: meta-learner and calibration, copy by copy"):
        st.markdown("<p class='calc-text'>Each calibrated copy k turns its base learners' estimates p<sub>kj</sub> "
                    "into a meta-learner score z<sub>k</sub> (the average log-odds of its 15 logistic regressions), "
                    "calibrates it with its own sigmoid parameters A<sub>k</sub> and B<sub>k</sub>, and the model "
                    "averages the 7 results.</p>", unsafe_allow_html=True)
        st.latex(r"z_k = b_k + \sum_{j=1}^{7} w_{kj}\, p_{kj} \qquad P_k = \frac{1}{1 + e^{\,A_k z_k + B_k}}"
                 r" \qquad P = \frac{1}{7} \sum_{k=1}^{7} P_k")
        copy_tabs = st.tabs([f"Copy {k}" for k in range(1, len(internals["copies"]) + 1)])
        for k, (tab, copy) in enumerate(zip(copy_tabs, internals["copies"]), 1):
            with tab:
                st.latex(_copy_latex(k, copy, learners))
                st.latex(rf"P_{k} = \frac{{1}}{{1 + e^{{({copy['a']:.4f})({copy['z']:.4f}) + ({copy['b']:.4f})}}}}"
                         rf" = {copy['probability']:.4f}")
                st.caption(f"Under each term: the base learner, as (meta-learner weight)(its estimate). For reference, "
                           f"this copy's meta-learner probability before calibration is {copy['meta_probability']:.4f} "
                           f"(the average of its 15 logistic regressions' probabilities); calibration uses the score z.")
        st.markdown(_copies_table_html(internals), unsafe_allow_html=True)

    with st.expander("🧮 Show the calculation: meta-learner SHAP (Formulas 6–8 over the 7 base learners)"):
        _render_calculation(
            key="meta", players=learners, labels=labels, short_labels=META_SHORT_LABELS, result=meta,
            player_word="base learner", f_s_text=META_F_S_TEXT,
            presence_note="z'ᵢ = 1 for every base learner, because all seven take part in this patient's prediction.",
        )

    st.markdown("""
<p class="calc-text" style="margin-top: 0.75rem;"><b>Why can this differ from the meta-learner summary plot further down?</b>
That plot (from the thesis notebook) explains an uncalibrated stacking model refit on the full training set, over 100
test patients. This section explains the 7 calibrated copies that actually produce this patient's prediction, so its
numbers add up exactly to the result card.</p>
""", unsafe_allow_html=True)


# ============================================
# 7. STREAMLIT SECTION
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
.interp-card {
    background: rgba(255, 255, 255, 0.92); border: 1px solid #e2e8f0; border-left: 4px solid #4f46e5;
    border-radius: 12px; padding: 0.9rem 1.1rem; margin: 0.5rem 0 1rem; font-size: 0.9rem; line-height: 1.65;
    position: relative; z-index: 10;
}
.interp-card p { margin: 0 0 0.5rem; color: #0f172a; font-size: 0.9rem; }
.interp-card ul { margin: 0.25rem 0; padding-left: 1.2rem; }
.interp-card li { margin-bottom: 0.3rem; color: #1e293b; font-size: 0.88rem; }
.interp-card ul ul { margin-top: 0.25rem; }
.interp-card .interp-note { font-size: 0.82rem; color: #475569; margin: 0.5rem 0 0; }
.interp-summary { border-left-color: #0284c7; }
.calc-step {
    display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 1.1rem 0 0.25rem;
    font-weight: 800; color: #0f172a; font-size: 0.95rem;
}
.calc-step-num {
    background: #4f46e5; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex;
    align-items: center; justify-content: center; font-size: 0.8rem;
}
.calc-formula {
    font-size: 0.7rem; font-weight: 700; color: #4f46e5; background: #eef2ff; border: 1px solid #c7d2fe;
    border-radius: 9999px; padding: 2px 8px;
}
.calc-text { color: #334155; font-size: 0.86rem; line-height: 1.6; margin: 0.25rem 0 0.5rem; }
.katex-display { overflow-x: auto; overflow-y: hidden; padding-bottom: 2px; }  /* scroll long formulas instead of cutting them off */
.calc-table-scroll {
    overflow-x: auto; border-radius: 12px; background: rgba(255, 255, 255, 0.95);
    border: 1px solid #e2e8f0; margin: 0.25rem 0 0.75rem;
}
.calc-table td { white-space: nowrap; font-variant-numeric: tabular-nums; }
.calc-table tr.calc-sum td { background: #f8fafc; font-weight: 700; border-top: 2px solid #e2e8f0; }
.calc-muted { color: #64748b; font-size: 0.78rem; font-weight: 500; }
.meta-flow { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 1rem 0 0.5rem; position: relative; z-index: 10; }
.meta-box {
    background: rgba(255, 255, 255, 0.92); border: 1px solid #e2e8f0; border-top: 4px solid #7c3aed;
    border-radius: 12px; padding: 0.8rem 1rem;
}
.meta-box:last-child { border-top-color: #4f46e5; }
.meta-box-title { font-size: 0.72rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; }
.meta-box-value { font-size: 1.5rem; font-weight: 800; color: #0f172a; margin: 0.2rem 0; }
.meta-box-note { font-size: 0.78rem; color: #64748b; }
@media (max-width: 1200px) { .shap-wrapper { width: 95vw; } }
@media (max-width: 768px) {
    .shap-wrapper { width: 100%; left: 0; transform: none; }
    .meta-flow { grid-template-columns: 1fr; }
}
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

    patient = patient_tuple(input_df)
    patient_values = dict(zip(FEATURES, patient))
    imputed_features = {c for c in get_proposed_predictor().medians if float(input_df[c].iloc[0]) == 0}
    reference_means = load_reference_patients().mean().to_dict()
    feature_order = sorted(
        FEATURES, key=lambda f: -np.mean([abs(r["shap_values"][f]) for r in results.values()])
    )

    # --- Plain-English summary of all models ---
    _section_title("📝 What the Models Say, in Plain English", "1rem")
    st.markdown(overall_summary(results, PROPOSED_MODEL_NAME), unsafe_allow_html=True)

    # --- A. Side-by-side comparison of all models ---
    _section_title("📊 Feature Contributions Across All Models")
    st.markdown(_comparison_table_html(results, patient_values, imputed_features, feature_order), unsafe_allow_html=True)

    # --- B. Probability breakdown per model ---
    _section_title("🧮 Feature Impact by Model")
    st.markdown('<p style="color: #475569; font-size: 0.9rem; margin: 0 0 0.5rem;">Pick a model to see which features pushed its prediction up or down, what that means in plain English, and the full SHAP calculation behind it. Every tab uses the same scale.</p>', unsafe_allow_html=True)

    max_abs = max(abs(v) for r in results.values() for v in r["shap_values"].values())
    presence_note = "z'ᵢ = 1 for every feature, because all eight are present for this patient" + (
        " (fields left blank were filled with the training median, so they count as present)." if imputed_features
        else "."
    )
    names = list(results.keys())
    tab_labels = ["⭐ Stacking Ensemble" if n == PROPOSED_MODEL_NAME else n for n in names]
    for i, (tab, name) in enumerate(zip(st.tabs(tab_labels), names)):
        with tab:
            chart_labels = {f: _feature_label(f, patient_values[f], imputed_features) for f in FEATURES}
            chart_df = _contribution_df(results[name]["shap_values"], chart_labels)
            st.altair_chart(_contribution_chart(chart_df, max_abs), width="stretch")
            st.markdown(model_interpretation(name, results[name], patient_values, imputed_features, reference_means,
                                             is_proposed=name == PROPOSED_MODEL_NAME), unsafe_allow_html=True)
            with st.expander("🧮 Show the calculation: base risk, Shapley values and additive formula (Formulas 6–8)"):
                _render_calculation(
                    key=f"calc_{i}", players=FEATURES, labels=FEATURE_LABELS, short_labels=SHORT_LABELS,
                    result=results[name], player_word="feature", f_s_text=FEATURE_F_S_TEXT,
                    presence_note=presence_note,
                )

    # --- C. The Stacking Ensemble's meta-learner ---
    _render_meta_learner_section(patient)

    # Plain HTML <details> keeps this note compact; the calculations above use st.expander because
    # they contain widgets.
    st.markdown("""
<details class="shap-method">
<summary>ℹ️ How these SHAP values are calculated</summary>
<ul>
<li><b>Method:</b> exact Shapley values from Formula 7, computed directly over all 2⁸ = 256 combinations of the 8 features (2⁷ = 128 combinations of the 7 base learners for the meta-learner), so there is no sampling noise. The results are identical to the <code>shap</code> library's <code>Exact</code> explainer; run <code>verify_shap.py</code> to check.</li>
<li><b>What is explained:</b> the predicted probability of diabetes, so each contribution is in percentage points of risk, measured against the model's average prediction over the reference patients.</li>
<li><b>Reference patients (background):</b> the same 100 patients as the notebook, sampled from the imputed training set with <code>train_test_split(X_train, y_train, train_size=100, stratify=y_train, random_state=42)</code>. Their average prediction is each model's base risk (see <i>Base Risk per Model</i> below). Rebuild the training file with <code>build_reference_data.py</code>.</li>
<li><b>Same inputs for every model:</b> features are explained after median imputation and before scaling; baseline models are explained together with their scaler, and the Stacking Ensemble end to end (including calibration).</li>
<li><b>Imputed fields:</b> values marked <i>median</i> were left blank or zero and replaced with the training median before prediction; their contribution reflects that median value.</li>
<li><b>Plain-English text:</b> written from fixed templates filled with the numbers on this page (no AI model), so the same input always gives the same explanation.</li>
</ul>
</details>
""", unsafe_allow_html=True)


# ============================================
# 8. BASE RISK PER MODEL (NOTEBOOK "CONFIRM BASE RISK" CELL)
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
