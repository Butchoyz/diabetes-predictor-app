"""
Plain-English interpretation of the predictions and their SHAP explanations.

The text is written from fixed sentence templates, filled in with numbers shown elsewhere on the
page: the predicted probability and cut-off, the base risk, the SHAP values, the patient's values,
the reference patients' averages and the normal/risk ranges listed under the input fields. No AI
model writes it, so the same input always gives the same explanation, and it never states anything
the numbers don't show.
"""
from collections import Counter
from html import escape

NEGLIGIBLE = 0.005  # contributions under 0.5 percentage points count as "little or no effect"
BORDERLINE = 0.05   # predictions within 5 points of the cut-off are called borderline
TOP_N = 3           # drivers described in detail per direction

# Labels, units and the normal / risk ranges shown under the input fields in app.py.
FEATURE_INFO = {
    "Pregnancies": {"label": "Pregnancies", "unit": "", "normal": (0, 2), "risk": (">=", 3)},
    "Glucose": {"label": "Glucose", "unit": "mg/dL", "normal": (70, 139), "risk": (">=", 140)},
    "BloodPressure": {"label": "Blood Pressure", "unit": "mm Hg", "normal": (60, 79), "risk": (">=", 80)},
    "SkinThickness": {"label": "Skin Thickness", "unit": "mm", "normal": (12, 23), "risk": (">", 24)},
    "Insulin": {"label": "Insulin", "unit": "μU/mL", "normal": (16, 166), "risk": (">", 166)},
    "BMI": {"label": "BMI", "unit": "kg/m²", "normal": (18.5, 24.9), "risk": (">=", 25.0)},
    "DiabetesPedigreeFunction": {"label": "Diabetes Pedigree", "unit": "", "normal": (None, 0.59), "risk": (">=", 0.60)},
    "Age": {"label": "Age", "unit": "years", "normal": None, "risk": ("band", (46, 64))},
}


# ============================================
# 1. FORMATTING
# ============================================
def _num(v: float) -> str:
    return f"{v:.0f}" if float(v).is_integer() else f"{v:g}"


def _with_unit(feature: str, v: float) -> str:
    unit = FEATURE_INFO[feature]["unit"]
    return f"{_num(v)} {unit}" if unit else _num(v)


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _signed(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def _pts(v: float) -> str:
    return f"{_signed(v * 100)} points"


def _sublist(items: list) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _join(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def range_status(feature: str, v: float):
    """Where a value falls in the ranges shown under the input fields, or None (Age outside its band)."""
    info = FEATURE_INFO[feature]
    unit = f" {info['unit']}" if info["unit"] else ""
    op, limit = info["risk"]
    if op == "band":
        lo, hi = limit
        return f"in the {lo}–{hi} age band marked as higher risk" if lo <= v <= hi else None

    risk_text = f"{'≥' if op == '>=' else '>'} {_num(limit)}{unit}"
    if (v >= limit) if op == ">=" else (v > limit):
        return f"in the risk range, {risk_text}"
    lo, hi = info["normal"]
    normal_text = f"{_num(lo)}–{_num(hi)}{unit}" if lo is not None else f"≤ {_num(hi)}{unit}"
    if lo is not None and v < lo:
        return f"below the normal range of {normal_text}"
    if v <= hi:
        return f"within the normal range of {normal_text}"
    return f"above the normal range of {normal_text} but below the risk level ({risk_text})"


# ============================================
# 2. ONE MODEL
# ============================================
def model_interpretation(model_name, result, patient_values, imputed_features, reference_means,
                         is_proposed=False) -> str:
    """An HTML paragraph explaining one model's prediction from its SHAP values."""
    p, cutoff, base = result["probability"], result["threshold"], result["base_value"]
    phi = result["shap_values"]
    features = list(phi)
    at_risk = p >= cutoff

    lead = (
        f"<b>{escape(model_name)}</b> estimates a <b>{_pct(p)}</b> probability of diabetes. That is "
        f"{'above' if at_risk else 'below'} its {cutoff * 100:.0f}% cut-off, so the patient is classified as "
        f"<b>{'diabetic risk' if at_risk else 'non-diabetic'}</b>."
    )
    if abs(p - cutoff) < BORDERLINE:
        lead += " The estimate is within 5 points of the cut-off, so this result is borderline."
    delta = p - base
    lead += (
        f" The model starts from a <b>base risk of {_pct(base)}</b> (its average prediction for the 100 "
        f"reference patients), and this patient's eight features {'raised' if delta >= 0 else 'lowered'} it "
        f"by <b>{abs(delta) * 100:.1f} percentage points</b>."
    )

    def describe(f):
        label = FEATURE_INFO[f]["label"]
        if f in imputed_features:
            return (f"{label}: <b>{_pts(phi[f])}</b> (left blank, so the training median of "
                    f"{_with_unit(f, patient_values[f])} was used)")
        details = [range_status(f, patient_values[f]),
                   f"reference average {_with_unit(f, round(reference_means[f], 1))}"]
        return (f"{label} = {_with_unit(f, patient_values[f])}: <b>{_pts(phi[f])}</b> "
                f"({'; '.join(d for d in details if d)})")

    def short(f):
        return f"{FEATURE_INFO[f]['label']} ({_pts(phi[f])})"

    up = sorted([f for f in features if phi[f] >= NEGLIGIBLE], key=lambda f: -phi[f])
    down = sorted([f for f in features if phi[f] <= -NEGLIGIBLE], key=lambda f: phi[f])
    small = [f for f in features if abs(phi[f]) < NEGLIGIBLE]

    bullets = []
    for title, group in (("Raised the risk", up), ("Lowered the risk", down)):
        if not group:
            bullets.append(f"<b>{title}:</b> no feature changed it by 0.5 points or more.")
            continue
        items = [describe(f) for f in group[:TOP_N]]
        if group[TOP_N:]:
            items.append(f"Also {_join([short(f) for f in group[TOP_N:]])}")
        bullets.append(f"<b>{title}:</b>{_sublist(items)}")
    if small:
        bullets.append(f"<b>Little or no effect</b> (under 0.5 points each): "
                       f"{_join([FEATURE_INFO[f]['label'] for f in small])}.")

    total_impact = sum(abs(v) for v in phi.values())
    if total_impact > 0:
        top = max(features, key=lambda f: abs(phi[f]))
        bullets.append(
            f"<b>{FEATURE_INFO[top]['label']}</b> was the single biggest driver, accounting for "
            f"{abs(phi[top]) / total_impact * 100:.0f}% of the total feature impact."
        )

    notes = []
    for f in up + down:
        if f in imputed_features:
            continue
        status = range_status(f, patient_values[f]) or ""
        ref = _with_unit(f, round(reference_means[f], 1))
        if phi[f] > 0 and status.startswith("within") and patient_values[f] > reference_means[f]:
            notes.append(f"{FEATURE_INFO[f]['label']} is within the normal range but still raised the risk; "
                         f"it is higher than the reference patients' average ({ref}).")
        elif phi[f] < 0 and status.startswith("in the risk") and patient_values[f] < reference_means[f]:
            notes.append(f"{FEATURE_INFO[f]['label']} is in the risk range but lowered the risk; "
                         f"it is lower than the reference patients' average ({ref}).")
    if notes:
        notes.append("SHAP measures each feature against the average reference patient, not against clinical "
                     "cut-offs.")
    if imputed_features:
        names = _join([FEATURE_INFO[f]["label"] for f in features if f in imputed_features])
        notes.append(f"{names} {'was' if len(imputed_features) == 1 else 'were'} left blank, so the training "
                     f"median{'' if len(imputed_features) == 1 else 's'} stood in; "
                     f"{'its contribution reflects' if len(imputed_features) == 1 else 'their contributions reflect'} "
                     f"the median, not a measurement.")
    if is_proposed:
        notes.append("To see how the meta-learner combined its seven base learners into this result, "
                     "see <i>Inside the Stacking Ensemble</i> below.")

    items = "".join(f"<li>{b}</li>" for b in bullets)
    note_html = f'<p class="interp-note">{" ".join(notes)}</p>' if notes else ""
    return f'<div class="interp-card"><p>{lead}</p><ul>{items}</ul>{note_html}</div>'


# ============================================
# 3. ALL MODELS TOGETHER
# ============================================
def overall_summary(results: dict, proposed_name: str) -> str:
    """An HTML summary of what all models say about this patient."""
    names = list(results)
    total = len(names)
    at_risk = [n for n in names if results[n]["probability"] >= results[n]["threshold"]]
    proposed = results[proposed_name]
    proposed_risk = proposed_name in at_risk

    verdicts = f"<b>{len(at_risk)} of {total} models</b> classify this patient as <b>diabetic risk</b>"
    if len(at_risk) < total:
        verdicts += f" and {total - len(at_risk)} as non-diabetic"
    verdicts += (
        f". The proposed {escape(proposed_name)} estimates <b>{_pct(proposed['probability'])}</b> against its "
        f"{proposed['threshold'] * 100:.0f}% cut-off, so it classifies the patient as "
        f"<b>{'diabetic risk' if proposed_risk else 'non-diabetic'}</b>."
    )
    lowest = min(names, key=lambda n: results[n]["probability"])
    highest = max(names, key=lambda n: results[n]["probability"])
    spread = (f"Predicted risk ranges from {_pct(results[lowest]['probability'])} ({escape(lowest)}) "
              f"to {_pct(results[highest]['probability'])} ({escape(highest)}).")

    top_by_model = {n: max(results[n]["shap_values"], key=lambda f: abs(results[n]["shap_values"][f])) for n in names}
    top_feature, count = Counter(top_by_model.values()).most_common(1)[0]
    if count == total:
        drivers = f"All {total} models rank <b>{FEATURE_INFO[top_feature]['label']}</b> as the biggest driver."
    else:
        others = [f"{FEATURE_INFO[f]['label']} for {escape(n)}" for n, f in top_by_model.items() if f != top_feature]
        drivers = (f"<b>{FEATURE_INFO[top_feature]['label']}</b> is the biggest driver for {count} of {total} models "
                   f"(the others: {_join(others)}).")

    features = list(proposed["shap_values"])
    all_up = [f for f in features if all(results[n]["shap_values"][f] >= NEGLIGIBLE for n in names)]
    all_down = [f for f in features if all(results[n]["shap_values"][f] <= -NEGLIGIBLE for n in names)]
    agreement = []
    if all_up:
        agreement.append(f"raised the risk: {_join([FEATURE_INFO[f]['label'] for f in all_up])}")
    if all_down:
        agreement.append(f"lowered the risk: {_join([FEATURE_INFO[f]['label'] for f in all_down])}")
    agree = f"Every model agrees on what {'; and what '.join(agreement)}." if agreement else ""

    bullets = [verdicts, spread, drivers] + ([agree] if agree else [])
    if 0 < len(at_risk) < total:
        bullets.append(
            f"Why the verdicts differ: the Stacking Ensemble flags risk from {proposed['threshold'] * 100:.0f}% "
            f"upward (a cut-off tuned to catch more true diabetes cases), while the baseline models use the "
            f"default 50%. Each model also learned different patterns from the training data."
        )
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return f'<div class="interp-card interp-summary"><ul>{items}</ul></div>'


# ============================================
# 4. THE STACKING ENSEMBLE'S META-LEARNER
# ============================================
def meta_learner_interpretation(meta: dict, internals: dict, labels: dict) -> str:
    """An HTML paragraph explaining how the meta-learner combined the base learners for this patient."""
    preds, ref_preds, phi = meta["patient_predictions"], meta["reference_predictions"], meta["shap_values"]
    learners = list(phi)
    weights = {n: sum(c["weights"][j] for c in internals["copies"]) / len(internals["copies"])
               for j, n in enumerate(internals["learners"])}
    raw = sum(c["meta_probability"] for c in internals["copies"]) / len(internals["copies"])
    lowest = min(learners, key=lambda n: preds[n])
    highest = max(learners, key=lambda n: preds[n])

    lead = (
        f"For this patient, the seven base learners estimated the risk between <b>{_pct(preds[lowest])}</b> "
        f"({labels[lowest]}) and <b>{_pct(preds[highest])}</b> ({labels[highest]}), averaged over the 7 calibrated "
        f"copies. The Bagged Logistic Regression meta-learner combined these estimates into <b>{_pct(raw)}</b>, "
        f"and sigmoid calibration adjusted that to the final <b>{_pct(internals['probability'])}</b> shown on the "
        f"result card."
    )

    def describe(n):
        return (f"{labels[n]}: <b>{_pts(phi[n])}</b> (its estimate {_pct(preds[n])} vs. {_pct(ref_preds[n])} for "
                f"the average reference patient; meta-learner weight {_signed(weights[n], 2)})")

    up = sorted([n for n in learners if phi[n] >= NEGLIGIBLE], key=lambda n: -phi[n])
    down = sorted([n for n in learners if phi[n] <= -NEGLIGIBLE], key=lambda n: phi[n])
    small = [n for n in learners if abs(phi[n]) < NEGLIGIBLE]

    bullets = [
        f"Measured from the base risk of {_pct(meta['base_value'])}, the base learners together moved the final "
        f"probability by <b>{_pts(meta['probability'] - meta['base_value'])}</b>."
    ]
    if up:
        bullets.append(f"<b>Pushed the final risk up:</b>{_sublist([describe(n) for n in up])}")
    if down:
        bullets.append(f"<b>Pulled the final risk down:</b>{_sublist([describe(n) for n in down])}")
    if small:
        bullets.append(f"<b>Little or no influence</b> (under 0.5 points each): {_join([labels[n] for n in small])}.")
    top = max(learners, key=lambda n: abs(phi[n]))
    bullets.append(
        f"The meta-learner relied most on <b>{labels[top]}</b> for this patient. A base learner's influence "
        f"depends on both its weight and how far its estimate is from what it predicts for the average "
        f"reference patient."
    )
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return f'<div class="interp-card"><p>{lead}</p><ul>{items}</ul></div>'
