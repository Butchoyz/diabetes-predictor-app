import streamlit as st
import pandas as pd
import joblib
import numpy as np
from abc import ABC, abstractmethod

# ============================================
# PAGE CONFIG 
# ============================================
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🏥",
    layout="centered" 
)

# ============================================
# 1. CORE PREDICTOR CLASSES
# ============================================
class BasePredictor(ABC):
    def __init__(self, model_path, medians_path, threshold_path=None, scaler_path=None):
        self.model_path = model_path
        self.medians_path = medians_path
        self.threshold_path = threshold_path
        self.scaler_path = scaler_path
        
        self.model = None
        self.medians = None
        self.threshold = 0.50
        self.scaler = None
        self.load_assets()

    def load_assets(self):
        self.model = joblib.load(self.model_path)
        self.medians = joblib.load(self.medians_path)
        if self.threshold_path:
            self.threshold = joblib.load(self.threshold_path)
        if self.scaler_path:
            self.scaler = joblib.load(self.scaler_path)

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def predict(self, df: pd.DataFrame):
        processed_df = self.preprocess(df.copy())
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(processed_df)[0, 1]
        else:
            probability = float(self.model.predict(processed_df)[0])
            
        prediction = int(probability >= self.threshold)
        confidence = int(round(probability * 100, 0))
        return prediction, confidence


class ProposedPredictor(BasePredictor):
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, median_val in self.medians.items():
            df[col] = df[col].replace(0, np.nan)
            df[col].fillna(median_val, inplace=True)
        return df


class BaselinePredictor(BasePredictor):
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, median_val in self.medians.items():
            df[col] = df[col].replace(0, np.nan)
            df[col].fillna(median_val, inplace=True)
        if self.scaler:
            scaled_data = self.scaler.transform(df)
            df = pd.DataFrame(scaled_data, columns=df.columns)
        return df

# ============================================
# MODEL CONFIGURATION & CACHING
# ============================================
BASELINE_MODELS = {
    "Gradient Boosting": "Gradient_Boosting_model.pkl",
    "K-Nearest Neighbors": "K_Nearest_Neighbors_model.pkl",
    "LightGBM": "LightGBM_model.pkl",
    "Logistic Regression": "Logistic_Regression_model.pkl",
    "Naive Bayes": "Naive_Bayes_model.pkl",
    "Random Forest": "Random_Forest_model (1).pkl",
    "Support Vector Machine": "Support_Vector_Machine_model.pkl",
    "XGBoost": "XGBoost_model.pkl"
}

@st.cache_resource
def get_proposed_predictor():
    return ProposedPredictor(
        model_path="diabetes_model.joblib",
        medians_path="imputation_medians.joblib",
        threshold_path="optimal_threshold.joblib"
    )

@st.cache_resource
def get_baseline_predictor(model_name):
    return BaselinePredictor(
        model_path=BASELINE_MODELS[model_name],
        medians_path="imputation_medians.joblib",
        scaler_path="scaler.pkl"
    )

# ============================================
# PREMIUM STYLING & ANIMATED BACKGROUND CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* { font-family: 'Plus Jakarta Sans', sans-serif !important; }
[data-testid="stSidebar"] { display: none !important; }

/* Base Canvas */
.stApp {
    background-color: #f8fafc !important;
    overflow-x: hidden;
}
[data-testid="stHeader"] { background-color: transparent !important; }

/* Floating Animated Orbs in Background */
.bg-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.55;
    pointer-events: none;
    z-index: 0;
}

.orb-1 {
    top: -10%; left: -10%; width: 450px; height: 450px;
    background: radial-gradient(circle, #818cf8 0%, #c084fc 100%);
    animation: floatOrb1 18s ease-in-out infinite alternate;
}

.orb-2 {
    bottom: -10%; right: -10%; width: 500px; height: 500px;
    background: radial-gradient(circle, #f472b6 0%, #38bdf8 100%);
    animation: floatOrb2 22s ease-in-out infinite alternate;
}

.orb-3 {
    top: 40%; left: 35%; width: 380px; height: 380px;
    background: radial-gradient(circle, #a78bfa 0%, #818cf8 100%);
    animation: floatOrb3 16s ease-in-out infinite alternate;
}

/* Animations for Floating Orbs */
@keyframes floatOrb1 {
    0% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(140px, 90px) scale(1.15); }
    100% { transform: translate(-50px, 120px) scale(0.95); }
}

@keyframes floatOrb2 {
    0% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(-120px, -100px) scale(1.2); }
    100% { transform: translate(60px, -80px) scale(0.9); }
}

@keyframes floatOrb3 {
    0% { transform: translate(0px, 0px) scale(0.9); }
    50% { transform: translate(80px, -70px) scale(1.1); }
    100% { transform: translate(-90px, 60px) scale(1); }
}

/* Main Title Styling */
.main-title {
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; padding: 0.5rem 0; letter-spacing: -1px;
    animation: textShimmer 6s linear infinite;
}

/* Cleaner & Larger Reference Pills */
.ref-pill {
    font-size: 0.85rem;
    font-weight: 600;
    color: #475569;
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(8px);
    border-radius: 8px;
    padding: 8px 12px;
    margin-top: -8px;
    margin-bottom: 20px;
    border: 1px solid rgba(203, 213, 225, 0.8);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
}

/* Fix Tooltip for Mobile */
div[data-baseweb="tooltip"], .stTooltipIcon { 
    z-index: 999999 !important; 
    pointer-events: auto !important;
}
.stTooltipIcon { color: #4f46e5 !important; }

/* Custom Input Box Styling */
div[data-baseweb="input"] > div {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(203, 213, 225, 0.8) !important;
}

/* Main Button */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    color: white; font-size: 1.1rem; font-weight: 700; padding: 0.9rem 2rem;
    border-radius: 14px; border: none; box-shadow: 0 10px 20px rgba(79, 70, 229, 0.25);
    margin-top: 1rem;
}

/* ========================================================
   CSS BREAKOUT HACK: Makes ONLY the results section wide!
   ======================================================== */
.results-wrapper {
    width: 92vw;
    max-width: 1400px;
    position: relative;
    left: 50%;
    transform: translateX(-50%); /* Centers the wide div */
    z-index: 10;
    margin-top: 1rem;
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
}

.hero-grid-item {
    grid-column: 1 / -1;
    margin-bottom: 10px; /* Dagdag space bago mag baseline models */
}

/* Responsive Rules for Results Grid */
@media (max-width: 1200px) {
    .results-grid { grid-template-columns: repeat(2, 1fr); }
    .results-wrapper { width: 95vw; }
}

@media (max-width: 768px) {
    .results-grid { grid-template-columns: 1fr; }
    /* Reset layout wrapper on small phones so it doesn't overflow */
    .results-wrapper { width: 100%; left: 0; transform: none; }
}

@keyframes textShimmer { 0% { background-position: 0% center; } 50% { background-position: 100% center; } 100% { background-position: 0% center; } }
@keyframes floatIcon { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-8px); } }
</style>

<!-- BACKGROUND ANIMATION ORBS -->
<div class="bg-orb orb-1"></div>
<div class="bg-orb orb-2"></div>
<div class="bg-orb orb-3"></div>
""", unsafe_allow_html=True)

# ============================================
# APP TITLE
# ============================================
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1.5rem; position: relative; z-index: 10;">
<div style="display: inline-flex; width: 90px; height: 90px; background: rgba(255, 255, 255, 0.5); backdrop-filter: blur(12px); border: 2px solid rgba(99, 102, 241, 0.35); border-radius: 50%; box-shadow: 0 12px 30px -5px rgba(99, 102, 241, 0.25); align-items: center; justify-content: center; animation: floatIcon 4s ease-in-out infinite;">
<span style="font-size: 3.2rem;">🏥</span>
</div>
<div class="main-title">Diabetes Predictor</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# UI/UX DISCLAIMER / SCREENING NOTICE
# ============================================
st.markdown("""
<div style="background: rgba(255, 251, 235, 0.7); backdrop-filter: blur(12px); padding: 1rem 1.25rem; border-radius: 12px; border: 1px solid rgba(253, 230, 138, 0.9); margin-bottom: 2rem; display: flex; gap: 14px; align-items: flex-start; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03); position: relative; z-index: 10;">
    <div style="font-size: 1.4rem; margin-top: 2px;">🛡️</div>
    <div>
        <h4 style="margin: 0 0 4px 0; color: #92400e; font-weight: 800; font-size: 0.95rem;">Screening Tool Notice</h4>
        <p style="margin: 0; color: #b45309; font-size: 0.85rem; line-height: 1.6; font-weight: 500;">This application is designed strictly as an initial <b>risk assessment and screening tool</b>. It does not provide definitive medical diagnoses. Always consult with a licensed healthcare professional for proper medical evaluation, advice, and treatment.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# CLEAN HEADER FOR DIAGNOSTIC PROFILE
# ============================================
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.65); backdrop-filter: blur(16px); padding: 1.25rem; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.9); margin-bottom: 2rem; box-shadow: 0 8px 30px rgba(0,0,0,0.03); position: relative; z-index: 10;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <div style="background: #4f46e5; color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; box-shadow: 0 4px 10px rgba(79,70,229,0.3);">📋</div>
        <h3 style="margin: 0; color: #0f172a; font-weight: 800; font-size: 1.4rem; letter-spacing: -0.5px;">Patient Diagnostic Profile</h3>
    </div>
    <p style="margin: 0; color: #475569; font-size: 0.9rem; line-height: 1.5;">Please enter the patient's medical details below. Tap the <b>(?)</b> icon next to each field to understand the clinical context and risk factors.</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# INPUT FIELDS (2-COLUMN GRID WITH REFS)
# ============================================
col1, col2 = st.columns(2, gap="medium")

with col1:
    pregnancies = st.text_input(
        "🤰 Pregnancies", 
        placeholder="e.g., 2",
        help="What to input:\nNumber of times the patient has been pregnant (Enter 0 if male or never pregnant).\n\nWhy it's a factor:\nPregnancy causes hormonal changes that can lead to temporary insulin resistance (Gestational Diabetes). This increases the chances of getting Type 2 Diabetes later in life."
    )
    st.markdown('<div class="ref-pill"><span style="color: #059669;">🟢 Normal: 0–2</span><span style="color: #e11d48;">🔴 Risk: ≥ 3</span></div>', unsafe_allow_html=True)

    glucose = st.text_input(
        "🩸 Plasma Glucose (mg/dL)", 
        placeholder="e.g., 120",
        help="What to input:\nBlood sugar level from a Fasting Glucose or 2-hour Oral Glucose Tolerance Test (in mg/dL).\n\nWhy it's a factor:\nThis is the main indicator of diabetes. High blood sugar means the body isn't making enough insulin or isn't using it properly."
    )
    st.markdown('<div class="ref-pill"><span style="color: #059669;">🟢 Normal: 70–139</span><span style="color: #e11d48;">🔴 Risk: ≥ 140</span></div>', unsafe_allow_html=True)

    blood_pressure = st.text_input(
        "❤️ Blood Pressure (mm Hg)", 
        placeholder="e.g., 72",
        help="What to input:\nDiastolic Blood Pressure, which is the lower number in your blood pressure reading (in mm Hg).\n\nWhy it's a factor:\nHigh blood pressure and diabetes are closely related. High BP damages blood vessels and makes insulin resistance worse."
    )
    st.markdown('<div class="ref-pill"><span style="color: #059669;">🟢 Normal: 60–80</span><span style="color: #e11d48;">🔴 Risk: ≥ 90</span></div>', unsafe_allow_html=True)

    skin_thickness = st.text_input(
        "📏 Skin Thickness (mm)", 
        placeholder="e.g., 20",
        help="What to input:\nTriceps skin fold thickness measured using a caliper tool (in mm).\n\nWhy it's a factor:\nThis is used to estimate body fat. Having too much body fat is directly linked to insulin resistance and poor blood sugar control."
    )
    st.markdown('<div class="ref-pill"><span style="color: #059669;">🟢 Normal: 10–20</span><span style="color: #e11d48;">🔴 Risk: > 25</span></div>', unsafe_allow_html=True)

with col2:
    insulin = st.text_input(
        "💉 Serum Insulin (μU/mL)", 
        placeholder="e.g., 85",
        help="What to input:\n2-Hour Serum Insulin level after consuming glucose (in μU/mL).\n\nWhy it's a factor:\nVery high insulin levels show that the body is working extra hard because the cells are ignoring the insulin. This is known as Insulin Resistance."
    )
    st.markdown('<div class="ref-pill"><span style="color: #059669;">🟢 Normal: 16–166</span><span style="color: #e11d48;">🔴 Risk: > 166</span></div>', unsafe_allow_html=True)

    bmi = st.text_input(
        "⚖️ BMI Index (kg/m²)", 
        placeholder="e.g., 25.5",
        help="What to input:\nBody Mass Index, which is your weight in kilograms divided by your height in meters squared.\n\nWhy it's a factor:\nExcess body weight makes it harder for the body to use insulin properly, causing sugar to build up in the blood instead of going into the cells."
    )
    st.markdown('<div class="ref-pill"><span style="color: #059669;">🟢 Normal: 18.5–24.9</span><span style="color: #e11d48;">🔴 Risk: ≥ 25.0</span></div>', unsafe_allow_html=True)

    dpf = st.text_input(
        "🧬 Diabetes Pedigree *", 
        placeholder="e.g., 0.35",
        help="What to input:\n[REQUIRED] A genetic score based on your family's history of diabetes (usually ranges from 0.08 to 2.42).\n\nWhy it's a factor:\nDiabetes is strongly connected to genetics. Having parents or close relatives with diabetes significantly increases your own genetic risk."
    )
    st.markdown('<div class="ref-pill"><span style="color: #059669;">🟢 Normal: < 0.50</span><span style="color: #e11d48;">🔴 Risk: ≥ 0.50</span></div>', unsafe_allow_html=True)

    age = st.text_input(
        "🎂 Age (Years) *", 
        placeholder="e.g., 35",
        help="What to input:\n[REQUIRED] The patient's current age in years.\n\nWhy it's a factor:\nAs we get older, our pancreas naturally produces less insulin and we tend to become less physically active, both of which increase the risk of diabetes."
    )
    st.markdown('<div class="ref-pill"><span style="color: #059669;">🟢 Normal: < 35 yrs</span><span style="color: #e11d48;">🔴 Risk: ≥ 35 yrs</span></div>', unsafe_allow_html=True)

submitted = st.button("🔍 Analyze Sample Across Models", type="primary", use_container_width=True)

# ============================================
# HELPER: HTML RESULT CARD GENERATOR
# ============================================
def create_result_card(model_name, prediction, confidence, threshold, is_proposed=False):
    if prediction == 0:
        status_text = "NON-DIABETIC"
        status_color = "#166534"
        status_bg = "#dcfce7"
        status_border = "#bbf7d0"
        fill_color = "linear-gradient(90deg, #34d399 0%, #10b981 100%)"
        tier = "Low Risk"
        tier_color = "#10b981"
    else:
        status_text = "DIABETIC RISK"
        status_color = "#991b1b"
        status_bg = "#fee2e2"
        status_border = "#fecaca"
        fill_color = "linear-gradient(90deg, #f87171 0%, #ef4444 100%)"
        tier = "High Risk" if confidence >= 50 else "Moderate Risk"
        tier_color = "#ef4444" if confidence >= 50 else "#f59e0b"
        
    if is_proposed:
        # Pinalitan ko yung inline style para magka-max-width na 700px at mag-center (margin: 0 auto)
        card_style = "background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(12px); border: 2px solid #818cf8; border-radius: 20px; padding: 1.5rem; box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.2); width: 100%; max-width: 700px; margin: 0 auto;"
        badge_label = "⭐ PROPOSED STACKING ENSEMBLE"
        badge_color = "#4f46e5"
        wrapper_class = "hero-grid-item"
    else:
        card_style = "background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 16px; padding: 1.25rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);"
        badge_label = "📊 BASELINE MODEL"
        badge_color = "#64748b"
        wrapper_class = ""

    # Note: No indentation used in HTML to prevent Streamlit Markdown errors
    return f"""<div class="{wrapper_class}">
<div style="{card_style} height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 12px;">
<div>
<div style="font-size: 0.75rem; font-weight: 800; color: {badge_color}; letter-spacing: 0.8px; margin-bottom: 0.25rem;">{badge_label}</div>
<div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; line-height: 1.2;">{model_name}</div>
</div>
<div style="font-size: 0.7rem; font-weight: 800; color: {status_color}; background: {status_bg}; border: 1px solid {status_border}; padding: 4px 10px; border-radius: 9999px; display: inline-flex; align-items: center;">
{status_text}
</div>
</div>
<div style="margin-bottom: 1.5rem;">
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 0.5rem;">
<span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Risk Probability</span>
<span style="font-size: 1.2rem; color: #0f172a; font-weight: 800; line-height: 1;">{confidence}%</span>
</div>
<div style="position: relative; width: 100%; height: 8px; background: #f1f5f9; border-radius: 9999px; overflow: visible;">
<div style="position: absolute; left: 0; top: 0; height: 100%; width: {confidence}%; background: {fill_color}; border-radius: 9999px; transition: width 0.5s ease-in-out; z-index: 1;"></div>
<div style="position: absolute; left: {threshold}%; top: -4px; height: 16px; width: 2px; background: #cbd5e1; border-radius: 2px; z-index: 2;"></div>
<div style="position: absolute; left: {threshold}%; top: 16px; transform: translateX(-50%); font-size: 0.65rem; color: #94a3b8; font-weight: 600; white-space: nowrap;">Cut-off: {threshold}%</div>
</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #f1f5f9; padding-top: 0.85rem; font-size: 0.85rem;">
<span style="color: #64748b; font-weight: 500;">Calculated Risk Level</span> 
<span style="color: {tier_color}; font-weight: 700; display: flex; align-items: center; gap: 4px;">
<div style="width: 8px; height: 8px; border-radius: 50%; background-color: {tier_color};"></div>
{tier}
</span>
</div>
</div>
</div>"""

# ============================================
# RESULTS PROCESSOR
# ============================================
if submitted:
    if not dpf or not age:
        st.error("⚠️ **Missing Input:** Please provide the Diabetes Pedigree Function and Age.")
        st.stop() 

    try:
        input_values = {
            'Pregnancies': int(pregnancies) if pregnancies else 0,
            'Glucose': float(glucose) if glucose else 0.0,
            'BloodPressure': float(blood_pressure) if blood_pressure else 0.0,
            'SkinThickness': float(skin_thickness) if skin_thickness else 0.0,
            'Insulin': float(insulin) if insulin else 0.0,
            'BMI': float(bmi) if bmi else 0.0,
            'DiabetesPedigreeFunction': float(dpf),
            'Age': int(age)
        }
        input_df = pd.DataFrame([input_values])

        st.markdown("<h3 style='text-align: center; color: #0f172a; margin: 3rem 0 0rem; font-weight: 800; letter-spacing: -0.5px; position: relative; z-index: 10;'>📊 Diagnostic Report & Models</h3>", unsafe_allow_html=True)
        
        with st.spinner("🔬 Running advanced prediction models..."):
            # WRAPPED IN results-wrapper TO BREAK OUT OF CENTERED LAYOUT
            results_html = "<div class='results-wrapper'><div class='results-grid'>"
            
            proposed_pred = get_proposed_predictor()
            p_val, p_conf = proposed_pred.predict(input_df)
            p_thr = int(round(proposed_pred.threshold * 100, 0))
            results_html += create_result_card("Stacking Ensemble Model", p_val, p_conf, p_thr, is_proposed=True)
            
            for model_name in BASELINE_MODELS.keys():
                base_pred = get_baseline_predictor(model_name)
                b_val, b_conf = base_pred.predict(input_df)
                b_thr = int(round(base_pred.threshold * 100, 0))
                results_html += create_result_card(model_name, b_val, b_conf, b_thr, is_proposed=False)
                
            results_html += "</div></div>"
            
            st.markdown(results_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error Processing Inputs: {e}")