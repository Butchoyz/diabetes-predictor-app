import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# PREMIUM BUBBLES & AURORA BACKGROUND CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ============================================ */
/* 1. AURORA MOTION BACKGROUND                  */
/* ============================================ */
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 35%, #fdf2f8 70%, #e0f2fe 100%) !important;
    background-size: 300% 300% !important;
    animation: auroraFlow 15s ease-in-out infinite !important;
}

[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
    z-index: 10; 
}
[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* ============================================ */
/* 2. MAIN TITLE SHIMMER                        */
/* ============================================ */
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 1rem 0 0.5rem;
    letter-spacing: -1px;
    animation: textShimmer 6s linear infinite;
    position: relative;
    z-index: 10;
}

/* ============================================ */
/* 3. PREMIUM GLASSMORPHISM CARD STYLES         */
/* ============================================ */
.stExpander {
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(20px) saturate(120%);
    -webkit-backdrop-filter: blur(20px) saturate(120%);
    border-radius: 20px !important;
    margin-bottom: 2rem !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    box-shadow: 0 10px 40px -10px rgba(31, 38, 135, 0.1), inset 0 1px 0 rgba(255,255,255,1) !important;
    position: relative;
    z-index: 10;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.stExpander:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 45px -10px rgba(31, 38, 135, 0.15), inset 0 1px 0 rgba(255,255,255,1) !important;
}
.stExpander > div:first-child {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.9) 0%, rgba(124, 58, 237, 0.9) 100%) !important;
    color: white !important;
    padding: 1.2rem !important;
    border-radius: 20px 20px 0 0 !important;
    font-weight: 600 !important;
}

.glass-card {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.9);
    box-shadow: 0 8px 30px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,1);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: transform 0.3s ease;
    position: relative;
    z-index: 10;
}
.glass-card:hover {
    transform: translateY(-2px);
}

/* Inputs */
div[data-baseweb="input"] > div {
    background-color: rgba(255, 255, 255, 0.8) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    padding: 6px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
}
div[data-baseweb="input"] > div:hover {
    border-color: #818cf8 !important;
    background-color: rgba(255, 255, 255, 0.95) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08), inset 0 2px 4px rgba(0,0,0,0.01) !important;
}
div[data-baseweb="input"]:focus-within > div {
    border-color: #6366f1 !important;
    background-color: #ffffff !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
    transform: translateY(-1px);
}
.stTextInput label p {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b !important;
    margin-bottom: 6px;
}

/* Glow Button */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    background-size: 200% auto;
    color: white;
    font-size: 1.15rem;
    font-weight: 700;
    padding: 1rem 2rem;
    border-radius: 16px;
    border: none;
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.4);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    letter-spacing: 0.5px;
    position: relative;
    z-index: 10;
}
div.stButton > button:first-child:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 15px 35px rgba(99, 102, 241, 0.45), 0 0 20px rgba(219, 39, 119, 0.3);
    background-position: right center;
}

/* Animations */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(40px) scale(0.97); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes auroraFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes textShimmer {
    0% { background-position: 0% center; }
    50% { background-position: 100% center; }
    100% { background-position: 0% center; }
}
@keyframes floatIcon {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-8px) rotate(2deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}
</style>
""", unsafe_allow_html=True)

# ============================================
# MORE ANIMATED BUBBLES BACKGROUND (PURE CSS)
# ============================================
st.markdown("""
<style>
    .bubbles-container {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: 0; 
        overflow: hidden; 
        pointer-events: none;
    }
    .glass-bubble {
        position: absolute;
        bottom: -150px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        box-shadow: 0 8px 32px rgba(255, 255, 255, 0.1), inset 0 2px 10px rgba(255,255,255,0.2);
        animation: floatUp infinite ease-in-out;
    }
    
    /* 15 Dynamic Bubbles for richer background */
    .glass-bubble:nth-child(1) { left: 10%; width: 60px; height: 60px; animation-duration: 12s; animation-delay: 0s; }
    .glass-bubble:nth-child(2) { left: 30%; width: 100px; height: 100px; animation-duration: 18s; animation-delay: 2s; background: rgba(124, 58, 237, 0.05); }
    .glass-bubble:nth-child(3) { left: 55%; width: 40px; height: 40px; animation-duration: 10s; animation-delay: 5s; }
    .glass-bubble:nth-child(4) { left: 75%; width: 80px; height: 80px; animation-duration: 22s; animation-delay: 1s; background: rgba(79, 70, 229, 0.05); }
    .glass-bubble:nth-child(5) { left: 85%; width: 50px; height: 50px; animation-duration: 15s; animation-delay: 4s; }
    .glass-bubble:nth-child(6) { left: 20%; width: 120px; height: 120px; animation-duration: 25s; animation-delay: 8s; background: rgba(219, 39, 119, 0.05); }
    .glass-bubble:nth-child(7) { left: 65%; width: 70px; height: 70px; animation-duration: 19s; animation-delay: 6s; }
    .glass-bubble:nth-child(8) { left: 5%; width: 45px; height: 45px; animation-duration: 14s; animation-delay: 3s; }
    .glass-bubble:nth-child(9) { left: 45%; width: 90px; height: 90px; animation-duration: 21s; animation-delay: 7s; background: rgba(124, 58, 237, 0.04); }
    .glass-bubble:nth-child(10) { left: 95%; width: 65px; height: 65px; animation-duration: 17s; animation-delay: 2s; }
    .glass-bubble:nth-child(11) { left: 35%; width: 55px; height: 55px; animation-duration: 16s; animation-delay: 9s; background: rgba(79, 70, 229, 0.06); }
    .glass-bubble:nth-child(12) { left: 80%; width: 110px; height: 110px; animation-duration: 26s; animation-delay: 4s; background: rgba(219, 39, 119, 0.03); }
    .glass-bubble:nth-child(13) { left: 15%; width: 35px; height: 35px; animation-duration: 11s; animation-delay: 6s; }
    .glass-bubble:nth-child(14) { left: 50%; width: 75px; height: 75px; animation-duration: 20s; animation-delay: 1s; }
    .glass-bubble:nth-child(15) { left: 70%; width: 50px; height: 50px; animation-duration: 13s; animation-delay: 5s; }

    @keyframes floatUp {
        0% { transform: translateY(0) scale(1) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 0.8; }
        100% { transform: translateY(-120vh) scale(1.2) rotate(360deg); opacity: 0; }
    }
</style>
<div class="bubbles-container">
    <div class="glass-bubble"></div><div class="glass-bubble"></div>
    <div class="glass-bubble"></div><div class="glass-bubble"></div>
    <div class="glass-bubble"></div><div class="glass-bubble"></div>
    <div class="glass-bubble"></div><div class="glass-bubble"></div>
    <div class="glass-bubble"></div><div class="glass-bubble"></div>
    <div class="glass-bubble"></div><div class="glass-bubble"></div>
    <div class="glass-bubble"></div><div class="glass-bubble"></div>
    <div class="glass-bubble"></div>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL & ASSETS
# ============================================
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.joblib")
    medians = joblib.load("imputation_medians.joblib")
    threshold = joblib.load("optimal_threshold.joblib")
    return model, medians, threshold

model, imputation_medians, threshold = load_model()

def preprocess_input(df, medians):
    for col, median_val in medians.items():
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(median_val, inplace=True)
    return df

# ============================================
# HEADER WITH PREMIUM SHIELD ICON
# ============================================
st.markdown("""
<div style="text-align: center; padding: 2.5rem 0 1rem; position: relative; z-index: 10;">
    <div style="
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100px;
        height: 100px;
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 2px solid rgba(99, 102, 241, 0.35);
        border-radius: 50%;
        box-shadow: 0 12px 30px -5px rgba(99, 102, 241, 0.25), inset 0 2px 5px rgba(255,255,255,0.5);
        animation: floatIcon 4s ease-in-out infinite, slideUp 0.8s ease;
        margin-bottom: 0.5rem;
    ">
        <span style="font-size: 3.5rem;">🏥</span>
    </div>
    <div class="main-title">Diabetes Predictor</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# USER GUIDE EXPANDER (FIXED RENDERING BUG)
# ============================================
st.markdown("""
<style>
div[data-testid="stExpander"] summary * {
    color: #4F46E5 !important;
    font-weight: 700;
    font-size: 1.05rem;
}
div[data-testid="stExpander"] summary svg {
    fill: #4F46E5 !important;
    color: #4F46E5 !important;
}
</style>
""", unsafe_allow_html=True)

with st.expander("📖 Click here for instructions and measurement guide", expanded=False):
    st.markdown("""
<div class="glass-card" style="border-left: 5px solid #6366F1; margin-top: 1.2rem;">
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem;">
<span style="font-size: 1.5rem;">🔬</span>
<h3 style="color: #1F2937; margin: 0; font-weight: 800; font-size: 1.2rem;">What This App Does</h3>
</div>
<p style="color: #4B5563; font-size: 0.95rem; line-height: 1.6; margin: 0; font-weight: 500;">
This intelligent screening tool predicts diabetes risk using 8 medical measurements. 
Powered by machine learning trained on the Pima Indians Diabetes dataset.
</p>
</div>

<div class="glass-card" style="background: rgba(254, 242, 242, 0.65); border-left: 5px solid #EF4444; border-color: rgba(254, 202, 202, 0.9);">
<strong style="color: #EF4444;">⚠️ Important:</strong>
<span style="color: #7F1D1D; font-weight: 500;">This is a <strong>screening tool</strong>, not a medical diagnosis.
Always consult healthcare professionals for proper medical advice.</span>
</div>

<div class="glass-card">
<h3 style="color: #4F46E5; margin-bottom: 1.5rem; font-weight: 800; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">
📋 How to Use This Tool
</h3>
<div style="display: flex; flex-direction: column; gap: 1rem;">
<div style="display: flex; align-items: flex-start; gap: 12px; width: 100%;">
<div style="background: rgba(255, 255, 255, 0.9); color: #4F46E5; min-width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.9rem; border: 1px solid #C7D2FE; box-shadow: 0 2px 5px rgba(0,0,0,0.05); flex-shrink: 0;">1</div>
<div style="color: #374151; font-size: 0.95rem; line-height: 1.5; padding-top: 4px; font-weight: 500; flex-grow: 1;">
Enter all available patient measurements in the fields below.
</div>
</div>
<div style="display: flex; align-items: flex-start; gap: 12px; width: 100%;">
<div style="background: rgba(255, 255, 255, 0.9); color: #4F46E5; min-width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.9rem; border: 1px solid #C7D2FE; box-shadow: 0 2px 5px rgba(0,0,0,0.05); flex-shrink: 0;">2</div>
<div style="color: #374151; font-size: 0.95rem; line-height: 1.6; padding: 12px; background: rgba(255, 255, 255, 0.65); border-radius: 12px; border: 1px dashed rgba(99, 102, 241, 0.4); margin: 4px 0; width: 100%; box-sizing: border-box; flex-grow: 1;">
<strong>✓ Missing Data:</strong> If a measurement is <strong>unavailable</strong>, leave it as <strong>0</strong> or empty. 
The app will automatically handle it and fill in the standard <strong>median values</strong> for:
<span style="color: #4F46E5; font-weight: 700; display: block; margin-top: 6px; font-size: 0.9rem;">
• Glucose • Blood Pressure • Skin Thickness • Insulin • BMI
</span>
</div>
</div>
<div style="display: flex; align-items: flex-start; gap: 12px; width: 100%;">
<div style="background: rgba(255, 255, 255, 0.9); color: #4F46E5; min-width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.9rem; border: 1px solid #C7D2FE; box-shadow: 0 2px 5px rgba(0,0,0,0.05); flex-shrink: 0;">3</div>
<div style="color: #374151; font-size: 0.95rem; line-height: 1.5; padding-top: 4px; font-weight: 500; flex-grow: 1;">
Click <strong>"Analyze Sample"</strong> to get instant prediction results.
</div>
</div>
</div>
</div>

<div class="glass-card">
<h3 style="color: #4F46E5; margin-bottom: 1.25rem; font-weight: 800; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">
📊 Required Measurements & Units
</h3>
<div style="overflow-x: auto; background: rgba(255, 255, 255, 0.7); border: 1px solid rgba(255,255,255,0.9); border-radius: 12px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02); width: 100%;">
<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; min-width: 500px;">
<thead>
<tr style="background: rgba(241, 245, 249, 0.8); border-bottom: 2px solid rgba(226, 232, 240, 0.8);">
<th style="padding: 14px 16px; text-align: left; font-weight: 700; color: #1F2937; white-space: nowrap;">Field</th>
<th style="padding: 14px 16px; text-align: left; font-weight: 700; color: #1F2937;">Description</th>
<th style="padding: 14px 16px; text-align: left; font-weight: 700; color: #1F2937;">Units</th>
</tr>
</thead>
<tbody style="color: #4B5563; font-weight: 500;">
<tr style="border-bottom: 1px solid rgba(226, 232, 240, 0.5);">
<td style="padding: 12px 16px; font-weight: 700; color: #111827; white-space: nowrap;">🤰 Pregnancies</td>
<td style="padding: 12px 16px;">Number of times pregnant</td>
<td style="padding: 12px 16px;">Count</td>
</tr>
<tr style="background: rgba(248, 250, 252, 0.5); border-bottom: 1px solid rgba(226, 232, 240, 0.5);">
<td style="padding: 12px 16px; font-weight: 700; color: #111827; white-space: nowrap;">🩸 Glucose</td>
<td style="padding: 12px 16px;">Blood sugar (2-hour oral test)</td>
<td style="padding: 12px 16px;">mg/dL</td>
</tr>
<tr style="border-bottom: 1px solid rgba(226, 232, 240, 0.5);">
<td style="padding: 12px 16px; font-weight: 700; color: #111827; white-space: nowrap;">❤️ Blood Pressure</td>
<td style="padding: 12px 16px;">Diastolic blood pressure</td>
<td style="padding: 12px 16px;">mm Hg</td>
</tr>
<tr style="background: rgba(248, 250, 252, 0.5); border-bottom: 1px solid rgba(226, 232, 240, 0.5);">
<td style="padding: 12px 16px; font-weight: 700; color: #111827; white-space: nowrap;">📏 Skin Thickness</td>
<td style="padding: 12px 16px;">Triceps skin fold thickness</td>
<td style="padding: 12px 16px;">mm</td>
</tr>
<tr style="border-bottom: 1px solid rgba(226, 232, 240, 0.5);">
<td style="padding: 12px 16px; font-weight: 700; color: #111827; white-space: nowrap;">💉 Insulin</td>
<td style="padding: 12px 16px;">Serum insulin after 2 hours</td>
<td style="padding: 12px 16px;">μU/mL</td>
</tr>
<tr style="background: rgba(248, 250, 252, 0.5); border-bottom: 1px solid rgba(226, 232, 240, 0.5);">
<td style="padding: 12px 16px; font-weight: 700; color: #111827; white-space: nowrap;">⚖️ BMI</td>
<td style="padding: 12px 16px;">Body Mass Index</td>
<td style="padding: 12px 16px;">kg/m²</td>
</tr>
<tr style="border-bottom: 1px solid rgba(226, 232, 240, 0.5);">
<td style="padding: 12px 16px; font-weight: 700; color: #111827; white-space: nowrap;">👨‍👩‍👧 DPF Score</td>
<td style="padding: 12px 16px;">Diabetes pedigree function</td>
<td style="padding: 12px 16px;">Score</td>
</tr>
<tr style="background: rgba(248, 250, 252, 0.5);">
<td style="padding: 12px 16px; font-weight: 700; color: #111827; white-space: nowrap;">🎂 Age</td>
<td style="padding: 12px 16px;">Patient's age</td>
<td style="padding: 12px 16px;">Years</td>
</tr>
</tbody>
</table>
</div>
</div>

<div class="glass-card" style="background: rgba(240, 253, 244, 0.65); border-left: 5px solid #10B981; border-color: rgba(220, 252, 231, 0.9);">
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
<span style="font-size: 1.25rem;">💡</span>
<h4 style="color: #166534; margin: 0; font-weight: 800; font-size: 1.05rem;">
Tips for Accurate Predictions
</h4>
</div>
<ul style="color: #15803D; font-size: 0.95rem; line-height: 1.6; margin: 0; padding-left: 1.2rem; list-style-type: none;">
<li style="margin-bottom: 0.75rem; display: flex; align-items: flex-start; gap: 8px;">
<span style="color: #22C55E; font-weight: 900;">✓</span>
<span style="font-weight: 500;"><strong>Fill All Fields:</strong> Enter all information for the most accurate prediction.</span>
</li>
<li style="margin-bottom: 0.75rem; display: flex; align-items: flex-start; gap: 8px;">
<span style="color: #22C55E; font-weight: 900;">✓</span>
<span style="font-weight: 500;"><strong>Missing Data:</strong> Leave as <strong>0</strong> if unavailable – the app will fill in median values.</span>
</li>
<li style="margin-bottom: 0.75rem; display: flex; align-items: flex-start; gap: 8px;">
<span style="color: #22C55E; font-weight: 900;">✓</span>
<span style="font-weight: 500;"><strong>Units Matter:</strong> Ensure all values are in the specified units (mg/dL, mm Hg, etc).</span>
</li>
<li style="margin-top: 1rem; width: 100%;">
<!-- FULLY RESPONSIVE BMI FORMULA CONTAINER -->
<div style="background: rgba(255, 255, 255, 0.9); border: 1px solid #86EFAC; padding: 0.75rem 1rem; border-radius: 8px; display: block; width: 100%; max-width: 320px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); box-sizing: border-box;">
<strong style="color: #166534; font-size: 0.85rem; letter-spacing: 0.5px;">BMI FORMULA:</strong><br>
<code style="font-family: monospace; color: #047857; font-size: 0.95rem; font-weight: 700; background: transparent; word-break: break-word; display: inline-block; max-width: 100%;">weight(kg) ÷ [height(m)]²</code>
</div>
</li>
</ul>
</div>

<div class="glass-card">
<h4 style="color: #4F46E5; margin-bottom: 1.25rem; font-weight: 800; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">
📊 Understanding Your Result
</h4>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
<div style="background: rgba(240, 253, 244, 0.7); border: 1px solid rgba(187, 247, 208, 0.9); border-radius: 16px; padding: 1.5rem; text-align: center; box-shadow: inset 0 2px 4px rgba(255,255,255,1);">
<div style="background: rgba(255, 255, 255, 0.9); width: 56px; height: 56px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem auto; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
<span style="font-size: 1.5rem;">✅</span>
</div>
<div style="color: #166534; font-weight: 800; font-size: 0.95rem; letter-spacing: 0.5px; margin-bottom: 0.25rem;">NON-DIABETIC</div>
<div style="color: #15803D; font-size: 0.85rem; line-height: 1.4; font-weight: 600;">Low risk profile based on inputs</div>
</div>
<div style="background: rgba(254, 242, 242, 0.7); border: 1px solid rgba(254, 202, 202, 0.9); border-radius: 16px; padding: 1.5rem; text-align: center; box-shadow: inset 0 2px 4px rgba(255,255,255,1);">
<div style="background: rgba(255, 255, 255, 0.9); width: 56px; height: 56px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem auto; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
<span style="font-size: 1.5rem;">⚠️</span>
</div>
<div style="color: #991B1B; font-weight: 800; font-size: 0.95rem; letter-spacing: 0.5px; margin-bottom: 0.25rem;">DIABETIC</div>
<div style="color: #B91C1C; font-size: 0.85rem; line-height: 1.4; font-weight: 600;">High risk profile detected</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# SECTION HEADER & INPUT FIELDS
# ============================================
st.markdown("""
<div style="margin-bottom: 1.5rem; display: flex; align-items: center; gap: 12px; position: relative; z-index: 10;">
    <div style="background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); padding: 10px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.9); box-shadow: 0 4px 15px rgba(0,0,0,0.02);">
        <span style="font-size: 1.3rem;">👤</span>
    </div>
    <div>
        <h4 style="margin: 0; color: #1e293b; font-weight: 800;">Patient Information</h4>
        <p style="margin: 0; font-size: 0.9rem; font-weight: 500; color: #475569;">Enter the diagnostic measurements below</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        pregnancies = st.text_input("🤰 Pregnancies", placeholder="e.g., 2", help="Number of times pregnant (0 if never)")
        glucose = st.text_input("🩸 Glucose", placeholder="e.g., 120", help="Plasma glucose concentration (mg/dL)")
        blood_pressure = st.text_input("❤️ Blood Pressure", placeholder="e.g., 70", help="Diastolic blood pressure (mm Hg)")
        skin_thickness = st.text_input("📏 Skin Thickness", placeholder="e.g., 20", help="Triceps skin fold thickness (mm)")

    with col2:
        insulin = st.text_input("💉 Insulin Level", placeholder="e.g., 85", help="2-Hour serum insulin (μU/mL)")
        bmi = st.text_input("⚖️ BMI Index", placeholder="e.g., 25.5", help="Body mass index (weight in kg / height in m²)")
        dpf = st.text_input("🧬 Diabetes Pedigree *", placeholder="e.g., 0.35", help="Diabetes Pedigree Function (genetic score) [REQUIRED]")
        age = st.text_input("🎂 Age *", placeholder="e.g., 35", help="Age in years [REQUIRED]")

# ============================================
# PREDICT BUTTON
# ============================================
st.markdown("###")
submitted = st.button("🔍 Analyze Sample", type="primary", use_container_width=True)

# ============================================
# PROFESSIONAL RESULTS DISPLAY
# ============================================
if submitted:
    # 🚨 ONLY DPF & AGE ARE ABSOLUTELY REQUIRED TO PREVENT CRASHES 🚨
    if not dpf or not age:
        st.error("⚠️ **Missing Information:** Please fill in the required fields (Diabetes Pedigree and Age).")
        st.stop() 

    try:
        # Imputation defaults to 0.0 if left blank, passing cleanly into your model pipeline
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

        with st.spinner("🔬 Analyzing sample..."):
            processed = preprocess_input(input_df, imputation_medians)
            probability = model.predict_proba(processed)[0, 1]
            prediction = int(probability >= threshold)
            confidence_percent = int(round(probability * 100, 0))

        threshold_pct = int(round(threshold * 100, 0))
        model_conf_pct = confidence_percent
        
        if prediction == 0:
            tier = "Low Risk"
            tier_color = "#10B981"
            result_title = "NON-DIABETIC"
            result_color = "#047857"
            result_bg = "linear-gradient(135deg, rgba(209, 250, 229, 0.75) 0%, rgba(167, 243, 208, 0.75) 100%)"
            result_border = "rgba(255, 255, 255, 0.9)"
            result_icon = "✅"
        else:
            tier = "High Risk" if model_conf_pct >= 50 else "Moderate Risk"
            tier_color = "#EF4444" if model_conf_pct >= 50 else "#F59E0B"
            result_title = "DIABETIC"
            result_color = "#991B1B"
            result_bg = "linear-gradient(135deg, rgba(254, 226, 226, 0.75) 0%, rgba(254, 202, 202, 0.75) 100%)"
            result_border = "rgba(255, 255, 255, 0.9)"
            result_icon = "⚠️"

        result_html = f"""
        <div style="
            background: {result_bg};
            border: 2px solid {result_border};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 28px;
            padding: 3.5rem 2rem;
            margin: 2.5rem 0;
            text-align: center;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.08), inset 0 2px 10px rgba(255,255,255,0.5);
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
            position: relative;
            z-index: 10;
        ">
            <div style="font-size: 5rem; margin-bottom: 0.5rem; text-shadow: 0 10px 20px rgba(0,0,0,0.1);">{result_icon}</div>
            <div style="
                font-size: 3.5rem;
                font-weight: 900;
                color: {result_color};
                letter-spacing: -2px;
                margin-bottom: 0.5rem;
                text-shadow: 0 2px 10px rgba(255,255,255,0.5);
            ">{result_title}</div>
            <div style="
                color: {result_color};
                font-size: 1.15rem;
                font-weight: 700;
                opacity: 0.9;
            ">{"Low risk based on current parameters" if prediction == 0 else "High risk detected – Medical consultation recommended"}</div>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)

        # ============================================
        # RESPONSIVE GAUGE
        # ============================================
        marker_pos = max(5, min(95, model_conf_pct))
        threshold_pos = max(5, min(95, threshold_pct))

        gauge_html = f"""
        <style>
            .risk-card {{
                background: rgba(255, 255, 255, 0.6);
                backdrop-filter: blur(24px);
                -webkit-backdrop-filter: blur(24px);
                border-radius: 24px;
                padding: 2.5rem 2.5rem;
                box-shadow: 0 15px 45px rgba(0,0,0,0.05), inset 0 2px 0 rgba(255,255,255,0.9);
                border: 1px solid rgba(255,255,255,0.9);
                margin-top: 2rem;
                animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
                position: relative;
                z-index: 10;
            }}
            .header {{
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
                flex-wrap: wrap;
                gap: 12px;
                margin-bottom: 2.5rem;
            }}
            .badge {{
                background: rgba(255, 255, 255, 0.9);
                color: {tier_color};
                padding: 0.6rem 1.5rem;
                border-radius: 50px;
                font-weight: 800;
                font-size: 0.85rem;
                border: 1px solid {tier_color}40;
                box-shadow: 0 4px 15px {tier_color}20;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .track-container {{
                position: relative;
                height: 50px;
                margin: 2.5rem 0;
            }}
            .track {{
                position: absolute;
                top: 50%;
                left: 0;
                right: 0;
                height: 16px;
                background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%);
                border-radius: 100px;
                transform: translateY(-50%);
                opacity: 0.3;
                box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
            }}
            .threshold-line {{
                position: absolute;
                left: {threshold_pos}%;
                top: -5px;
                bottom: -5px;
                width: 3px;
                background: #0f172a;
                z-index: 5;
                border-radius: 5px;
            }}
            .threshold-label {{
                position: absolute;
                top: -32px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 0.75rem;
                color: #0f172a;
                font-weight: 800;
                background: #ffffff;
                padding: 4px 10px;
                border-radius: 6px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            }}
            .marker {{
                position: absolute;
                left: {marker_pos}%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: 30px;
                height: 30px;
                background: {tier_color};
                border: 6px solid white;
                border-radius: 50%;
                box-shadow: 0 6px 15px rgba(0,0,0,0.2);
                z-index: 10;
                transition: left 1s cubic-bezier(0.16, 1, 0.3, 1);
            }}
            .marker-label {{
                position: absolute;
                top: 38px;
                left: 50%;
                transform: translateX(-50%);
                background: {tier_color};
                color: white;
                padding: 4px 12px;
                border-radius: 8px;
                font-size: 0.85rem;
                font-weight: 800;
                box-shadow: 0 6px 15px {tier_color}40;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 1.5rem;
                margin-top: 3.5rem;
                padding-top: 2rem;
                border-top: 2px solid rgba(255,255,255,0.8);
            }}
            .stat-box {{
                text-align: center;
                padding: 1.5rem;
                background: rgba(255,255,255,0.7);
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,1);
                box-shadow: 0 8px 20px rgba(0,0,0,0.03);
                transition: transform 0.3s ease;
            }}
            .stat-box:hover {{
                transform: translateY(-3px);
            }}
            .stat-value {{ font-size: 2rem; font-weight: 900; color: #0f172a; letter-spacing: -1px; }}
            .stat-label {{ font-size: 0.85rem; font-weight: 700; color: #64748b; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px; }}
        </style>

        <div class="risk-card">
            <div class="header">
                <div style="font-size: 1.4rem; font-weight: 800; color: #0f172a; letter-spacing: -0.5px;">Analysis & Confidence</div> 
                <div class="badge">{tier}</div>
            </div>
            <div class="track-container">
                <div class="track"></div>
                <div class="threshold-line">
                    <div class="threshold-label">Cutoff {threshold_pct}%</div>
                </div>
                <div class="marker">
                    <div class="marker-label">{model_conf_pct}%</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 800; color: #94a3b8; margin-top: -5px; text-transform: uppercase;">
                <span>0% Safe</span>
                <span>100% Risk</span>
            </div>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value" style="color: {tier_color}">{model_conf_pct}%</div>
                    <div class="stat-label">Model Confidence</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{threshold_pct}%</div>
                    <div class="stat-label">Decision Threshold</div>
                </div>
            </div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ An error occurred while parsing the inputs: {e}")