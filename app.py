
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
# MODERN CUSTOM CSS
# ============================================
st.markdown("""
    <style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2.5rem 0 1rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Input fields container */
    .input-container {
        background: #ffffff;
        padding: 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
        transition: all 0.2s ease;
    }
    
    .input-container:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Individual input styling */
    .stTextInput > div > div > input {
        background: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Result card styling */
    .result-card {
        padding: 3rem 2rem;
        border-radius: 16px;
        margin: 2.5rem 0;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .diabetic {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 2px solid #EF4444;
    }
    
    .non-diabetic {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border: 2px solid #10B981;
    }
    
    .diagnosis-text {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1rem;
        color: #4B5563;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 16px 24px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6B46C1 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Error message */
    .error-box {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Loading spinner */
    .spinner-text {
        color: #4B5563;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #9CA3AF;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #E5E7EB;
        font-size: 0.875rem;
    }
    </style>
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

# ============================================
# PREPROCESS FUNCTION
# ============================================
def preprocess_input(df, medians):
    for col, median_val in medians.items():
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(median_val, inplace=True)
    return df

# ============================================
# HEADER
# ============================================
st.markdown('<div class="main-title">🏥 Diabetes Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient details for diagnosis</div>', unsafe_allow_html=True)

# ============================================
# INPUT FIELDS
# ============================================


col1, col2 = st.columns(2, gap="large")

with col1:
    pregnancies = st.text_input("🤰 Pregnancies", placeholder="0")
    glucose = st.text_input("🩸 Glucose (mg/dL)", placeholder="0")
    blood_pressure = st.text_input("❤️ Blood Pressure (mm Hg)", placeholder="0")
    skin_thickness = st.text_input("📏 Skin Thickness (mm)", placeholder="0")

with col2:
    insulin = st.text_input("💉 Insulin (μU/mL)", placeholder="0")
    bmi = st.text_input("⚖️ BMI (kg/m²)", placeholder="0.0")
    dpf = st.text_input("👨‍👩‍👧 DPF Score", placeholder="0.0")
    age = st.text_input("🎂 Age (years)", placeholder="0")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICT BUTTON
# ============================================
submitted = st.button("🔍 Analyze Sample", type="primary", use_container_width=True)

# ============================================
# RESULTS
# ============================================
if submitted:
    try:
        # Convert inputs
        input_values = {
            'Pregnancies': int(pregnancies) if pregnancies else 0,
            'Glucose': float(glucose) if glucose else 0.0,
            'BloodPressure': float(blood_pressure) if blood_pressure else 0.0,
            'SkinThickness': float(skin_thickness) if skin_thickness else 0.0,
            'Insulin': float(insulin) if insulin else 0.0,
            'BMI': float(bmi) if bmi else 0.0,
            'DiabetesPedigreeFunction': float(dpf) if dpf else 0.0,
            'Age': int(age) if age else 0
        }
        
        input_df = pd.DataFrame([input_values])
        
        with st.spinner("Analyzing sample..."):
            processed = preprocess_input(input_df, imputation_medians)
            probability = model.predict_proba(processed)[0, 1]
            prediction = int(probability >= threshold)
        
        # Show result
        if prediction == 1:
            st.markdown(
                '<div class="result-card diabetic"><div class="diagnosis-text">DIABETIC</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card non-diabetic"><div class="diagnosis-text">NON-DIABETIC</div></div>',
                unsafe_allow_html=True
            )
        
        st.markdown(f'<div class="confidence-text">Confidence: {probability:.1%}</div>', unsafe_allow_html=True)
        
    except ValueError:
        st.markdown('<div class="error-box">⚠️ Invalid input. Please enter numeric values.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")


