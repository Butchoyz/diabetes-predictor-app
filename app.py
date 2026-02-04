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
# MODERN CUSTOM CSS (COMPLETE)
# ============================================
st.markdown("""
    <style>
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
    
    .stExpander {
        background: #F8FAFC;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #E2E8F0;
    }
    .stExpander > div:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px 12px 0 0;
    }
    
    /* PROGRESS BAR COLORS */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .error-box {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* ============================================ */
    /* NEW ENHANCED BUTTON STYLE                    */
    /* ============================================ */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); /* Slight gradient shift */
    }

    div.stButton > button:first-child:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Keyframe animation for results */
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
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
# USER GUIDE EXPANDER
# ============================================
with st.expander("📖 Click here for instructions and measurement guide", expanded=False):
    st.markdown("""
    <div style="background: white; border-radius: 12px; border: 1px solid #E5E7EB; border-left: 5px solid #6366F1; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem;">
        <span style="font-size: 1.5rem;">🔬</span>
        <h3 style="color: #1F2937; margin: 0; font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.2rem;">What This App Does</h3>
    </div>
    <p style="color: #4B5563; font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.6; margin: 0;">
        This intelligent screening tool predicts diabetes risk using 8 medical measurements. 
        Powered by machine learning trained on the Pima Indians Diabetes dataset.
    </p>
    </div>

    <div style="background: #FEF2F2; border-left: 4px solid #EF4444; padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
        <strong style="color: #EF4444;">⚠️ Important:</strong>
        <span style="color: #7F1D1D;">This is a <strong>screening tool</strong>, not a medical diagnosis.
        Always consult healthcare professionals for proper medical advice.</span>
    </div>

    <div style="margin-bottom: 2rem;">
    <h3 style="color: #4F46E5; margin-bottom: 1.25rem; font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
        📋 How to Use This Tool
    </h3>
    <div style="display: flex; flex-direction: column; gap: 1rem;">
        
    <div style="display: flex; align-items: flex-start; gap: 12px;">
    <div style="background: #EEF2FF; color: #4F46E5; min-width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; border: 1px solid #C7D2FE;">
                1
    </div>
    <div style="color: #374151; font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.5; padding-top: 4px;">
                Enter all available patient measurements in the fields below.
    </div>
    </div>
    <div style="display: flex; align-items: flex-start; gap: 12px;">
    <div style="background: #EEF2FF; color: #4F46E5; min-width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; border: 1px solid #C7D2FE;">
                2
    </div>
    <div style="color: #374151; font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.5; padding-top: 4px;">
                If a measurement is <strong>unavailable</strong>, leave it as <strong>0</strong> – the app handles missing values automatically.
    </div>
    </div>

    <div style="display: flex; align-items: flex-start; gap: 12px;">
    <div style="background: #EEF2FF; color: #4F46E5; min-width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; border: 1px solid #C7D2FE;">
                3
    </div>
    <div style="color: #374151; font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.5; padding-top: 4px;">
                Click <strong>"Analyze Sample"</strong> to get instant prediction results.
    </div>
    </div>

    </div>
    </div>

    <div style="margin-bottom: 2rem;">
    <div style="margin-bottom: 2rem;">
    <h3 style="color: #4F46E5; margin-bottom: 1rem; font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
        📊 Required Measurements & Units
    </h3>
    <div style="overflow-x: auto; border: 1px solid #E5E7EB; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <table style="width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; font-size: 0.9rem; background: white;">
            <thead>
                <tr style="background: #F8FAFC; border-bottom: 1px solid #E5E7EB;">
                    <th style="padding: 12px 16px; text-align: left; font-weight: 600; color: #1F2937; white-space: nowrap;">Field</th>
                    <th style="padding: 12px 16px; text-align: left; font-weight: 600; color: #1F2937;">Description</th>
                    <th style="padding: 12px 16px; text-align: left; font-weight: 600; color: #1F2937;">Units</th>
                </tr>
            </thead>
            <tbody style="color: #4B5563;">
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 12px 16px; font-weight: 500; color: #111827; white-space: nowrap;">🤰 Pregnancies</td>
                    <td style="padding: 12px 16px;">Number of times pregnant</td>
                    <td style="padding: 12px 16px;">Count</td>
                </tr>
                <tr style="background: #F9FAFB; border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 12px 16px; font-weight: 500; color: #111827; white-space: nowrap;">🩸 Glucose</td>
                    <td style="padding: 12px 16px;">Blood sugar (2-hour oral test)</td>
                    <td style="padding: 12px 16px;">mg/dL</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 12px 16px; font-weight: 500; color: #111827; white-space: nowrap;">❤️ Blood Pressure</td>
                    <td style="padding: 12px 16px;">Diastolic blood pressure</td>
                    <td style="padding: 12px 16px;">mm Hg</td>
                </tr>
                <tr style="background: #F9FAFB; border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 12px 16px; font-weight: 500; color: #111827; white-space: nowrap;">📏 Skin Thickness</td>
                    <td style="padding: 12px 16px;">Triceps skin fold thickness</td>
                    <td style="padding: 12px 16px;">mm</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 12px 16px; font-weight: 500; color: #111827; white-space: nowrap;">💉 Insulin</td>
                    <td style="padding: 12px 16px;">Serum insulin after 2 hours</td>
                    <td style="padding: 12px 16px;">μU/mL</td>
                </tr>
                <tr style="background: #F9FAFB; border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 12px 16px; font-weight: 500; color: #111827; white-space: nowrap;">⚖️ BMI</td>
                    <td style="padding: 12px 16px;">Body Mass Index</td>
                    <td style="padding: 12px 16px;">kg/m²</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 12px 16px; font-weight: 500; color: #111827; white-space: nowrap;">👨‍👩‍👧 DPF Score</td>
                    <td style="padding: 12px 16px;">Diabetes pedigree function</td>
                    <td style="padding: 12px 16px;">Score</td>
                </tr>
                <tr style="background: #F9FAFB;">
                    <td style="padding: 12px 16px; font-weight: 500; color: #111827; white-space: nowrap;">🎂 Age</td>
                    <td style="padding: 12px 16px;">Patient's age</td>
                    <td style="padding: 12px 16px;">Years</td>
                </tr>
            </tbody>
        </table>
    </div>
    </div>

    <div style="background: #F0FDF4; border: 1px solid #DCFCE7; border-left: 5px solid #10B981; border-radius: 8px; padding: 1.5rem; margin: 2rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
        <span style="font-size: 1.25rem;">💡</span>
        <h4 style="color: #166534; margin: 0; font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.05rem;">
            Tips for Accurate Predictions
        </h4>
    </div>
    <ul style="color: #15803D; font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.6; margin: 0; padding-left: 1.2rem; list-style-type: none;">
        <li style="margin-bottom: 0.75rem; display: flex; align-items: flex-start; gap: 8px;">
            <span style="color: #22C55E; font-weight: bold;">✓</span>
            <span><strong>Fill All Fields:</strong> Enter all information for the most accurate prediction.</span>
        </li>
        <li style="margin-bottom: 0.75rem; display: flex; align-items: flex-start; gap: 8px;">
            <span style="color: #22C55E; font-weight: bold;">✓</span>
            <span><strong>Missing Data:</strong> Leave as <strong>0</strong> if unavailable – the app will fill in median values.</span>
        </li>
        <li style="margin-bottom: 0.75rem; display: flex; align-items: flex-start; gap: 8px;">
            <span style="color: #22C55E; font-weight: bold;">✓</span>
            <span><strong>Units Matter:</strong> Ensure all values are in the specified units (mg/dL, mm Hg, etc).</span>
        </li>
        <li style="margin-top: 1rem;">
            <div style="background: #FFFFFF; border: 1px solid #86EFAC; padding: 0.75rem 1rem; border-radius: 6px; display: inline-block;">
                <strong style="color: #166534; font-size: 0.85rem; letter-spacing: 0.5px;">BMI FORMULA:</strong><br>
                <code style="font-family: 'Roboto Mono', monospace; color: #047857; font-size: 0.9rem;">weight(kg) ÷ [height(m)]²</code>
            </div>
        </li>
    </ul>
    </div>

    <div style="background: white; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
    <h4 style="color: #4F46E5; margin-bottom: 1.25rem; font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
        📊 Understanding Your Result
    </h4>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 12px; padding: 1.5rem; text-align: center; transition: all 0.3s ease;">
            <div style="background: #DCFCE7; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem auto;">
                <span style="font-size: 1.5rem;">✅</span>
            </div>
            <div style="color: #166534; font-weight: 800; font-family: 'Inter', sans-serif; font-size: 0.95rem; letter-spacing: 0.5px; margin-bottom: 0.25rem;">NON-DIABETIC</div>
            <div style="color: #15803D; font-size: 0.85rem; line-height: 1.4;">Low risk profile based on inputs</div>
        </div>
        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 12px; padding: 1.5rem; text-align: center; transition: all 0.3s ease;">
            <div style="background: #FEE2E2; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem auto;">
                <span style="font-size: 1.5rem;">⚠️</span>
            </div>
            <div style="color: #991B1B; font-weight: 800; font-family: 'Inter', sans-serif; font-size: 0.95rem; letter-spacing: 0.5px; margin-bottom: 0.25rem;">DIABETIC</div>
            <div style="color: #B91C1C; font-size: 0.85rem; line-height: 1.4;">High risk profile detected</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    

# ============================================
# CUSTOM CSS FOR MODERN INPUTS
# ============================================
st.markdown("""
<style>
    /* 1. Style the Input Field Border & Background */
    div[data-baseweb="input"] > div {
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        padding: 4px;
        transition: all 0.2s ease;
    }

    /* 2. Style Hover State */
    div[data-baseweb="input"] > div:hover {
        border-color: #6366F1; /* Indigo hover */
    }

    /* 3. Remove Default ugly top margin on labels */
    .stTextInput {
        margin-top: -10px;
    }
    
    /* 4. Style the Labels (The text above the box) */
    .stTextInput label p {
        font-size: 0.9rem;
        font-weight: 600;
        color: #374151; /* Dark Gray */
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SECTION HEADER
# ============================================
st.markdown("""
<div style="margin-bottom: 1.5rem; display: flex; align-items: center; gap: 10px;">
    <div style="background: #EEF2FF; padding: 8px; border-radius: 8px;">
        <span style="font-size: 1.2rem;">👤</span>
    </div>
    <div>
        <h4 style="margin: 0; color: #1F2937; font-family: 'Inter', sans-serif; font-weight: 700;">Patient Information</h4>
        <p style="margin: 0; font-size: 0.9rem; color: #6B7280;">Enter the diagnostic measurements below</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# INPUT FIELDS (MODERN LAYOUT)
# ============================================
# We use a container to give it a "Card" feel
with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        pregnancies = st.text_input(
            "🤰 Pregnancies", 
            placeholder="e.g., 2", 
            help="Number of times pregnant (0 if never)"
        )
        
        glucose = st.text_input(
            "🩸 Glucose", 
            placeholder="e.g., 120", 
            help="Plasma glucose concentration (mg/dL)"
        )
        
        blood_pressure = st.text_input(
            "❤️ Blood Pressure", 
            placeholder="e.g., 70", 
            help="Diastolic blood pressure (mm Hg)"
        )
        
        skin_thickness = st.text_input(
            "📏 Skin Thickness", 
            placeholder="e.g., 20", 
            help="Triceps skin fold thickness (mm)"
        )

    with col2:
        insulin = st.text_input(
            "💉 Insulin Level", 
            placeholder="e.g., 85", 
            help="2-Hour serum insulin (μU/mL)"
        )
        
        bmi = st.text_input(
            "⚖️ BMI Index", 
            placeholder="e.g., 25.5", 
            help="Body mass index (weight in kg / height in m²)"
        )
        
        dpf = st.text_input(
            "🧬 Diabetes Pedigree", 
            placeholder="e.g., 0.35", 
            help="Diabetes Pedigree Function (genetic score)"
        )
        
        age = st.text_input(
            "🎂 Age", 
            placeholder="e.g., 35", 
            help="Age in years"
        )

# ============================================
# PREDICT BUTTON
# ============================================
st.markdown("###")

submitted = st.button("🔍 Analyze Sample", type="primary", use_container_width=True)

# ============================================
# PROFESSIONAL RESULTS DISPLAY
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

        with st.spinner("🔬 Analyzing sample..."):
            processed = preprocess_input(input_df, imputation_medians)
            probability = model.predict_proba(processed)[0, 1]
            prediction = int(probability >= threshold)
            confidence_percent = int(round(probability * 100, 0))

        # Calculate display values
        threshold_pct = int(round(threshold * 100, 0))
        model_conf_pct = confidence_percent
        
        # Risk tiering
        if prediction == 0:
            tier = "Low Risk"
            tier_color = "#10B981"
            result_title = "NON-DIABETIC"
            result_color = "#047857"
            result_bg = "linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%)"
            result_border = "#10B981"
            result_icon = "✅"
        else:
            tier = "High Risk" if model_conf_pct >= 50 else "Moderate Risk"
            tier_color = "#EF4444" if model_conf_pct >= 50 else "#F59E0B"
            result_title = "DIABETIC"
            result_color = "#DC2626"
            result_bg = "linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%)"
            result_border = "#EF4444"
            result_icon = "⚠️"

        # Result Card using .format() instead of f-string for safer HTML
        result_html = """
        <div style="
            background: {result_bg};
            border: 2px solid {result_border};
            border-radius: 20px;
            padding: 3rem 2rem;
            margin: 2rem 0;
            text-align: center;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            animation: slideUp 0.6s ease-out;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{result_icon}</div>
            <div style="
                font-size: 2.8rem;
                font-weight: 800;
                color: {result_color};
                margin-bottom: 0.5rem;
            ">{result_title}</div>
            <div style="
                color: {result_text_color};
                font-size: 1.1rem;
                font-weight: 500;
            ">{result_message}</div>
        </div>
        """.format(
            result_bg=result_bg,
            result_border=result_border,
            result_icon=result_icon,
            result_color=result_color,
            result_title=result_title,
            result_text_color="#065F46" if prediction == 0 else "#7F1D1D",
            result_message="Low risk based on current parameters" if prediction == 0 else "High risk detected – Medical consultation recommended"
        )
        
        st.markdown(result_html, unsafe_allow_html=True)

        # ============================================
        # IMPROVED GAUGE VISUALIZATION
        # ============================================
        
        # 1. Calculate safe positions so labels/markers don't fall off the edge
        # We clamp the percentage between 5% and 95% for the visual marker position
        marker_pos = max(5, min(95, model_conf_pct))
        threshold_pos = max(5, min(95, threshold_pct))

        # 2. Define the HTML Structure
        gauge_html = f"""
        <style>
            .risk-card {{
                font-family: 'Inter', -apple-system, sans-serif;
                background: white;
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                border: 1px solid #E5E7EB;
                margin-top: 2rem;
                animation: slideUp 0.8s ease-out;
            }}
            .header {{
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
                margin-bottom: 2rem;
            }}
            .badge {{
                background: {tier_color}15;
                color: {tier_color};
                padding: 0.5rem 1rem;
                border-radius: 50px;
                font-weight: 700;
                font-size: 0.85rem;
                border: 1px solid {tier_color}30;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            /* GAUGE TRACK STYLES */
            .track-container {{
                position: relative;
                height: 40px;
                margin: 2rem 0;
            }}
            .track {{
                position: absolute;
                top: 50%;
                left: 0;
                right: 0;
                height: 12px;
                background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%);
                border-radius: 10px;
                transform: translateY(-50%);
                opacity: 0.3;
            }}
            
            /* THRESHOLD MARKER (Vertical Line) */
            .threshold-line {{
                position: absolute;
                left: {threshold_pos}%;
                top: -5px;
                bottom: -5px;
                width: 2px;
                background: #374151;
                z-index: 5;
            }}
            .threshold-label {{
                position: absolute;
                top: -25px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 0.7rem;
                color: #374151;
                font-weight: 700;
                white-space: nowrap;
            }}
            
            /* USER RESULT MARKER (The Dot) */
            .marker {{
                position: absolute;
                left: {marker_pos}%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: 24px;
                height: 24px;
                background: {tier_color};
                border: 4px solid white;
                border-radius: 50%;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                z-index: 10;
                transition: left 1s ease-out;
            }}
            .marker-label {{
                position: absolute;
                top: 30px;
                left: 50%;
                transform: translateX(-50%);
                background: {tier_color};
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: bold;
            }}

            /* STATS GRID */
            .stats-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-top: 2.5rem;
                padding-top: 1.5rem;
                border-top: 1px solid #F3F4F6;
            }}
            .stat-box {{
                text-align: center;
                padding: 1rem;
                background: #F9FAFB;
                border-radius: 12px;
            }}
            .stat-value {{
                font-size: 1.5rem;
                font-weight: 800;
                color: #111827;
            }}
            .stat-label {{
                font-size: 0.8rem;
                color: #6B7280;
                margin-top: 0.25rem;
            }}
            
            /* INSIGHT BOX */
            .insight-box {{
                margin-top: 1.5rem;
                padding: 1rem;
                background: {tier_color}08;
                border-radius: 8px;
                border-left: 4px solid {tier_color};
                color: #4B5563;
                font-size: 0.9rem;
                line-height: 1.5;
            }}
        </style>

        <div class="risk-card">
            <div class="header">
                <div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #111827;">Analysis & Confidence</div> 
                </div>
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
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #9CA3AF; margin-top: -10px;">
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
            <div class="insight-box">
                <strong>Interpretation:</strong> The model is <strong>{model_conf_pct}%</strong> certain of this result. 
                {"This score is below the threshold, indicating a healthy result." if prediction == 0 else "This score exceeds the threshold, suggesting a high probability of diabetes."}
            </div>
        </div>
        """
        
        st.components.v1.html(gauge_html, height=600)

        # Additional insights for diabetic results
        if prediction == 1:
            st.markdown("""
            <div style="
                background: #FEF2F2;
                border: 1px solid #FECACA;
                border-radius: 12px;
                padding: 1.5rem;
                margin-top: 1.5rem;
                animation: slideUp 1s ease-out;
            ">
                <h4 style="color: #DC2626; margin-bottom: 0.5rem;">🩺 Recommended Next Steps</h4>
                <ul style="color: #7F1D1D; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    <li>Schedule follow-up appointment with healthcare provider</li>
                    <li>Consider additional diagnostic tests (HbA1c, fasting glucose)</li>
                    <li>Review lifestyle factors and family history</li>
                    <li>Monitor symptoms and follow medical guidance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    except ValueError as ve:
        st.markdown('<div class="error-box">⚠️ Invalid input format. Please enter valid numbers.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.info("💡 Tip: Make sure model files are in the same directory as this script.")