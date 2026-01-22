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
    
    /* Expander styling */
    .stExpander {
        background: #F8FAFC;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .stExpander > div:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px 12px 0 0;
    }
    
    /* Progress bar styling */
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
    
    /* Animation for results */
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
# USER GUIDE EXPANDER (SIMPLIFIED TABLE)
# ============================================
with st.expander("📖 Click here for instructions and measurement guide", expanded=True):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">🔬 What This App Does</h3>
        <p style="color: #4B5563; font-size: 1rem; line-height: 1.6;">
            This intelligent screening tool predicts diabetes risk using 8 medical measurements. 
            Powered by machine learning trained on the Pima Indians Diabetes dataset.
        </p>
    </div>
    
    <div style="background: #FEF2F2; border-left: 4px solid #EF4444; padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
        <strong style="color: #EF4444;">⚠️ Important:</strong> 
        <span style="color: #7F1D1D;">This is a <strong>screening tool</strong>, not a medical diagnosis. 
        Always consult healthcare professionals for proper medical advice.</span>
    </div>
    
    <h3 style="color: #667eea; margin-bottom: 1rem;">📋 How to Use This Tool</h3>
    <ol style="color: #4B5563; font-size: 1rem; line-height: 1.8; margin-bottom: 2rem;">
        <li>Enter all available patient measurements in the fields below</li>
        <li>If a measurement is <strong>unavailable</strong>, leave it as <strong>0</strong> - the app handles missing values automatically</li>
        <li>Click <strong>"Analyze Sample"</strong> to get instant prediction results</li>
    </ol>
    
    <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Required Measurements & Units</h3>
    <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <th style="padding: 1rem; text-align: left;">Field</th>
                    <th style="padding: 1rem; text-align: left;">What to Enter</th>
                    <th style="padding: 1rem; text-align: left;">Units</th>
                </tr>
            </thead>
            <tbody style="color: #4B5563;">
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 0.8rem 1rem; font-weight: 600;">🤰 Pregnancies</td>
                    <td style="padding: 0.8rem 1rem;">Number of times pregnant</td>
                    <td style="padding: 0.8rem 1rem;">Count</td>
                </tr>
                <tr style="background: #F9FAFB; border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 0.8rem 1rem; font-weight: 600;">🩸 Glucose</td>
                    <td style="padding: 0.8rem 1rem;">Blood sugar after 2-hour oral glucose test</td>
                    <td style="padding: 0.8rem 1rem;">mg/dL</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 0.8rem 1rem; font-weight: 600;">❤️ Blood Pressure</td>
                    <td style="padding: 0.8rem 1rem;">Diastolic pressure (bottom number)</td>
                    <td style="padding: 0.8rem 1rem;">mm Hg</td>
                </tr>
                <tr style="background: #F9FAFB; border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 0.8rem 1rem; font-weight: 600;">📏 Skin Thickness</td>
                    <td style="padding: 0.8rem 1rem;">Triceps skin fold thickness</td>
                    <td style="padding: 0.8rem 1rem;">mm</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 0.8rem 1rem; font-weight: 600;">💉 Insulin</td>
                    <td style="padding: 0.8rem 1rem;">Serum insulin after 2 hours</td>
                    <td style="padding: 0.8rem 1rem;">μU/mL</td>
                </tr>
                <tr style="background: #F9FAFB; border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 0.8rem 1rem; font-weight: 600;">⚖️ BMI</td>
                    <td style="padding: 0.8rem 1rem;">Body Mass Index</td>
                    <td style="padding: 0.8rem 1rem;">kg/m²</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 0.8rem 1rem; font-weight: 600;">👨‍👩‍👧 DPF Score</td>
                    <td style="padding: 0.8rem 1rem;">Diabetes pedigree function</td>
                    <td style="padding: 0.8rem 1rem;">Score</td>
                </tr>
                <tr style="background: #F9FAFB;">
                    <td style="padding: 0.8rem 1rem; font-weight: 600;">🎂 Age</td>
                    <td style="padding: 0.8rem 1rem;">Patient age</td>
                    <td style="padding: 0.8rem 1rem;">Years</td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 12px; padding: 1.5rem; margin: 2rem 0;">
        <h4 style="color: #047857; margin-bottom: 1rem;">💡 Tips for Accurate Predictions</h4>
        <ul style="color: #065F46; font-size: 0.95rem; line-height: 1.8;">
        <li><strong>Fill All Fields:</strong> Enter all information for the most accurate prediction.</li>
            <li><strong>Missing Data:</strong> Leave as <strong>0</strong> if unavailable – the app will fill in median values automatically, but missing inputs may reduce the accuracy or realism of the prediction.</li>
            <li><strong>Zero Values:</strong> The dataset uses 0 to indicate missing measurements, not actual zero values</li>
            <li><strong>Units Matter:</strong> Ensure all values are in the specified units</li>
            <li><strong>BMI Formula:</strong> <code style="background: #DCFCE7; padding: 2px 6px; border-radius: 4px;">BMI = weight(kg) ÷ [height(m)]²</code></li>
        </ul>
    </div>
    
    <div style="background: linear-gradient(135deg, #E0E7FF 0%, #DDD6FE 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
        <h4 style="color: #4338CA; margin-bottom: 1rem;">📊 Understanding Your Result</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 2px solid #10B981;">
                <strong style="color: #047857;">✅ NON-DIABETIC</strong>
                <p style="color: #6B7280; font-size: 0.9rem; margin: 0.5rem 0 0;">Low risk based on input parameters</p>
            </div>
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 2px solid #EF4444;">
                <strong style="color: #DC2626;">⚠️ DIABETIC</strong>
                <p style="color: #6B7280; font-size: 0.9rem; margin: 0.5rem 0 0;">High risk - recommend medical follow-up</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# INPUT FIELDS
# ============================================

col1, col2 = st.columns(2, gap="large")

with col1:
    pregnancies = st.text_input("🤰 Pregnancies", placeholder="e.g., 2 (0 if never pregnant)")
    glucose = st.text_input("🩸 Glucose (mg/dL)", placeholder="e.g., 120 (0 if unknown)")
    blood_pressure = st.text_input("❤️ Diastolic BP (mm Hg)", placeholder="e.g., 70 (0 if unknown)")
    skin_thickness = st.text_input("📏 Skin Fold Thickness (mm)", placeholder="e.g., 20 (0 if unknown)")

with col2:
    insulin = st.text_input("💉 2-Hour Insulin (μU/mL)", placeholder="e.g., 85 (0 if unknown)")
    bmi = st.text_input("⚖️ BMI (kg/m²)", placeholder="e.g., 25.5")
    dpf = st.text_input("👨‍👩‍👧 Diabetes Pedigree Function", placeholder="e.g., 0.3 (0.5+ = high risk)")
    age = st.text_input("🎂 Age (years)", placeholder="e.g., 35")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICT BUTTON
# ============================================
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
        
        # Premium Result Card
        if prediction == 1:
            result_card = """
            <div style="
                background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
                border: 2px solid #EF4444;
                border-radius: 20px;
                padding: 3rem 2rem;
                margin: 2rem 0;
                text-align: center;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                animation: slideUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: -10px;
                    right: -10px;
                    width: 60px;
                    height: 60px;
                    background: #EF4444;
                    border-radius: 50%;
                    opacity: 0.1;
                "></div>
                <div style="font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
                <div style="
                    font-size: 2.8rem;
                    font-weight: 800;
                    color: #DC2626;
                    letter-spacing: -1px;
                    margin-bottom: 0.5rem;
                ">DIABETIC</div>
                <div style="
                    color: #7F1D1D;
                    font-size: 1.1rem;
                    font-weight: 500;
                ">High risk detected - Medical consultation recommended</div>
            </div>
            """
        else:
            result_card = """
            <div style="
                background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
                border: 2px solid #10B981;
                border-radius: 20px;
                padding: 3rem 2rem;
                margin: 2rem 0;
                text-align: center;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                animation: slideUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: -10px;
                    right: -10px;
                    width: 60px;
                    height: 60px;
                    background: #10B981;
                    border-radius: 50%;
                    opacity: 0.1;
                "></div>
                <div style="font-size: 4rem; margin-bottom: 1rem;">✅</div>
                <div style="
                    font-size: 2.8rem;
                    font-weight: 800;
                    color: #047857;
                    letter-spacing: -1px;
                    margin-bottom: 0.5rem;
                ">NON-DIABETIC</div>
                <div style="
                    color: #065F46;
                    font-size: 1.1rem;
                    font-weight: 500;
                ">Low risk based on current parameters</div>
            </div>
            """
        
        st.markdown(result_card, unsafe_allow_html=True)
        
        # Professional Confidence Gauge
        st.markdown("""
        <div style="
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            border: 1px solid #E5E7EB;
            margin-top: 1.5rem;
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            ">
                <div style="
                    font-size: 1.3rem;
                    font-weight: 600;
                    color: #374151;
                ">Model Confidence Level</div>
                <div style="
                    font-size: 2rem;
                    font-weight: 800;
                    color: #667eea;
                ">{}%</div>
            </div>
            <div style="
                width: 100%;
                height: 12px;
                background: #E5E7EB;
                border-radius: 10px;
                overflow: hidden;
                position: relative;
            ">
                <div style="
                    width: {}%;
                    height: 100%;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                    transition: width 1.2s cubic-bezier(0.34, 1.56, 0.64, 1);
                    box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
                "></div>
            </div>
            <div style="
                display: flex;
                justify-content: space-between;
                margin-top: 0.5rem;
                font-size: 0.85rem;
                color: #9CA3AF;
            ">
                <span>Uncertain</span>
                <span>Very Confident</span>
            </div>
        </div>
        """.format(confidence_percent, confidence_percent), unsafe_allow_html=True)
        
        # Additional insights for diabetic results
        if prediction == 1:
            st.markdown("""
            <div style="
                background: #FEF2F2;
                border: 1px solid #FECACA;
                border-radius: 12px;
                padding: 1.5rem;
                margin-top: 1.5rem;
            ">
                <h4 style="color: #DC2626; margin-bottom: 0.5rem;">💡 Recommended Next Steps</h4>
                <ul style="color: #7F1D1D; line-height: 1.6; margin: 0;">
                    <li>Schedule follow-up appointment with healthcare provider</li>
                    <li>Consider additional diagnostic tests (HbA1c, fasting glucose)</li>
                    <li>Review lifestyle factors and family history</li>
                    <li>Monitor symptoms and follow medical guidance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
    except ValueError:
        st.markdown('<div class="error-box">⚠️ Invalid input. Please enter numeric values.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")