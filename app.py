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
    }
    .stExpander > div:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px 12px 0 0;
    }
    
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
        <li>If a measurement is <strong>unavailable</strong>, leave it as <strong>0</strong> – the app handles missing values automatically</li>
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
            <li><strong>Missing Data:</strong> Leave as <strong>0</strong> if unavailable – the app will fill in median values automatically.</li>
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
                    <div style="font-size: 0.85rem; color: #6B7280;">AI Model Certainty</div>
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