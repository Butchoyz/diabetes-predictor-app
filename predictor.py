import pandas as pd
import joblib
import numpy as np
import streamlit as st
from abc import ABC, abstractmethod

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
# 2. MODEL CONFIGURATION & CACHING
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
# 3. HELPER: HTML RESULT CARD GENERATOR
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
        card_style = "background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(12px); border: 2px solid #818cf8; border-radius: 20px; padding: 1.5rem; box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.2); width: 100%; max-width: 700px; margin: 0 auto;"
        badge_label = "⭐ PROPOSED STACKING ENSEMBLE"
        badge_color = "#4f46e5"
        wrapper_class = "hero-grid-item"
    else:
        card_style = "background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 16px; padding: 1.25rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);"
        badge_label = "📊 BASELINE MODEL"
        badge_color = "#64748b"
        wrapper_class = ""

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