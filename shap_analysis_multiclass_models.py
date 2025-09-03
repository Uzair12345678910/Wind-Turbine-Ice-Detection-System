"""
SHAP Analysis Script for Multiclass Logistic Regression and XGBoost Models
"""

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

# ────────────────────────────────────────────────
# 1. Setup
# ────────────────────────────────────────────────
# Ensure figures folder exists
os.makedirs("figures", exist_ok=True)

# Load dataset
df = pd.read_csv("processed_wind_turbine_data_multiclass.csv")

# Define features and target
features = [
    "LV ActivePower (kW)",
    "Wind Speed (m/s)",
    "Theoretical_Power_Curve (KWh)",
    "Wind Direction (°)",
    "Hour",
    "Day",
    "Month",
    "Power_Diff"
]
target = "Multi_Ice_Risk_Label"
X = df[features]

# Take a sample (for speed)
X_sample = X.sample(n=100, random_state=42)

# ────────────────────────────────────────────────
# 2. SHAP for XGBoost
# ────────────────────────────────────────────────
print("🔍 Running SHAP for XGBoost...")
xgb_model = joblib.load("multiclass_xgboost_model.pkl")
explainer_xgb = shap.Explainer(xgb_model)
shap_values_xgb = explainer_xgb(X_sample)

shap.summary_plot(shap_values_xgb, X_sample, show=False)
plt.title("SHAP Summary – XGBoost")
plt.tight_layout()
plt.savefig("figures/shap_summary_xgboost.png")
plt.clf()

# ────────────────────────────────────────────────
# 3. SHAP for Logistic Regression (KernelExplainer)
# ────────────────────────────────────────────────
print("🔍 Running SHAP for Logistic Regression (KernelExplainer)...")
log_model = joblib.load("multiclass_logistic_model.pkl")
explainer_log = shap.Explainer(log_model, X_sample)
shap_values_log = explainer_log(X_sample)

shap.summary_plot(shap_values_log, X_sample, show=False)
plt.title("SHAP Summary – Logistic Regression")
plt.tight_layout()
plt.savefig("figures/shap_summary_logistic.png")
plt.clf()

print("✅ SHAP analysis complete. Check the 'figures' folder.")
