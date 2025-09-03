import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Page Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Ice Risk Model Dashboard", layout="wide")
st.title("🧊 Ice Risk Prediction – Model Comparison Dashboard")

# ─────────────────────────────────────────────
# 2. Overview Section
# ─────────────────────────────────────────────
st.markdown("""
This dashboard summarizes the performance of two models (XGBoost and Logistic Regression) trained on wind turbine data to predict multi-class ice risk.
""")

# ─────────────────────────────────────────────
# 3. Accuracy Comparison
# ─────────────────────────────────────────────
st.subheader("📊 Accuracy Comparison")
st.image("figures/accuracy_comparison.png", caption="Model Accuracy Comparison", use_column_width=True)

# ─────────────────────────────────────────────
# 4. Classification Report
# ─────────────────────────────────────────────
st.subheader("📄 Classification Reports")
with open("figures/classification_reports.txt", "r") as f:
    st.text(f.read())

# ─────────────────────────────────────────────
# 5. Confusion Matrices
# ─────────────────────────────────────────────
st.subheader("🔲 Confusion Matrices")

cols = st.columns(2)

with cols[0]:
    st.image("figures/logistic_regression_confusion_matrix.png", caption="Logistic Regression", use_column_width=True)

with cols[1]:
    st.image("figures/xgboost_confusion_matrix.png", caption="XGBoost", use_column_width=True)

# Also show side-by-side comparison (optional)
st.image("figures/confusion_matrices_comparison.png", caption="Side-by-side Confusion Matrix Comparison", use_column_width=True)

# ─────────────────────────────────────────────
# 6. ROC Curve Comparison
# ─────────────────────────────────────────────
st.subheader("📈 ROC Curve Comparison")
st.image("figures/roc_curve_comparison.png", caption="ROC Curve – Class-wise Comparison", use_column_width=True)

# ─────────────────────────────────────────────
# 7. SHAP Summary Plots
# ─────────────────────────────────────────────
st.subheader("🔍 SHAP Summary – Feature Importance")

cols = st.columns(2)

with cols[0]:
    st.image("figures/shap_summary_logistic.png", caption="Logistic Regression", use_column_width=True)

with cols[1]:
    st.image("figures/shap_summary_xgboost.png", caption="XGBoost", use_column_width=True)

# ─────────────────────────────────────────────
# 8. Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("✅ All visualizations generated from the latest trained models. For raw predictions or live SHAP explanations, switch to the interactive app.")

