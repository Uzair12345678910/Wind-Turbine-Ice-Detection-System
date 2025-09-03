import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load models
logistic_model = joblib.load("multiclass_logistic_model.pkl")
xgboost_model = joblib.load("multiclass_xgboost_model.pkl")

# Load dataset
df = pd.read_csv("processed_wind_turbine_data_multiclass.csv")
features = [
    "LV ActivePower (kW)", "Wind Speed (m/s)",
    "Theoretical_Power_Curve (KWh)", "Wind Direction (°)",
    "Hour", "Day", "Month", "Power_Diff"
]
X = df[features]

st.set_page_config(page_title="Interactive Ice Risk Predictor", layout="wide")
st.title("🧊 Interactive Ice Risk Predictor")

# Select model
model_choice = st.selectbox("Select model:", ["XGBoost", "Logistic Regression"])
model = xgboost_model if model_choice == "XGBoost" else logistic_model

# Pick a sample index
index = st.slider("Select a sample index", 0, len(X)-1, 0)
input_row = X.iloc[[index]]
st.write("### Input Features", input_row.T)

# Predict
pred_class = model.predict(input_row)[0]
pred_proba = model.predict_proba(input_row)

st.write(f"### 🔮 Prediction: Class `{pred_class}`")
st.write("### 📊 Class Probabilities")
st.bar_chart(pd.DataFrame(pred_proba, columns=[f"Class {i}" for i in range(pred_proba.shape[1])]))

# SHAP
st.write("### 🔍 SHAP Explanation")

if model_choice == "XGBoost":
    explainer = shap.TreeExplainer(xgboost_model)
else:
    explainer = shap.LinearExplainer(logistic_model, X, feature_dependence="independent")

shap_values = explainer.shap_values(input_row)

# Plot SHAP
fig, ax = plt.subplots()
if model_choice == "XGBoost":
    shap.plots.waterfall(shap.Explanation(values=shap_values[pred_class][0], 
                                          base_values=explainer.expected_value[pred_class],
                                          data=input_row.iloc[0],
                                          feature_names=input_row.columns), max_display=10)
else:
    shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                          base_values=explainer.expected_value[0],
                                          data=input_row.iloc[0],
                                          feature_names=input_row.columns), max_display=10)

st.pyplot(fig)
