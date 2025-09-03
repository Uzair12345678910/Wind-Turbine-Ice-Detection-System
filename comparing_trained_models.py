import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
import seaborn as sns
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("processed_wind_turbine_data.csv")

features = [
    "LV ActivePower (kW)", "Wind Speed (m/s)", "Theoretical_Power_Curve (KWh)",
    "Wind Direction (°)", "Power_Diff", "Hour", "Day", "Month"
]
X = df[features]
y = df["Ice_Risk_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Load Models ===
lr_model = joblib.load("ice_risk_xgboost_model.pkl")
xgb_model = joblib.load("ice_risk_xgboost_model.pkl")

# === Predict ===
y_pred_lr = lr_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# === Classification Reports ===
print("=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_lr))

print("=== XGBoost Report ===")
print(classification_report(y_test, y_pred_xgb))

# === Confusion Matrices ===
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Ice", "Ice Risk"], yticklabels=["No Ice", "Ice Risk"])
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", ax=axes[1],
            xticklabels=["No Ice", "Ice Risk"], yticklabels=["No Ice", "Ice Risk"])
axes[1].set_title("XGBoost")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# === ROC Curves & AUC ===
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

auc_lr = auc(fpr_lr, tpr_lr)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.3f})", linewidth=2, linestyle="--")
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
