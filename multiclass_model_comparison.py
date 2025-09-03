import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# ────────────────────────────────────────────────
# 1. Load data
# ────────────────────────────────────────────────
df = pd.read_csv("processed_wind_turbine_data_multiclass.csv")

features = [
    "LV ActivePower (kW)",
    "Wind Speed (m/s)",
    "Theoretical_Power_Curve (KWh)",
    "Wind Direction (°)",
    "Hour", "Day", "Month", "Power_Diff"
]
target = "Multi_Ice_Risk_Label"

X = df[features]
y = df[target]

# Reproduce same train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ────────────────────────────────────────────────
# 2. Load models
# ────────────────────────────────────────────────
logistic_model = joblib.load("multiclass_logistic_model.pkl")
xgb_model = joblib.load("multiclass_xgboost_model.pkl")

# ────────────────────────────────────────────────
# 3. Make predictions
# ────────────────────────────────────────────────
logistic_preds = logistic_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

logistic_probs = logistic_model.predict_proba(X_test)
xgb_probs = xgb_model.predict_proba(X_test)

# ────────────────────────────────────────────────
# 4. Classification Report and Accuracy Plot
# ────────────────────────────────────────────────
report_log = classification_report(y_test, logistic_preds, output_dict=True)
report_xgb = classification_report(y_test, xgb_preds, output_dict=True)

acc_log = accuracy_score(y_test, logistic_preds)
acc_xgb = accuracy_score(y_test, xgb_preds)

# Save text report
with open("figures/classification_report.txt", "w") as f:
    f.write("=== Logistic Regression ===\n")
    f.write(classification_report(y_test, logistic_preds))
    f.write("\n\n=== XGBoost ===\n")
    f.write(classification_report(y_test, xgb_preds))

# Bar plot of accuracy
plt.figure(figsize=(6, 4))
plt.bar(["Logistic", "XGBoost"], [acc_log, acc_xgb], color=["skyblue", "lightgreen"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/accuracy_comparison.png")
plt.close()

# ────────────────────────────────────────────────
# 5. Confusion Matrices
# ────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

cm_log = confusion_matrix(y_test, logistic_preds)
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("Logistic Regression")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("Actual")

cm_xgb = confusion_matrix(y_test, xgb_preds)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", ax=axs[1])
axs[1].set_title("XGBoost")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("figures/confusion_matrices.png")
plt.close()

# ────────────────────────────────────────────────
# 6. ROC Curves (One per Class)
# ────────────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

fpr_log, tpr_log, roc_auc_log = {}, {}, {}
fpr_xgb, tpr_xgb, roc_auc_xgb = {}, {}, {}

for i in range(n_classes):
    fpr_log[i], tpr_log[i], _ = roc_curve(y_test_bin[:, i], logistic_probs[:, i])
    roc_auc_log[i] = auc(fpr_log[i], tpr_log[i])

    fpr_xgb[i], tpr_xgb[i], _ = roc_curve(y_test_bin[:, i], xgb_probs[:, i])
    roc_auc_xgb[i] = auc(fpr_xgb[i], tpr_xgb[i])

colors = ["red", "orange", "blue", "green"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i in range(n_classes):
    axes[i].plot(fpr_log[i], tpr_log[i], linestyle="--", color=colors[i], alpha=0.8,
                 label=f"Logistic (AUC = {roc_auc_log[i]:.2f})")
    axes[i].plot(fpr_xgb[i], tpr_xgb[i], linestyle="-", color=colors[i], alpha=0.8,
                 label=f"XGBoost (AUC = {roc_auc_xgb[i]:.2f})")
    axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[i].set_title(f"Class {i}")
    axes[i].set_xlabel("False Positive Rate")
    axes[i].set_ylabel("True Positive Rate")
    axes[i].legend(loc="lower right")
    axes[i].grid(True)

plt.suptitle("ROC Curve Comparison – One per Class", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/roc_curve_comparison.png")
plt.close()

print("✅ All outputs saved to 'figures/' folder.")
