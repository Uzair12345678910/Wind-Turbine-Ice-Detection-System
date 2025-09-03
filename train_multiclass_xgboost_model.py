"""
Train a 4-class XGBoost model to predict ice-risk levels.
Classes:
    0 = No Risk, 1 = Low, 2 = Medium, 3 = High
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib

# ────────────────────────────────────────────────
# 1. Load & clean data
# ────────────────────────────────────────────────
df = pd.read_csv("processed_wind_turbine_data_multiclass.csv")

# Drop unnecessary columns like timestamp or other labels
df = df.drop(columns=["Date/Time"], errors="ignore")  # ignore if already dropped

# Define target and features
target = "Multi_Ice_Risk_Label"
drop_labels = [target, "Ice_Risk_Label"]
features = [col for col in df.columns if col not in drop_labels]

X = df[features]
y = df[target]

# ────────────────────────────────────────────────
# 2. Train/test split
# ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ────────────────────────────────────────────────
# 3. Train XGBoost classifier
# ────────────────────────────────────────────────
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model.fit(X_train, y_train)

# ────────────────────────────────────────────────
# 4. Evaluate performance
# ────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("=== XGBoost – Classification Report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
)
plt.title("Confusion Matrix – XGBoost (Multiclass)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────
# 5. Save model
# ────────────────────────────────────────────────
joblib.dump(model, "multiclass_xgboost_model.pkl")
print("✅ Model saved as multiclass_xgboost_model.pkl")
