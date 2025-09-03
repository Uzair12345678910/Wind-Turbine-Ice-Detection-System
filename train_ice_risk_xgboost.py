import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === STEP 1: Load dataset ===
df = pd.read_csv("processed_wind_turbine_data.csv")  # Make sure this file is in your project folder
df = df.dropna()

# === STEP 2: Define features and label ===
features = [
    "LV ActivePower (kW)",
    "Wind Speed (m/s)",
    "Theoretical_Power_Curve (KWh)",
    "Wind Direction (°)",
    "Power_Diff",
    "Hour",
    "Day",
    "Month"
]
X = df[features]
y = df["Ice_Risk_Label"]

# === STEP 3: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 4: Train XGBoost model ===
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# === STEP 5: Predict & evaluate ===
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# === STEP 6: Confusion matrix ===
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Ice", "Ice Risk"], yticklabels=["No Ice", "Ice Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost Model")
plt.tight_layout()
plt.show()

# === STEP 7: Save model to file ===
joblib.dump(model, "ice_risk_xgboost_model.pkl")
print("✅ XGBoost model saved as 'ice_risk_xgboost_model.pkl'")
