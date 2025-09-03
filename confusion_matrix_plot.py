import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === STEP 1: Load your saved model and dataset ===
model = joblib.load("ice_risk_rf_model.pkl")  # Make sure this file is in your directory
df = pd.read_csv("processed_wind_turbine_data.csv")  # Full dataset with labels and features

# === STEP 2: Prepare features and labels ===
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

# === STEP 3: Predict on the full dataset ===
y_pred = model.predict(X)

# === STEP 4: Confusion Matrix ===
cm = confusion_matrix(y, y_pred)

# === STEP 5: Plot ===
sns.set(style="whitegrid")
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Ice", "Ice Risk"], yticklabels=["No Ice", "Ice Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ice Risk Model")
plt.tight_layout()
plt.show()
