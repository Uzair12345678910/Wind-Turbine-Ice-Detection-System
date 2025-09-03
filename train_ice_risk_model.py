import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === STEP 1: Load Dataset ===
df = pd.read_csv("processed_wind_turbine_data.csv")  # Make sure this file is in your directory

# === STEP 2: Drop Missing Values (if any) ===
df = df.dropna()

# === STEP 3: Define Features and Label ===
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

# === STEP 4: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 5: Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === STEP 6: Predict and Evaluate ===
y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# === STEP 7: Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Ice", "Ice"], yticklabels=["No Ice", "Ice"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ice Risk Model")
plt.tight_layout()
plt.show()

# === STEP 8: Save Model to Disk ===
joblib.dump(model, "ice_risk_rf_model.pkl")
print("\n✅ Model saved as 'ice_risk_rf_model.pkl'")
