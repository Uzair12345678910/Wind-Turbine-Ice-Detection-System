import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("processed_wind_turbine_data_multiclass.csv")

# Drop datetime and any label columns that should not be in features
df = df.drop(columns=["Date/Time"])

# Define target variable
y = df["Multi_Ice_Risk_Label"]

# Ensure that only feature columns are used (drop *all* label columns)
X = df.drop(columns=[col for col in ["Multi_Ice_Risk_Label", "Ice_Risk_Label"] if col in df.columns])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(
    max_iter=5000, solver='lbfgs', multi_class='multinomial'
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "multiclass_logistic_model.pkl")

# Evaluate on test set
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
