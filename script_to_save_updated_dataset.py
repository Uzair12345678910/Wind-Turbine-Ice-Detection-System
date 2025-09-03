import pandas as pd

# Load original dataset
df = pd.read_csv("processed_wind_turbine_data.csv")

# Define function to assign multiclass labels
def assign_multiclass_label(row):
    if row["Ice_Risk_Label"] == 0:
        return 0
    elif row["Temperature (C)"] < -15 or row["Humidity (%)"] > 95:
        return 2  # High Risk
    else:
        return 1  # Medium Risk

# Apply function
df["Multi_Ice_Risk_Label"] = df.apply(assign_multiclass_label, axis=1)

# Save to new CSV file
df.to_csv("updated_multiclass_wind_turbine_data.csv", index=False)
print("✅ Saved as 'updated_multiclass_wind_turbine_data.csv'")
