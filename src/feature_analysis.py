import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
model = joblib.load(
    BASE_DIR / "models" / "best_model_academic.pkl"
)

# Get feature coefficients
coefficients = model.named_steps["model"].coef_

features = [
    "Attendance",
    "Internal Marks",
    "Assignment Score",
    "Previous Semester Score",
    "Study Hours"
]

importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients,
    "Absolute Importance": abs(coefficients)
})

importance = importance.sort_values(
    "Absolute Importance",
    ascending=False
)

print("FEATURE IMPORTANCE")
print(importance.to_string(index=False))

# Create graph
plt.figure(figsize=(9, 5))

plt.barh(
    importance["Feature"],
    importance["Absolute Importance"]
)

plt.xlabel("Absolute Coefficient")
plt.title("Feature Importance - Linear Regression")
plt.gca().invert_yaxis()

plt.tight_layout()

output_path = BASE_DIR / "reports" / "figures" / "feature_importance.png"

plt.savefig(output_path)
plt.close()

print("\nFeature importance graph saved successfully!")