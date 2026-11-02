
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load new six-feature model
model = joblib.load(
    BASE_DIR / "models" / "best_model_academic_with_engagement.pkl"
)

# Get preprocessing and trained Linear Regression model
preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["model"]

# Get actual feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get regression coefficients
coefficients = regressor.coef_

# Create feature importance dataframe
importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute Importance": abs(coefficients)
})

# Clean feature names for display
importance["Feature"] = (
    importance["Feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "Engagement - ", regex=False)
)

importance = importance.sort_values(
    "Absolute Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE - WITH ENGAGEMENT LEVEL")
print(importance.to_string(index=False))

# Create graph
plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Absolute Importance"]
)

plt.xlabel("Absolute Standardized Coefficient")
plt.title("Feature Importance - Linear Regression")

plt.gca().invert_yaxis()

plt.tight_layout()

output_path = BASE_DIR / "reports" / "figures" / "feature_importance.png"

plt.savefig(output_path, dpi=300)
plt.close()

print("\nFeature importance graph saved successfully!")
print("Graph location:", output_path)
