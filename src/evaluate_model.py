
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned dataset
df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "student_performance_clean.csv"
)

# Select all 6 features
features = [
    "Attendance_Percentage",
    "Internal_Marks",
    "Assignment_Score",
    "Previous_Semester_Score",
    "Study_Hours_Per_Day",
    "Engagement_Level"
]

X = df[features]
y = df["Final_Academic_Score"]

# Same train-test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load new trained model
model = joblib.load(
    BASE_DIR / "models" / "best_model_academic_with_engagement.pkl"
)

# Predict academic scores
predictions = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION RESULTS - 6 FEATURES")
print("--------------------------------------")
print("MAE:", round(mae, 6))
print("RMSE:", round(rmse, 6))
print("R2 Score:", round(r2, 6))

# Create figures folder if not present
figures_dir = BASE_DIR / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Actual vs Predicted graph
plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions, alpha=0.5)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Academic Scores (6 Features)")

plt.tight_layout()

output_path = figures_dir / "actual_vs_predicted.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("\nActual vs Predicted graph saved:", output_path)

# Residual Analysis graph
residuals = y_test - predictions

plt.figure(figsize=(8, 6))

plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Final Score")
plt.ylabel("Residuals")
plt.title("Residual Analysis (6 Features)")

plt.tight_layout()

residual_path = figures_dir / "residual_analysis.png"
plt.savefig(residual_path, dpi=300)
plt.close()

print("Residual Analysis graph saved:", residual_path)

print("\nBoth graphs updated successfully!")
