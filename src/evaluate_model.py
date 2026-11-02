import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "student_performance_clean.csv"
)

# Select features and target
features = [
    "Attendance_Percentage",
    "Internal_Marks",
    "Assignment_Score",
    "Previous_Semester_Score",
    "Study_Hours_Per_Day"
]

X = df[features]
y = df["Final_Academic_Score"]

# Same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = joblib.load(
    BASE_DIR / "models" / "best_model_academic.pkl"
)

# Predict test scores
predictions = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("MODEL EVALUATION RESULTS")
print("------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

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
plt.title("Actual vs Predicted Academic Scores")

plt.tight_layout()

output_path = BASE_DIR / "reports" / "figures" / "actual_vs_predicted.png"

plt.savefig(output_path)
plt.close()

print("\nGraph saved at:", output_path)

# Residual Analysis
residuals = y_test - predictions

plt.figure(figsize=(8, 6))

plt.scatter(predictions, residuals, alpha=0.5)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Final Score")
plt.ylabel("Residuals")
plt.title("Residual Analysis")

plt.tight_layout()

residual_path = BASE_DIR / "reports" / "figures" / "residual_analysis.png"

plt.savefig(residual_path)
plt.close()

print("Residual graph saved at:", residual_path)