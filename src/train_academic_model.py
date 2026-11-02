import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned dataset
df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "student_performance_clean.csv"
)

# Select relevant input features
features = [
    "Attendance_Percentage",
    "Internal_Marks",
    "Assignment_Score",
    "Previous_Semester_Score",
    "Study_Hours_Per_Day"
]

X = df[features]
y = df["Final_Academic_Score"]

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# Define models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []
trained_models = {}

# Train and evaluate models
for name, algorithm in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

# Compare models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE")

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Select model with lowest RMSE
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

# Save best model separately
model_path = BASE_DIR / "models" / "best_model_academic.pkl"
joblib.dump(best_model, model_path)

# Save comparison report
report_path = BASE_DIR / "reports" / "academic_model_comparison.csv"
results_df.to_csv(report_path, index=False)

print("\nBest Model:", best_name)
print("Model saved at:", model_path)
print("Comparison report saved at:", report_path)