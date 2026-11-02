import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load original dataset
input_path = BASE_DIR / "data" / "raw" / "student_performance_challenge33.csv.xls"

df = pd.read_csv(input_path)

print("Original Dataset Shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove invalid values
df = df[
    (df["Attendance_Percentage"].between(0, 100)) &
    (df["Internal_Marks"].between(0, 100)) &
    (df["Assignment_Score"].between(0, 100)) &
    (df["Previous_Semester_Score"].between(0, 100)) &
    (df["Final_Academic_Score"].between(0, 100)) &
    (df["Study_Hours_Per_Day"].between(0, 24))
].copy()

# Handle missing values
df["Attendance_Percentage"] = df["Attendance_Percentage"].fillna(
    df["Attendance_Percentage"].median()
)

df["Assignment_Score"] = df["Assignment_Score"].fillna(
    df["Assignment_Score"].median()
)

df["Study_Hours_Per_Day"] = df["Study_Hours_Per_Day"].fillna(
    df["Study_Hours_Per_Day"].median()
)

df["Engagement_Level"] = df["Engagement_Level"].fillna(
    df["Engagement_Level"].mode()[0]
)

# Save cleaned dataset
output_path = BASE_DIR / "data" / "processed" / "student_performance_clean.csv"

df.to_csv(output_path, index=False)

print("\nCleaning Completed!")
print("Cleaned Dataset Shape:", df.shape)
print("Remaining Missing Values:", df.isnull().sum().sum())
print("Remaining Duplicates:", df.duplicated().sum())
print("Saved At:", output_path)