import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "student_performance_clean.csv"
)

print("DATASET INFORMATION")
print("-------------------")
print("Total Students:", len(df))
print("Average Final Score:", round(df["Final_Academic_Score"].mean(), 2))
print("Highest Final Score:", df["Final_Academic_Score"].max())
print("Lowest Final Score:", df["Final_Academic_Score"].min())

print("\nSTATISTICAL SUMMARY")
print("-------------------")
print(df.describe().round(2))

print("\nAVERAGE SCORE BY ATTENDANCE")
print("---------------------------")
print(
    df.groupby(
        pd.cut(df["Attendance_Percentage"], bins=[0, 60, 75, 90, 100])
    )["Final_Academic_Score"].mean().round(2)
)

print("\nAVERAGE SCORE BY STUDY HOURS")
print("----------------------------")
print(
    df.groupby(
        pd.cut(df["Study_Hours_Per_Day"], bins=[0, 2, 4, 6, 24])
    )["Final_Academic_Score"].mean().round(2)
)


import matplotlib.pyplot as plt

# Create figures folder
figures_path = BASE_DIR / "reports" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

# 1. Final Score Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Final_Academic_Score"], bins=20, edgecolor="black")
plt.title("Distribution of Final Academic Scores")
plt.xlabel("Final Academic Score")
plt.ylabel("Number of Students")
plt.savefig(figures_path / "final_score_distribution.png")
plt.close()

# 2. Attendance vs Final Score
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Attendance_Percentage"],
    df["Final_Academic_Score"],
    alpha=0.4
)
plt.title("Attendance vs Final Academic Score")
plt.xlabel("Attendance Percentage")
plt.ylabel("Final Academic Score")
plt.savefig(figures_path / "attendance_vs_score.png")
plt.close()

# 3. Study Hours vs Final Score
study_data = df.groupby(
    pd.cut(df["Study_Hours_Per_Day"], bins=[0, 2, 4, 6, 24])
)["Final_Academic_Score"].mean()

plt.figure(figsize=(8, 5))
study_data.plot(kind="bar", edgecolor="black")
plt.title("Study Hours vs Average Final Score")
plt.xlabel("Study Hours Per Day")
plt.ylabel("Average Final Score")
plt.xticks(rotation=0)
plt.savefig(figures_path / "study_hours_vs_score.png")
plt.close()

print("\n3 EDA graphs generated successfully!")