import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load model and dataset
model = joblib.load(
    BASE_DIR / "models" / "best_model_academic.pkl"
)

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "student_performance_clean.csv"
)

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Header
st.title("🎓 Student Academic Performance Prediction")

st.write(
    "Predict a student's final academic score using "
    "attendance, previous marks, study hours and academic performance."
)

st.divider()

# Dashboard
st.subheader("Academic Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))

col2.metric(
    "Average Final Score",
    f"{df['Final_Academic_Score'].mean():.2f} / 100"
)

col3.metric(
    "Highest Score",
    f"{df['Final_Academic_Score'].max():.2f} / 100"
)

st.divider()

# Prediction Form
st.subheader("Predict Student Performance")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.number_input(
            "Attendance Percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        internal_marks = st.number_input(
            "Internal Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        assignment_score = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

    with col2:
        previous_score = st.number_input(
            "Previous Semester Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        study_hours = st.number_input(
            "Study Hours Per Day",
            min_value=0.0,
            max_value=24.0,
            value=4.0
        )

    submitted = st.form_submit_button(
        "Predict Final Score",
        use_container_width=True
    )

# Prediction Result
if submitted:

    input_data = pd.DataFrame([{
        "Attendance_Percentage": attendance,
        "Internal_Marks": internal_marks,
        "Assignment_Score": assignment_score,
        "Previous_Semester_Score": previous_score,
        "Study_Hours_Per_Day": study_hours
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Final Academic Score",
        f"{prediction:.2f} / 100"
    )

    st.progress(prediction / 100)

    if prediction >= 80:
        st.success("Performance Level: Excellent 🌟")

    elif prediction >= 60:
        st.info("Performance Level: Good 👍")

    elif prediction >= 40:
        st.warning("Performance Level: Average 📚")

    else:
        st.error("Performance Level: Needs Improvement")

    st.caption(
        "This prediction is an estimate based on the trained ML model. "
        "It does not guarantee actual examination results."
    )

st.divider()

st.caption(
    "Machine Learning Project | Student Academic Performance Prediction"
)

# Model Performance Section

st.divider()

st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "4.68")
col2.metric("RMSE", "5.90")
col3.metric("R² Score", "0.5798")

st.subheader("Actual vs Predicted Scores")

actual_graph = BASE_DIR / "reports" / "figures" / "actual_vs_predicted.png"

st.image(str(actual_graph), use_container_width=True)

st.subheader("Residual Analysis")

residual_graph = BASE_DIR / "reports" / "figures" / "residual_analysis.png"

st.image(str(residual_graph), use_container_width=True)