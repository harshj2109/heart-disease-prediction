
import streamlit as st
import pandas as pd

from src.Heart.pipeline.Prediction_pipeline import (
    CustomData,
    PredictPipeline
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CardioCare | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #f4f8fc 0%, #edf4fa 100%);
}

/* Main content */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #102c46, #176b87);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(16, 44, 70, 0.15);
}

.hero h1 {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #d8eaf3;
    font-size: 16px;
    margin-bottom: 0;
}

/* Section cards */
.section-card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #e1eaf1;
    box-shadow: 0 4px 15px rgba(20, 50, 80, 0.05);
    margin-bottom: 18px;
}

/* Section headings */
.section-title {
    color: #123b59;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #718397;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Input labels */
label {
    color: #28465e !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(120deg, #126b8a, #1596a6);
    color: white;
    border: none;
    border-radius: 10px;
    height: 52px;
    font-size: 16px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(120deg, #0e526d, #117b8c);
    color: white;
    border: none;
    transform: translateY(-2px);
}

/* Result cards */
.result-card {
    padding: 28px;
    border-radius: 16px;
    margin-top: 15px;
    text-align: center;
}

.result-positive {
    background: #fff0f0;
    border: 1px solid #ffcaca;
}

.result-negative {
    background: #eaf8f0;
    border: 1px solid #bce8ce;
}

.result-card h2 {
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 10px;
}

.result-card p {
    font-size: 14px;
    margin-bottom: 0;
}

.footer {
    text-align: center;
    color: #8495a5;
    font-size: 13px;
    padding: 25px 0 10px;
}

[data-testid="stSidebar"] {
    background: #102c46;
}

[data-testid="stSidebar"] * {
    color: white;
}

hr {
    border-color: #dce7ef;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## ❤️ CardioCare")

    st.caption("ML-powered health prediction demo")

    st.divider()

    st.markdown("### About the Project")

    st.write(
        "CardioCare is a machine learning application "
        "that uses patient health parameters to generate "
        "a heart disease prediction."
    )

    st.divider()

    st.markdown("### Technology Stack")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Support Vector Classifier
    """)

    st.divider()

    st.markdown("### Project Information")

    st.write("**Project:** Heart Disease Prediction")

    st.write("**Model:** SVC")

    st.write("**Platform:** Machine Learning")

    st.divider()

    st.caption("Developed as an academic ML project.")


# ==========================================
# HERO HEADER
# ==========================================

st.markdown("""
<div class="hero">

    <h1>❤️ Heart Disease Prediction</h1>

    <p>
        Intelligent health parameter analysis using
        Machine Learning.
    </p>

    <p style="margin-top:15px; font-size:13px;">
        CARDIOCARE &nbsp; | &nbsp; MACHINE LEARNING PROJECT
    </p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# INTRODUCTION
# ==========================================

st.markdown(
    "### Patient Health Assessment"
)

st.markdown(
    "Enter the required health parameters below to "
    "generate a prediction using the trained ML model."
)

st.write("")


# ==========================================
# INPUT FORM
# ==========================================

with st.form("prediction_form"):

    st.markdown("""
    <div class="section-title">
        👤 Personal Information
    </div>
    <div class="section-subtitle">
        Basic patient details
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=45
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x:
                "Female" if x == 0 else "Male"
        )

    with col2:

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=1,
            value=120
        )

        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=1,
            value=200
        )

    st.divider()

    st.markdown("""
    <div class="section-title">
        🫀 Cardiovascular Parameters
    </div>
    <div class="section-subtitle">
        Clinical measurements and exercise-related features
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            help="Select the encoded chest pain category."
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar",
            [0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=1,
            value=150
        )

    with col2:

        exang = st.selectbox(
            "Exercise-Induced Angina",
            [0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        oldpeak = st.number_input(
            "Oldpeak",
            value=1.0,
            step=0.1,
            format="%.1f"
        )

        slope = st.selectbox(
            "Slope",
            [0, 1, 2]
        )

        ca = st.selectbox(
            "Major Vessels (ca)",
            [0, 1, 2, 3, 4]
        )

    st.divider()

    st.markdown("""
    <div class="section-title">
        🔬 Additional Parameters
    </div>
    <div class="section-subtitle">
        Additional encoded model features
    </div>
    """, unsafe_allow_html=True)

    thal = st.selectbox(
        "Thalassemia (thal)",
        [0, 1, 2, 3]
    )

    st.write("")

    submit = st.form_submit_button(
        "🔍  Generate Prediction",
        use_container_width=True
    )


# ==========================================
# PREDICTION LOGIC
# ==========================================

if submit:

    try:

        with st.spinner(
            "Analyzing patient parameters..."
        ):

            # Create CustomData object
            data = CustomData(
                age=age,
                sex=sex,
                cp=cp,
                trestbps=trestbps,
                chol=chol,
                fbs=fbs,
                restecg=restecg,
                thalach=thalach,
                exang=exang,
                oldpeak=oldpeak,
                slope=slope,
                ca=ca,
                thal=thal
            )

            # Convert to DataFrame
            final_data = data.get_data_as_dataframe()

            # Load trained model
            predict_pipeline = PredictPipeline()

            # Generate prediction
            prediction = predict_pipeline.predict(
                final_data
            )

            result = int(prediction[0])

        # ==================================
        # DISPLAY RESULT
        # ==================================

        st.divider()

        st.markdown(
            "## 📊 Prediction Analysis"
        )

        if result == 0:

            st.markdown("""
            <div class="result-card result-negative">

                <div style="font-size:45px;">
                    ✅
                </div>

                <h2 style="color:#16834b;">
                    Heart Disease Not Predicted
                </h2>

                <p style="color:#347354;">
                    The machine learning model returned
                    prediction class 0.
                </p>

            </div>
            """, unsafe_allow_html=True)

        elif result == 1:

            st.markdown("""
            <div class="result-card result-positive">

                <div style="font-size:45px;">
                    ⚠️
                </div>

                <h2 style="color:#c03939;">
                    Heart Disease Predicted
                </h2>

                <p style="color:#9f4b4b;">
                    The machine learning model returned
                    prediction class 1.
                </p>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.info(
                f"Unexpected model output: {result}"
            )

        # Input summary
        with st.expander(
            "View Patient Input Details"
        ):

            st.dataframe(
                final_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            "Something went wrong while generating prediction."
        )

        st.exception(e)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div class="footer">

    <b>CardioCare | Heart Disease Prediction System</b>

    <br><br>

    Built using Python, Streamlit and Machine Learning.

    <br><br>

    ⚠️ Educational demonstration only. This application
    is not a medical diagnostic tool. Model predictions
    should not be used for medical decisions.

</div>
""", unsafe_allow_html=True)
