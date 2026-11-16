
import streamlit as st
import pandas as pd

from src.Heart.pipeline.Prediction_pipeline import (
    CustomData,
    PredictPipeline
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CardioCare | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0B1220;
    color: #E5EDF7;
}

/* Main content width */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hide Streamlit top toolbar controls except GitHub link */
[data-testid="stToolbar"] button[aria-label*="Share"],
[data-testid="stToolbar"] button[title*="Share"],
[data-testid="stToolbar"] button[aria-label*="Star"],
[data-testid="stToolbar"] button[title*="Star"],
[data-testid="stToolbar"] button[aria-label*="Edit"],
[data-testid="stToolbar"] button[title*="Edit"],
[data-testid="stToolbar"] button[aria-label*="menu"],
[data-testid="stToolbar"] button[title*="menu"],
[data-testid="stToolbar"] button[data-testid="stMainMenu"] {
    display: none !important;
}

/* Hide sidebar */
[data-testid="stSidebar"] {
    display: none !important;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #F1F5F9;
}

/* Paragraphs */
.stApp p {
    color: #C5D2E3;
}

/* Hero container */
.hero-box {
    background: #142A43;
    border: 1px solid #24536A;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 25px;
}

.hero-box h1 {
    color: #FFFFFF !important;
    font-size: 34px;
    font-weight: 800;
}

.hero-box p {
    color: #B9DCEB !important;
    font-size: 15px;
    line-height: 1.7;
}

/* Section headings */
.stApp h2,
.stApp h3 {
    color: #E8F3FC;
}

/* Form */
[data-testid="stForm"] {
    background-color: #111D30;
    border: 1px solid #263A54;
    border-radius: 16px;
    padding: 24px;
}

/* Labels */
.stApp label {
    color: #D5E2EF !important;
}

/* Input fields */
.stApp div[data-baseweb="input"] {
    background-color: #17263B;
    border: 1px solid #344A65;
    border-radius: 8px;
}

.stApp input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Dropdowns */
.stApp div[data-baseweb="select"] > div {
    background-color: #17263B !important;
    border-color: #344A65 !important;
    color: #FFFFFF !important;
}

.stApp div[data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* Dropdown options */
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
    background-color: #17263B !important;
}

div[data-baseweb="menu"] li {
    color: #FFFFFF !important;
}

/* Buttons */
.stFormSubmitButton button {
    background-color: #16B8CC !important;
    color: #071522 !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 48px;
    font-weight: 750 !important;
}

.stFormSubmitButton button:hover {
    background-color: #43D5E4 !important;
    color: #071522 !important;
}

/* Divider */
hr {
    border-color: #293B53 !important;
}

/* Result cards */
.result-card {
    padding: 28px;
    border-radius: 16px;
    margin-top: 20px;
    text-align: center;
}

.result-success {
    background-color: #102E2C;
    border: 1px solid #238B79;
}

.result-warning {
    background-color: #382329;
    border: 1px solid #B85B68;
}

.result-card h2 {
    font-size: 25px;
    font-weight: 800;
}

.result-card p {
    color: #D2DCE8 !important;
}

.footer-text {
    color: #8296AC;
    font-size: 12px;
    text-align: center;
}

/* Mobile layout */
@media (max-width: 768px) {
    .hero-box {
        padding: 20px;
    }

    .hero-box h1 {
        font-size: 25px;
    }

    [data-testid="stForm"] {
        padding: 14px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HERO SECTION
# =====================================================

st.markdown("""
<div class="hero-box">

    <p style="color:#43D5E4;">
        MACHINE LEARNING • HEALTHCARE
    </p>

    <h1>❤️ Heart Disease Prediction</h1>

    <p>
        An intelligent prediction system powered by
        Machine Learning. Enter patient health parameters
        to generate a model prediction.
    </p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 4. INTRODUCTION
# =====================================================

st.subheader("Patient Health Assessment")

st.write(
    "Enter the 13 patient parameters required by "
    "your trained machine learning model."
)


# =====================================================
# 5. PATIENT INPUT FORM
# =====================================================

with st.form("prediction_form"):

    st.subheader("👤 Personal Information")

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
                "Female (0)" if x == 0 else "Male (1)"
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=1,
            value=120
        )

    with col2:

        chol = st.number_input(
            "Cholesterol",
            min_value=1,
            value=200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar",
            [0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=1,
            value=150
        )

    st.divider()

    st.subheader("🫀 Cardiovascular Parameters")

    col1, col2 = st.columns(2)

    with col1:

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            [0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

    with col2:

        oldpeak = st.number_input(
            "Oldpeak",
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            [0, 1, 2]
        )

    st.divider()

    st.subheader("🔬 Additional Parameters")

    col1, col2 = st.columns(2)

    with col1:

        ca = st.selectbox(
            "Major Vessels (ca)",
            [0, 1, 2, 3, 4]
        )

    with col2:

        thal = st.selectbox(
            "Thalassemia (thal)",
            [0, 1, 2, 3]
        )

    st.write("")

    submit = st.form_submit_button(
        "🔍 Generate Prediction",
        use_container_width=True
    )


# =====================================================
# 6. PREDICTION LOGIC
# =====================================================

if submit:

    try:

        with st.spinner("Generating prediction..."):

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

            # Convert inputs into DataFrame
            final_data = data.get_data_as_dataframe()

            # Initialize prediction pipeline
            predict_pipeline = PredictPipeline()

            # Generate prediction
            prediction = predict_pipeline.predict(final_data)

            result = int(prediction[0])

        # =================================================
        # 7. DISPLAY RESULTS
        # =================================================

        st.divider()

        st.subheader("📊 Prediction Results")

        if result == 0:

            st.success("✅ Heart Disease Not Predicted")

            st.write(
                "The trained model returned prediction class 0."
            )

        elif result == 1:

            st.error("⚠️ Heart Disease Predicted")

            st.write(
                "The trained model returned prediction class 1."
            )

        else:

            st.warning(
                f"Unexpected model output: {result}"
            )

        # Display patient information
        with st.expander("📋 View Patient Input Details"):

            st.dataframe(
                final_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        with st.expander("View Error Details"):

            st.exception(e)


# =====================================================
# 8. FOOTER
# =====================================================

st.divider()

st.markdown(
    "<p class='footer-text'>"
    "❤️ CardioCare | Heart Disease Prediction System"
    "<br><br>"
    "Built using Python, Streamlit and Machine Learning."
    "</p>",
    unsafe_allow_html=True
)

st.warning(
    "Educational demonstration only. This application "
    "is not a medical diagnostic tool. Do not use its "
    "predictions to make medical decisions."
)
