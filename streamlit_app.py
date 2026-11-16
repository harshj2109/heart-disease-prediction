
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
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. CUSTOM DARK THEME
# =====================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #0B1220;
    color: #E5EDF7;
}

/* Hide unnecessary Streamlit header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Main content spacing */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* General text */
.stMarkdown,
.stMarkdown p,
.stMarkdown li {
    color: #C5D2E3;
}

/* Headings */
h1, h2, h3 {
    color: #F1F5F9 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111D30;
    border-right: 1px solid #243750;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #C5D2E3 !important;
}

/* Hero section */
.hero {
    background: linear-gradient(
        120deg,
        #142A43,
        #103D55
    );
    border: 1px solid #24536A;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 28px;
}

.hero-title {
    color: #FFFFFF !important;
    font-size: 34px;
    font-weight: 800;
    margin: 0 0 10px 0;
}

.hero-subtitle {
    color: #B9DCEB !important;
    font-size: 15px;
    margin: 0;
    line-height: 1.7;
}

.hero-tag {
    display: inline-block;
    background: #16445B;
    color: #63D9E8;
    border: 1px solid #26758A;
    border-radius: 30px;
    padding: 6px 13px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 16px;
}

/* Section heading */
.section-heading {
    color: #E8F3FC !important;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-description {
    color: #91A6BD !important;
    font-size: 13px;
    margin-bottom: 18px;
}

/* Input labels */
.stApp label {
    color: #C9D8E8 !important;
    font-weight: 600 !important;
}

/* Input boxes */
.stApp div[data-baseweb="input"] {
    background-color: #17263B;
    border: 1px solid #344A65;
    border-radius: 9px;
}

.stApp div[data-baseweb="input"]:focus-within {
    border-color: #25C4D8;
    box-shadow: 0 0 0 1px #25C4D8;
}

.stApp div[data-baseweb="input"] input {
    color: #FFFFFF !important;
    background-color: transparent !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Dropdowns */
.stApp div[data-baseweb="select"] > div {
    background-color: #17263B !important;
    border-color: #344A65 !important;
    border-radius: 9px;
    color: #FFFFFF !important;
}

.stApp div[data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* Dropdown popup */
div[data-baseweb="popover"] {
    background-color: #17263B !important;
}

div[data-baseweb="menu"] {
    background-color: #17263B !important;
}

div[data-baseweb="menu"] li {
    color: #FFFFFF !important;
}

/* Form border */
[data-testid="stForm"] {
    background-color: #111D30;
    border: 1px solid #263A54;
    border-radius: 16px;
    padding: 25px;
}

/* Buttons */
.stFormSubmitButton button,
.stButton button {
    background: #16B8CC !important;
    color: #071522 !important;
    border: none !important;
    border-radius: 10px !important;
    height: 48px;
    font-weight: 750 !important;
    transition: all 0.2s ease;
}

.stFormSubmitButton button:hover,
.stButton button:hover {
    background: #43D5E4 !important;
    color: #071522 !important;
    border: none !important;
}

/* Divider */
hr {
    border-color: #293B53 !important;
}

/* Result cards */
.result-card {
    border-radius: 16px;
    padding: 30px;
    margin: 20px 0;
    text-align: center;
}

.result-negative {
    background-color: #102E2C;
    border: 1px solid #238B79;
}

.result-positive {
    background-color: #382329;
    border: 1px solid #B85B68;
}

.result-icon {
    font-size: 42px;
    margin-bottom: 12px;
}

.result-title {
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 10px;
}

.result-description {
    color: #D2DCE8 !important;
    font-size: 14px;
    line-height: 1.6;
}

/* Information cards */
.info-card {
    background: #14243A;
    border: 1px solid #2A405B;
    border-radius: 12px;
    padding: 18px;
    height: 100%;
}

.info-label {
    color: #8FA8C2 !important;
    font-size: 12px;
    margin-bottom: 7px;
}

.info-value {
    color: #42D3E2 !important;
    font-size: 18px;
    font-weight: 750;
}

/* Footer */
.footer {
    text-align: center;
    color: #8296AC !important;
    font-size: 12px;
    line-height: 1.8;
    padding: 25px 0 10px;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .hero {
        padding: 23px;
    }

    .hero-title {
        font-size: 26px;
    }

    [data-testid="stForm"] {
        padding: 15px;
    }

    .result-title {
        font-size: 21px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#43D5E4;'>❤️ CardioCare</h2>",
        unsafe_allow_html=True
    )

    st.caption("HEART DISEASE PREDICTION SYSTEM")

    st.divider()

    st.markdown("### About the Project")

    st.write(
        "CardioCare is a machine learning application "
        "that analyzes patient health parameters and "
        "generates a heart disease prediction."
    )

    st.divider()

    st.markdown("### Technology Stack")

    st.markdown("""
    - 🐍 Python
    - 🧠 Scikit-learn
    - 📊 Pandas
    - ⚡ Streamlit
    - 🔬 Support Vector Classifier
    """)

    st.divider()

    st.markdown("### Model Information")

    st.write("**Algorithm:** SVC")

    st.write("**Input Features:** 13")

    st.write("**Prediction:** Binary Classification")

    st.divider()

    st.caption("Academic Machine Learning Project")


# =====================================================
# 4. HERO SECTION
# =====================================================

st.markdown("""
<div class="hero">

    <div class="hero-tag">
        MACHINE LEARNING • HEALTHCARE
    </div>

    <div class="hero-title">
        ❤️ Heart Disease Prediction
    </div>

    <p class="hero-subtitle">
        An intelligent prediction system powered by
        Machine Learning. Enter patient health parameters
        to generate a model prediction.
    </p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 5. INTRODUCTION
# =====================================================

st.markdown(
    "<div class='section-heading'>Patient Assessment</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='section-description'>"
    "Enter all 13 parameters required by the trained model."
    "</div>",
    unsafe_allow_html=True
)


# =====================================================
# 6. PATIENT INPUT FORM
# =====================================================

with st.form("prediction_form"):

    # Personal details
    st.markdown(
        "<div class='section-heading'>👤 Personal Information</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-description'>"
        "Basic patient details and measurements."
        "</div>",
        unsafe_allow_html=True
    )

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

    # Cardiovascular parameters
    st.markdown(
        "<div class='section-heading'>🫀 Cardiovascular Parameters</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-description'>"
        "Enter the cardiovascular and exercise-related features."
        "</div>",
        unsafe_allow_html=True
    )

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
            min_value=-5.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            [0, 1, 2]
        )

    st.divider()

    # Additional parameters
    st.markdown(
        "<div class='section-heading'>🔬 Additional Parameters</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-description'>"
        "Additional model input features."
        "</div>",
        unsafe_allow_html=True
    )

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
        "🔍  Generate Prediction",
        use_container_width=True
    )


# =====================================================
# 7. PREDICTION PIPELINE
# =====================================================

if submit:

    try:

        with st.spinner(
            "Processing patient information and generating prediction..."
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

            # Convert inputs into DataFrame
            final_data = data.get_data_as_dataframe()

            # Initialize prediction pipeline
            predict_pipeline = PredictPipeline()

            # Generate prediction
            prediction = predict_pipeline.predict(final_data)

            result = int(prediction[0])

        # =============================================
        # 8. DISPLAY PREDICTION RESULT
        # =============================================

        st.divider()

        st.markdown(
            "<div class='section-heading'>"
            "📊 Prediction Results"
            "</div>",
            unsafe_allow_html=True
        )

        if result == 0:

            st.markdown("""
            <div class="result-card result-negative">

                <div class="result-icon">✅</div>

                <div class="result-title" style="color:#58D6AA;">
                    Heart Disease Not Predicted
                </div>

                <p class="result-description">
                    The trained machine learning model returned
                    prediction class 0 for the supplied input.
                </p>

            </div>
            """, unsafe_allow_html=True)

        elif result == 1:

            st.markdown("""
            <div class="result-card result-positive">

                <div class="result-icon">⚠️</div>

                <div class="result-title" style="color:#FF929D;">
                    Heart Disease Predicted
                </div>

                <p class="result-description">
                    The trained machine learning model returned
                    prediction class 1 for the supplied input.
                </p>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.warning(
                f"Unexpected model output: {result}"
            )

        # =============================================
        # 9. INPUT SUMMARY
        # =============================================

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

        with st.expander("View error details"):

            st.exception(e)


# =====================================================
# 10. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">

    <strong style="color:#43D5E4;">
        CardioCare | Heart Disease Prediction System
    </strong>

    <br>

    Built with Python, Streamlit and Machine Learning.

    <br><br>

    Educational demonstration only. This application is
    not a medical diagnostic tool and must not be used
    to make medical decisions.

</div>
""", unsafe_allow_html=True)
