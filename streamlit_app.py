
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
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #0B1220;
    color: #E5EDF7;
}

/* Main container */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Text */
.stApp p {
    color: #C5D2E3;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #F1F5F9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111D30;
    border-right: 1px solid #263A54;
}

/* Sidebar text */
[data-testid="stSidebar"] p {
    color: #C5D2E3;
}

/* Hero container */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #2A405B;
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

/* Number inputs */
.stApp input {
    color: #FFFFFF !important;
}

/* Input fields */
.stApp div[data-baseweb="input"] {
    background-color: #17263B;
    border-color: #344A65;
}

/* Select boxes */
.stApp div[data-baseweb="select"] > div {
    background-color: #17263B;
    border-color: #344A65;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    background-color: #16B8CC;
    color: #071522;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 46px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background-color: #43D5E4;
    color: #071522;
    border: none;
}

/* Divider */
hr {
    border-color: #293B53;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #14243A;
    border: 1px solid #2A405B;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #A9BDD2;
}

[data-testid="stMetricValue"] {
    color: #43D5E4;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #111D30;
    border: 1px solid #263A54;
    border-radius: 10px;
}

/* Alerts */
.stAlert {
    border-radius: 12px;
}

/* Footer */
footer {
    visibility: hidden;
}

/* Mobile */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    [data-testid="stForm"] {
        padding: 14px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("❤️ CardioCare")

    st.caption("HEART DISEASE PREDICTION")

    st.divider()

    st.subheader("About Project")

    st.write(
        "CardioCare is a machine learning application "
        "that analyzes patient health parameters and "
        "generates a heart disease prediction."
    )

    st.divider()

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Support Vector Classifier
    """)

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** SVC")

    st.write("**Input Features:** 13")

    st.write("**Task:** Binary Classification")

    st.divider()

    st.caption("Academic Machine Learning Project")


# =====================================================
# 4. HERO SECTION
# =====================================================

with st.container(border=True):

    st.caption("MACHINE LEARNING  •  HEALTHCARE")

    st.title("❤️ Heart Disease Prediction")

    st.write(
        "An intelligent prediction system powered by "
        "Machine Learning. Enter patient health parameters "
        "to generate a model prediction."
    )

    st.caption(
        "CARDIOCARE | MACHINE LEARNING PROJECT"
    )


# =====================================================
# 5. INTRODUCTION
# =====================================================

st.subheader("Patient Health Assessment")

st.write(
    "Enter the 13 patient parameters required by "
    "your trained machine learning model."
)


# =====================================================
# 6. PATIENT INPUT FORM
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
# 7. PREDICTION LOGIC
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
        # 8. DISPLAY RESULTS
        # =================================================

        st.divider()

        st.subheader("📊 Prediction Results")

        if result == 0:

            st.success(
                "✅ Heart Disease Not Predicted"
            )

            st.write(
                "The trained model returned prediction class 0."
            )

        elif result == 1:

            st.error(
                "⚠️ Heart Disease Predicted"
            )

            st.write(
                "The trained model returned prediction class 1."
            )

        else:

            st.warning(
                f"Unexpected model output: {result}"
            )

        # Display input information
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
# 9. FOOTER
# =====================================================

st.divider()

st.caption(
    "❤️ CardioCare | Heart Disease Prediction System"
)

st.caption(
    "Built using Python, Streamlit and Machine Learning."
)

st.warning(
    "Educational demonstration only. This application "
    "is not a medical diagnostic tool. Do not use its "
    "predictions to make medical decisions."
)
