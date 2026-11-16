
import streamlit as st
import pandas as pd

from src.Heart.pipeline.Prediction_pipeline import (
    CustomData,
    PredictPipeline
)

# Configure Streamlit page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Application title
st.title("❤️ Heart Disease Prediction System")
st.write(
    "Enter the patient details below to generate "
    "a machine learning prediction."
)

st.divider()

# Patient information form
with st.form("prediction_form"):

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=1, max_value=120, value=45
        )

        sex = st.selectbox(
            "Sex", [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

        cp = st.selectbox(
            "Chest Pain Type", [0, 1, 2, 3]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure", min_value=1, value=120
        )

        chol = st.number_input(
            "Cholesterol", min_value=1, value=200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar", [0, 1]
        )

        restecg = st.selectbox(
            "Resting ECG", [0, 1, 2]
        )

    with col2:
        thalach = st.number_input(
            "Maximum Heart Rate", min_value=1, value=150
        )

        exang = st.selectbox(
            "Exercise-Induced Angina", [0, 1]
        )

        oldpeak = st.number_input(
            "Oldpeak", value=1.0, step=0.1
        )

        slope = st.selectbox(
            "Slope", [0, 1, 2]
        )

        ca = st.selectbox(
            "Major Vessels (ca)", [0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thalassemia (thal)", [0, 1, 2, 3]
        )

    submit = st.form_submit_button(
        "Predict Heart Disease",
        use_container_width=True
    )


# Prediction logic - follows your original app.py
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

            st.write("### Input Data")
            st.dataframe(final_data, use_container_width=True)

            # Load model and make prediction
            predict_pipeline = PredictPipeline()

            # Generate prediction
            prediction = predict_pipeline.predict(final_data)

            result = int(prediction[0])

        
# Display result
st.divider()
st.subheader("Prediction Result")

if result == 0:
    st.success("✅ Heart Disease Not Predicted")

elif result == 1:
    st.error("⚠️ Heart Disease Predicted")

else:
    st.info(f"Unexpected model output: {result}")

        

        st.caption(
            "Class labels should be interpreted using your "
            "training dataset's target definition."
        )

    except Exception as e:
        st.error("Prediction failed!")
        st.exception(e)


st.divider()

st.warning(
    "Educational demonstration only. This application is not "
    "a medical diagnostic tool. Consult a qualified healthcare "
    "professional for medical advice."
)
