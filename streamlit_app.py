from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "Artifacts" / "Model.pkl"
PREPROCESSOR_PATH = PROJECT_DIR / "Artifacts" / "Preprocessor.pkl"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]


# -------------------- Load saved ML artifacts --------------------
@st.cache_resource
def load_artifacts():
    """Load the exact model and preprocessor saved by the training project."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor file not found: {PREPROCESSOR_PATH}"
        )

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    with PREPROCESSOR_PATH.open("rb") as prep_file:
        preprocessor = pickle.load(prep_file)

    return model, preprocessor


# -------------------- Header --------------------
st.title("❤️ Heart Disease Prediction")
st.caption("Machine-learning demonstration using your trained Support Vector Classifier (SVC).")

st.info(
    "Enter the 13 input features and select **Predict**. "
    "The app applies the saved preprocessor before calling the trained model."
)

with st.expander("About this project"):
    st.write(
        "This application uses the Model.pkl and Preprocessor.pkl artifacts "
        "from the original project. The input columns and order match the "
        "training pipeline."
    )
    st.write("Input features: " + ", ".join(FEATURE_COLUMNS))


# -------------------- Input form --------------------
st.subheader("Patient information")

with st.form("heart_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
        sex_label = st.selectbox("Sex", ["Female (0)", "Male (1)"])
        cp_label = st.selectbox(
            "Chest pain type",
            [
                "Typical angina (0)",
                "Atypical angina (1)",
                "Non-anginal pain (2)",
                "Asymptomatic (3)",
            ],
        )
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)", min_value=1, max_value=300, value=120
        )
        chol = st.number_input(
            "Serum cholesterol (mg/dl)", min_value=1, max_value=1000, value=200
        )
        fbs_label = st.selectbox(
            "Fasting blood sugar > 120 mg/dl", ["No (0)", "Yes (1)"]
        )
        restecg_label = st.selectbox(
            "Resting ECG",
            [
                "Normal (0)",
                "ST-T wave abnormality (1)",
                "Left ventricular hypertrophy (2)",
            ],
        )

    with col2:
        thalach = st.number_input(
            "Maximum heart rate achieved", min_value=1, max_value=300, value=150
        )
        exang_label = st.selectbox(
            "Exercise-induced angina", ["No (0)", "Yes (1)"]
        )
        oldpeak = st.number_input(
            "Oldpeak (ST depression)", min_value=-5.0, max_value=10.0,
            value=1.0, step=0.1, format="%.1f"
        )
        slope_label = st.selectbox(
            "Slope of peak exercise ST segment",
            ["Upsloping (0)", "Flat (1)", "Downsloping (2)"],
        )
        ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3, 4])
        thal_label = st.selectbox(
            "Thalassemia (thal)",
            ["Normal (0)", "Fixed defect (1)", "Reversible defect (2)", "Other/unknown (3)"],
        )

    submitted = st.form_submit_button(
        "🔍 Predict",
        use_container_width=True,
        type="primary",
    )


# -------------------- Prediction --------------------
if submitted:
    input_values = {
        "age": int(age),
        "sex": 0 if sex_label.startswith("Female") else 1,
        "cp": int(cp_label.split("(")[-1].rstrip(")")),
        "trestbps": int(trestbps),
        "chol": int(chol),
        "fbs": 0 if fbs_label.startswith("No") else 1,
        "restecg": int(restecg_label.split("(")[-1].rstrip(")")),
        "thalach": int(thalach),
        "exang": 0 if exang_label.startswith("No") else 1,
        "oldpeak": float(oldpeak),
        "slope": int(slope_label.split("(")[-1].rstrip(")")),
        "ca": int(ca),
        "thal": int(thal_label.split("(")[-1].rstrip(")")),
    }

    input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)

    try:
        with st.spinner("Running the trained machine-learning model..."):
            model, preprocessor = load_artifacts()
            transformed_input = preprocessor.transform(input_df)
            prediction = int(model.predict(transformed_input)[0])

        st.divider()
        st.subheader("Prediction result")

        if prediction == 0:
            st.success("The model predicted class 0.")
        elif prediction == 1:
            st.warning("The model predicted class 1.")
        else:
            st.info(f"The model returned class: {prediction}")

        st.caption(
            "This is the model's output label, not a clinical assessment. "
            "The class labels should be interpreted according to the dataset "
            "and training documentation."
        )

        with st.expander("View submitted input values"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error("The prediction could not be completed.")
        st.exception(exc)


# -------------------- Medical disclaimer --------------------
st.divider()
st.caption(
    "Educational project only. This tool is not a medical device and must not "
    "be used to diagnose, treat, or rule out heart disease. Consult a qualified "
    "healthcare professional for medical advice."
)
