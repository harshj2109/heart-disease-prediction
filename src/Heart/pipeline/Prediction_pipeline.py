
import os
import sys
import pandas as pd

from pathlib import Path

from src.Heart.logger import logging
from src.Heart.utils.utils import load_object
from src.Heart.exception import customexception


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:
            # Get project root directory
            root_dir = Path(__file__).resolve().parents[3]

            # Model and preprocessor paths
            preprocessor_path = root_dir / "Artifacts" / "Preprocessor.pkl"
            model_path = root_dir / "Artifacts" / "Model.pkl"

            # Check whether files exist
            if not preprocessor_path.exists():
                raise FileNotFoundError(
                    f"Preprocessor not found: {preprocessor_path}"
                )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}"
                )

            # Load saved objects
            preprocessor = load_object(str(preprocessor_path))
            model = load_object(str(model_path))

            logging.info("Model and preprocessor loaded successfully")

            # Transform input features
            scaled_data = preprocessor.transform(features)

            logging.info("Input data transformed successfully")

            # Generate prediction
            prediction = model.predict(scaled_data)

            logging.info(f"Prediction generated: {prediction}")

            return prediction

        except Exception as e:
            logging.exception("Prediction pipeline failed")
            raise customexception(e, sys)


class CustomData:

    def __init__(
        self,
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ):

        self.age = int(age)
        self.sex = int(sex)
        self.cp = int(cp)
        self.trestbps = int(trestbps)
        self.chol = int(chol)
        self.fbs = int(fbs)
        self.restecg = int(restecg)
        self.thalach = int(thalach)
        self.exang = int(exang)
        self.oldpeak = float(oldpeak)
        self.slope = int(slope)
        self.ca = int(ca)
        self.thal = int(thal)

    # IMPORTANT: This method must be INSIDE CustomData

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {
                'age': [self.age],
                'sex': [self.sex],
                'cp': [self.cp],
                'trestbps': [self.trestbps],
                'chol': [self.chol],
                'fbs': [self.fbs],
                'restecg': [self.restecg],
                'thalach': [self.thalach],
                'exang': [self.exang],
                'oldpeak': [self.oldpeak],
                'slope': [self.slope],
                'ca': [self.ca],
                'thal': [self.thal]
            }

            df = pd.DataFrame(custom_data_input_dict)

            print("Input DataFrame:")
            print(df)

            print("\nData Types:")
            print(df.dtypes)

            logging.info("Prediction dataframe created successfully")

            return df

        except Exception as e:

            logging.exception("Error creating prediction dataframe")

            raise customexception(e, sys)
