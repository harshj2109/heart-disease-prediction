
import os
import sys
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.Heart.logger import logging
from src.Heart.utils.utils import save_object
from src.Heart.exception import customexception


@dataclass
class DataTransformationConfig:

    root_dir = Path(__file__).resolve().parents[3]

    preprocessor_obj_file_path = (
        root_dir / "Artifacts" / "Preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):

        try:
            logging.info("Data Transformation initiated")

            numerical_cols = [
                'age', 'sex', 'cp', 'trestbps', 'chol',
                'fbs', 'restecg', 'thalach', 'exang',
                'oldpeak', 'slope', 'ca', 'thal'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols)
                ],
                remainder='drop'
            )

            logging.info("Preprocessor created successfully")

            return preprocessor

        except Exception as e:
            logging.exception("Error creating preprocessor")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            print("Training Data Shape:", train_df.shape)
            print("Testing Data Shape:", test_df.shape)

            target_column_name = 'target'

            if target_column_name not in train_df.columns:
                raise ValueError("Target column missing in training data")

            if target_column_name not in test_df.columns:
                raise ValueError("Target column missing in testing data")

            numerical_cols = [
                'age', 'sex', 'cp', 'trestbps', 'chol',
                'fbs', 'restecg', 'thalach', 'exang',
                'oldpeak', 'slope', 'ca', 'thal'
            ]

            missing_columns = [
                col for col in numerical_cols
                if col not in train_df.columns or col not in test_df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            # Separate input features and target
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[target_column_name]

            logging.info("Input and target separation completed")

            # Create preprocessing object
            preprocessing_obj = self.get_data_transformation()

            # Fit on training data only
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # Transform test data
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            logging.info("Data preprocessing completed")

            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr,
                np.asarray(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.asarray(target_feature_test_df)
            ]

            # Save preprocessor
            self.data_transformation_config.preprocessor_obj_file_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            save_object(
                file_path=str(
                    self.data_transformation_config.preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            logging.info("Preprocessor saved successfully")

            print("Preprocessor saved successfully")

            return train_arr, test_arr

        except Exception as e:

            logging.exception("Error during data transformation")

            raise customexception(e, sys)
