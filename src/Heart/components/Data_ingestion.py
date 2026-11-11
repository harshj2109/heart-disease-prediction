
import os
import sys
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.Heart.logger import logging
from src.Heart.exception import customexception


@dataclass
class DataIngestionConfig:

    # Project root directory
    root_dir: Path = Path(__file__).resolve().parents[3]

    @property
    def raw_data_path(self):
        return self.root_dir / "Artifacts" / "raw_data.csv"

    @property
    def train_data_path(self):
        return self.root_dir / "Artifacts" / "train_data.csv"

    @property
    def test_data_path(self):
        return self.root_dir / "Artifacts" / "test_data.csv"


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Data ingestion started")

        try:
            # Locate the original dataset
            data_path = (
                self.ingestion_config.root_dir
                / "Notebook_Experiments"
                / "Data"
                / "heart.csv"
            )

            # Check whether dataset exists
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at: {data_path}"
                )

            # Read dataset
            data = pd.read_csv(data_path)

            logging.info("Dataset successfully loaded")

            print("Dataset Shape:", data.shape)
            print("Dataset Columns:", data.columns.tolist())

            # Create Artifacts directory
            self.ingestion_config.raw_data_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Save raw data
            data.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            logging.info("Raw data saved successfully")

            # Split data into training and testing
            train_data, test_data = train_test_split(
                data,
                test_size=0.2,
                random_state=42
            )

            # Save training data
            train_data.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            # Save testing data
            test_data.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info("Training and testing data saved")
            logging.info("Data ingestion completed successfully")

            print("Training Data Shape:", train_data.shape)
            print("Testing Data Shape:", test_data.shape)

            # IMPORTANT: Return training path first
            return (
                str(self.ingestion_config.train_data_path),
                str(self.ingestion_config.test_data_path)
            )

        except Exception as e:

            logging.exception("Error occurred during data ingestion")

            raise customexception(e, sys)
