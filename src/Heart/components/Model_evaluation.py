
import os
import mlflow
import mlflow.sklearn

from src.Heart.utils.utils import load_object

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluation:

    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):

        accuracy = accuracy_score(actual, pred)

        precision = precision_score(
            actual, pred, zero_division=0
        )

        recall = recall_score(
            actual, pred, zero_division=0
        )

        f1 = f1_score(
            actual, pred, zero_division=0
        )

        return accuracy, precision, recall, f1

    def initate_model_evaluation(self, train_array, test_array):

        try:
            # Configure local SQLite tracking backend
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

            # Create or select experiment
            mlflow.set_experiment("Heart-Disease-Prediction")

            # Separate input and target
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Load trained model
            model_path = os.path.join(
                "Artifacts", "Model.pkl"
            )

            model = load_object(model_path)

            # Start MLflow run
            with mlflow.start_run():

                # Generate predictions
                predictions = model.predict(X_test)

                # Calculate evaluation metrics
                accuracy, precision, recall, f1 = (
                    self.eval_metrics(y_test, predictions)
                )

                print("\nModel Evaluation Results")
                print("------------------------")
                print("Accuracy:", accuracy)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1 Score:", f1)

                # Log metrics
                mlflow.log_metric(
                    "Testing Accuracy", accuracy
                )

                mlflow.log_metric(
                    "Precision Score", precision
                )

                mlflow.log_metric(
                    "Recall Score", recall
                )

                mlflow.log_metric(
                    "F1 Score", f1
                )

                # Log trained model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="Model"
                )

                print("\nModel logged successfully in MLflow")

        except Exception as e:
            print("Error during model evaluation:", str(e))
            raise
