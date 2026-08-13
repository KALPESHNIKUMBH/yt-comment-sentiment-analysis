import os
import json
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature

# =========================
# Logging configuration
# =========================
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_evaluation_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =========================
# Utility functions
# =========================
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna("", inplace=True)
    logger.debug(f"Loaded data from {file_path}")
    return df


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.debug(f"Loaded model from {model_path}")
    return model


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    logger.debug(f"Loaded vectorizer from {vectorizer_path}")
    return vectorizer


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    logger.debug(f"Loaded parameters from {params_path}")
    return params


def log_params_recursive(params, prefix=""):
    for key, value in params.items():
        if isinstance(value, dict):
            log_params_recursive(value, prefix=f"{prefix}{key}.")
        else:
            mlflow.log_param(f"{prefix}{key}", value)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    logger.debug("Model evaluation completed")
    return report, cm


def log_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    file_path = "confusion_matrix_test.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(file_path)

    if os.path.exists(file_path):
        os.remove(file_path)


def save_model_info(run_id: str, model_uri: str, file_path: str):
    info = {
        "run_id": run_id,
        "model_uri": model_uri
    }

    with open(file_path, "w") as f:
        json.dump(info, f, indent=4)

    logger.debug(f"Saved model info to {file_path}")

# =========================
# Main
# =========================
def main():
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    # MLflow configuration
    mlflow.set_tracking_uri(
        "http://ec2-3-80-142-29.compute-1.amazonaws.com:8000"
    )
    mlflow.set_experiment("dvc-pipeline-runs")

    try:
        with mlflow.start_run() as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")

            # Load parameters
            params = load_params(os.path.join(root_dir, "params.yaml"))
            log_params_recursive(params)

            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, "lgbm_model.pkl"))
            vectorizer = load_vectorizer(
                os.path.join(root_dir, "tfidf_vectorizer.pkl")
            )

            # Load test data
            test_data = load_data(
                os.path.join(root_dir, "data/interim/test_processed.csv")
            )

            X_test = vectorizer.transform(test_data["clean_comment"].values)
            y_test = test_data["category"].values

            # Signature
            input_example = pd.DataFrame(
                X_test[:5].toarray(),
                columns=vectorizer.get_feature_names_out()
            )

            signature = infer_signature(
                input_example,
                model.predict(X_test[:5])
            )

            # Log model (pickle serialization fixes LightGBM issue)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="lgbm_model",
                signature=signature,
                input_example=input_example,
                serialization_format="pickle"
            )

            model_uri = f"runs:/{run.info.run_id}/lgbm_model"

            save_model_info(
                run.info.run_id,
                model_uri,
                os.path.join(root_dir, "experiment_info.json")
            )

            mlflow.log_artifact(
                os.path.join(root_dir, "experiment_info.json")
            )

            # Log vectorizer
            mlflow.log_artifact(
                os.path.join(root_dir, "tfidf_vectorizer.pkl")
            )

            # Evaluate model
            report, cm = evaluate_model(model, X_test, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metric(
                        f"test_{label}_precision",
                        metrics.get("precision", 0.0)
                    )
                    mlflow.log_metric(
                        f"test_{label}_recall",
                        metrics.get("recall", 0.0)
                    )
                    mlflow.log_metric(
                        f"test_{label}_f1_score",
                        metrics.get("f1-score", 0.0)
                    )

            if "accuracy" in report:
                mlflow.log_metric("test_accuracy", report["accuracy"])

            log_confusion_matrix(cm)

            # Tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

            logger.info("Model evaluation completed successfully")

    except Exception as e:
        logger.exception("Model evaluation failed")
        raise


if __name__ == "__main__":
    main()