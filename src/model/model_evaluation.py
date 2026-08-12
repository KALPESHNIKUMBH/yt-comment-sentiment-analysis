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

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature


# ----------------------------
# Logging configuration
# ----------------------------
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


# ----------------------------
# Utility functions
# ----------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna("", inplace=True)
    logger.debug(f"Data loaded from {file_path}")
    return df


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.debug(f"Model loaded from {model_path}")
    return model


def load_vectorizer(vectorizer_path):
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    logger.debug(f"Vectorizer loaded from {vectorizer_path}")
    return vectorizer


def load_params(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    logger.debug(f"Parameters loaded from {params_path}")
    return params


def log_confusion_matrix(cm, name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    file_name = f"confusion_matrix_{name}.png"
    plt.savefig(file_name)
    plt.close()

    mlflow.log_artifact(file_name)


def save_model_info(run_id, model_path, file_path):
    with open(file_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "model_path": model_path,
            },
            f,
            indent=4,
        )


def log_metrics(prefix, report):
    mlflow.log_metric(
        f"{prefix}_precision",
        report["weighted avg"]["precision"],
    )
    mlflow.log_metric(
        f"{prefix}_recall",
        report["weighted avg"]["recall"],
    )
    mlflow.log_metric(
        f"{prefix}_f1_score",
        report["weighted avg"]["f1-score"],
    )


# ----------------------------
# Main
# ----------------------------
def main():
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    mlflow.set_tracking_uri(
        "http://ec2-3-80-142-29.compute-1.amazonaws.com:8000/"
    )
    mlflow.set_experiment("dvc-pipeline-runs")

    with mlflow.start_run() as run:
        try:
            params = load_params(os.path.join(root_dir, "params.yaml"))

            for key, value in params.items():
                mlflow.log_param(key, value)

            model = load_model(os.path.join(root_dir, "lgbm_model.pkl"))
            vectorizer = load_vectorizer(
                os.path.join(root_dir, "tfidf_vectorizer.pkl")
            )

            train_data = load_data(
                os.path.join(root_dir, "data/interim/train_processed.csv")
            )
            test_data = load_data(
                os.path.join(root_dir, "data/interim/test_processed.csv")
            )

            X_train = vectorizer.transform(train_data["clean_comment"])
            y_train = train_data["category"]

            X_test = vectorizer.transform(test_data["clean_comment"])
            y_test = test_data["category"]

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")

            train_report = classification_report(
                y_train,
                y_train_pred,
                output_dict=True,
            )
            test_report = classification_report(
                y_test,
                y_test_pred,
                output_dict=True,
            )

            print("\nTraining Classification Report")
            print(classification_report(y_train, y_train_pred))

            print("\nTest Classification Report")
            print(classification_report(y_test, y_test_pred))

            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)

            log_metrics("train", train_report)
            log_metrics("test", test_report)

            input_example = pd.DataFrame(
                X_test.toarray()[:5],
                columns=vectorizer.get_feature_names_out(),
            )

            signature = infer_signature(
                input_example,
                model.predict(X_test[:5]),
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="lgbm_model",
                signature=signature,
                input_example=input_example,
                skops_trusted_types=[
                    "collections.OrderedDict",
                    "lightgbm.basic.Booster",
                    "lightgbm.sklearn.LGBMClassifier",
                ],
            )

            save_model_info(
                run.info.run_id,
                "lgbm_model",
                "experiment_info.json",
            )

            mlflow.log_artifact(
                os.path.join(root_dir, "tfidf_vectorizer.pkl")
            )

            train_cm = confusion_matrix(y_train, y_train_pred)
            test_cm = confusion_matrix(y_test, y_test_pred)

            log_confusion_matrix(train_cm, "train_data")
            log_confusion_matrix(test_cm, "test_data")

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

            logger.info("Model evaluation completed successfully")

        except Exception:
            logger.exception("Failed to complete model evaluation")
            raise


if __name__ == "__main__":
    main()