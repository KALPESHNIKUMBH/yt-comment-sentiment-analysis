import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging


import logging

# This code helps with main logging
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

# Prevent duplicate logs
logger.propagate = False

# For terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# For error.log
file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers only once
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)



def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        logger.info(f"Parameters loaded successfully from {params_path}")
        return params

    except FileNotFoundError:
        logger.error(f"File not found: {params_path}")
        raise

    except yaml.YAMLError:
        logger.error(f"Invalid YAML file: {params_path}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error while loading parameters: {e}")
        raise
    
    
def load_data(data_url: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from: {data_url}")
        df = pd.read_csv(data_url)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}", exc_info=True)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting data preprocessing")

        initial_shape = df.shape

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df["clean_comment"].str.strip() != ""]

        logger.info(
            f"Data preprocessing completed. Initial shape: {initial_shape}, Final shape: {df.shape}"
        )

        return df

    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}", exc_info=True)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        logger.info("Saving train and test data")

        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_file_path = os.path.join(raw_data_path, "train_data.csv")
        test_file_path = os.path.join(raw_data_path, "test_data.csv")

        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)

        logger.info(
            f"Train data saved to {train_file_path} and test data saved to {test_file_path}"
        )

    except Exception as e:
        logger.error(f"Error saving data: {e}", exc_info=True)
        raise
    
def main():
    try:
        logger.info("================ Data Ingestion Pipeline Started ================")

        # Define paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Current directory: {current_dir}")

        params_path = os.path.join(current_dir, "../../params.yaml")
        data_path = os.path.join(current_dir, "../../data")

        logger.info(f"Params path: {params_path}")
        logger.info(f"Data path: {data_path}")

        # Load parameters
        logger.info("Loading parameters from params.yaml")
        params = load_params(params_path=params_path)

        test_size = params["data_ingestion"]["test_size"]
        logger.info(f"Test size loaded: {test_size}")

        # Load dataset
        data_url = (
            "https://raw.githubusercontent.com/Himanshu-1703/"
            "reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        )
        logger.info(f"Loading dataset from URL: {data_url}")

        df = load_data(data_url=data_url)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

        # Preprocess data
        logger.info("Starting data preprocessing")
        final_df = preprocess_data(df)
        logger.info(f"Preprocessing completed. Final shape: {final_df.shape}")

        # Train-test split
        logger.info("Performing train-test split")
        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        logger.info(f"Train data shape: {train_data.shape}")
        logger.info(f"Test data shape: {test_data.shape}")

        # Save data
        logger.info("Saving processed train and test data")
        save_data(
            train_data=train_data,
            test_data=test_data,
            data_path=data_path
        )

        logger.info("Train and test data saved successfully")
        logger.info("================ Data Ingestion Pipeline Completed Successfully ================")

    except Exception as e:
        logger.exception("Data ingestion pipeline failed due to an unexpected error")
        raise


if __name__ == "__main__":
    main()
    
    