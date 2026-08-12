import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import traceback
import logging

# Create logger
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

# Prevent duplicate logs
logger.propagate = False

# Create logs directory (optional)
os.makedirs("logs", exist_ok=True)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# File handler (only ERROR and above)
file_handler = logging.FileHandler("logs/errors.log")
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

# Add handlers only once
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')


# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise



def main():
    logger.info("========== Data preprocessing started ==========")

    try:
        # Current working directory
        cwd = os.getcwd()
        logger.debug(f"Current working directory: {cwd}")

        train_path = "data/raw/train_data.csv"
        test_path = "data/raw/test_data.csv"

        logger.debug(f"Train file path: {os.path.abspath(train_path)}")
        logger.debug(f"Test file path: {os.path.abspath(test_path)}")

        logger.debug(f"Train file exists: {os.path.exists(train_path)}")
        logger.debug(f"Test file exists: {os.path.exists(test_path)}")

        # ---------------- Load train data ----------------
        try:
            logger.info("Loading training data...")
            train_data = pd.read_csv(train_path)
            logger.info(f"Training data loaded successfully. Shape: {train_data.shape}")
        except Exception as e:
            logger.exception("Error while loading training data")
            raise

        # ---------------- Load test data ----------------
        try:
            logger.info("Loading test data...")
            test_data = pd.read_csv(test_path)
            logger.info(f"Test data loaded successfully. Shape: {test_data.shape}")
        except Exception as e:
            logger.exception("Error while loading test data")
            raise

        # ---------------- Preprocess train data ----------------
        try:
            logger.info("Preprocessing training data...")
            train_processed_data = normalize_text(train_data)
            logger.info("Training data preprocessing completed")
        except Exception as e:
            logger.exception("Error during training data preprocessing")
            raise

        # ---------------- Preprocess test data ----------------
        try:
            logger.info("Preprocessing test data...")
            test_processed_data = normalize_text(test_data)
            logger.info("Test data preprocessing completed")
        except Exception as e:
            logger.exception("Error during test data preprocessing")
            raise

        # ---------------- Save processed data ----------------
        try:
            logger.info("Saving processed data...")
            save_data(train_processed_data, test_processed_data, data_path="./data")
            logger.info("Processed data saved successfully")
        except Exception as e:
            logger.exception("Error while saving processed data")
            raise

        logger.info("========== Data preprocessing completed successfully ==========")

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        logger.debug("Full traceback:")
        logger.debug(traceback.format_exc())
        print(f"Error: {e}")

if __name__ == "__main__":
    main()