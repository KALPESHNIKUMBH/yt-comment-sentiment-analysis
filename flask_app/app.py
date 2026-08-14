import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os
import json
import re
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)

# -----------------------------
# Text preprocessing
# -----------------------------
def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        stop_words = set(stopwords.words("english")) - {
            "not", "but", "however", "no", "yet"
        }

        comment = " ".join(
            [word for word in comment.split() if word not in stop_words]
        )

        lemmatizer = WordNetLemmatizer()
        comment = " ".join(
            [lemmatizer.lemmatize(word) for word in comment.split()]
        )

        return comment

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return comment

# -----------------------------
# Load MLflow model + vectorizer
# -----------------------------
def load_model_and_vectorizer():
    mlflow.set_tracking_uri(
        "http://ec2-3-80-142-29.compute-1.amazonaws.com:8000"
    )

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    info_path = os.path.join(base_dir, "experiment_info.json")
    vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")

    with open(info_path, "r") as f:
        model_info = json.load(f)

    model_uri = model_info["model_uri"]

    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():
    return "Welcome to our Flask API"

# -----------------------------
# Predict
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    comments = data.get("comments")

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [
            preprocess_comment(comment)
            for comment in comments
        ]

        transformed = vectorizer.transform(preprocessed_comments)

        X_df = pd.DataFrame(
            transformed.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        predictions = model.predict(X_df).tolist()
        predictions = [str(pred) for pred in predictions]

        response = [
            {
                "comment": comment,
                "sentiment": sentiment
            }
            for comment, sentiment in zip(comments, predictions)
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# -----------------------------
# Predict with timestamps
# -----------------------------
@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    data = request.get_json()
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item["text"] for item in comments_data]
        timestamps = [item["timestamp"] for item in comments_data]

        preprocessed_comments = [
            preprocess_comment(comment)
            for comment in comments
        ]

        transformed = vectorizer.transform(preprocessed_comments)

        X_df = pd.DataFrame(
            transformed.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        predictions = model.predict(X_df).tolist()
        predictions = [str(pred) for pred in predictions]

        response = [
            {
                "comment": comment,
                "sentiment": sentiment,
                "timestamp": timestamp
            }
            for comment, sentiment, timestamp in zip(
                comments,
                predictions,
                timestamps
            )
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# -----------------------------
# Pie chart
# -----------------------------
@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0))
        ]

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140
        )

        img = io.BytesIO()
        plt.savefig(img, format="PNG")
        img.seek(0)
        plt.close()

        return send_file(img, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Word cloud
# -----------------------------
@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get("comments")

        text = " ".join(
            preprocess_comment(comment)
            for comment in comments
        )

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues"
        ).generate(text)

        img = io.BytesIO()
        wordcloud.to_image().save(img, format="PNG")
        img.seek(0)

        return send_file(img, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Trend graph
# -----------------------------
@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data")

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        monthly = (
            df.resample("M")["sentiment"]
              .value_counts()
              .unstack(fill_value=0)
        )

        monthly = (
            monthly.T / monthly.sum(axis=1)
        ).T * 100

        for s in [-1, 0, 1]:
            if s not in monthly.columns:
                monthly[s] = 0

        monthly = monthly[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))

        colors = {
            -1: "red",
            0: "gray",
            1: "green"
        }

        labels = {
            -1: "Negative",
            0: "Neutral",
            1: "Positive"
        }

        for s in [-1, 0, 1]:
            plt.plot(
                monthly.index,
                monthly[s],
                marker="o",
                label=labels[s],
                color=colors[s]
            )

        plt.gca().xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )

        plt.legend()
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format="PNG")
        img.seek(0)
        plt.close()

        return send_file(img, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)