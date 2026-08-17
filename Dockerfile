FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /tfidf_vectorizer.pkl
COPY experiment_info.json /experiment_info.json
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000

CMD ["python", "app.py"]