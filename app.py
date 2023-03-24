from flask import Flask, request, jsonify, render_template
import pickle
import json
from textblob import TextBlob
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


def get_sentiment(text):
    if pd.isna(text):
        return None
    blob = TextBlob(text)
    return blob.sentiment.polarity
# Load the saved model
loaded_model = None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
with open("logistic (1).pkl", "rb") as f:
    loaded_model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON payload from the request
    # data = request.get_json(force=True)  # Added force=True to avoid issues with content-type
    data = json.loads(request.data.decode('utf-8'))
    # Get the sentiment value from the JSON payload
    # sentiment = data.get("sentiment", None)

    if data is None:
        return jsonify({"error": "Sentiment value is missing"}), 400
    sent=get_sentiment(data)
    prediction = loaded_model.predict([[sent]])[0]
    return jsonify({"is_popular": bool(prediction)})

if __name__ == "__main__":
    app.run(debug=True)