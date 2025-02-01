import numpy as np
from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all domains (for testing purposes)
CORS(app)

# Load the trained model and label encoder
model = joblib.load("crop_recommendation_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return "Welcome to the Crop Recommendation API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid request"}), 400

        # Extract input features
        N = int(data["N"])
        P = int(data["P"])
        K = int(data["K"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])

        # Make prediction using the trained model
        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])

        # Convert NumPy object to Python type
        predicted_crop = label_encoder.inverse_transform([prediction[0]])[0]

        # Return result as JSON
        return jsonify({"prediction": predicted_crop})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 5000))  # Get the assigned port or default to 5000
    app.run(host="0.0.0.0", port=PORT, debug=True)
