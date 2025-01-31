import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and label encoder
model_path = os.path.join(os.path.dirname(__file__), 'api', 'crop_recommendation_model.pkl')
model = joblib.load(model_path)
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract input features
        N = int(data["N"])
        P = int(data["P"])
        K = int(data["K"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])

        # Make prediction
        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])

        # Convert NumPy object to Python type using .item()
        predicted_crop = label_encoder.inverse_transform([prediction[0]])[0]

        # Return result as JSON
        return jsonify({"prediction": predicted_crop})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
