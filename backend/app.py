from flask import Flask, request, jsonify
import joblib
import torch
import numpy as np
from model import LSTM1
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model once
MODEL_PATH = 'backend/lstm_aqi_model.pth'
SCALER_PATH = 'backend/scaler.pkl'

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the trained LSTM model
def load_model():
    model = LSTM1(num_classes=1, input_size=13, hidden_size=2, num_layers=1, seq_length=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Use CPU mode if needed
    model.eval()
    print("✅ Model loaded successfully!")
    return model

# Load scaler
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

scaler = joblib.load(SCALER_PATH)  # Load the scaler
model = load_model()  # Load model once

# Define the correct order of input features
FEATURE_ORDER = ["PM25", "PM10", "NO2", "NOx", "CO", "SO2", "O3", 
                 "temp", "max_temp", "min_temp", "humid", "visible", "wind"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("📥 Received data:", data)  # Debugging output

        # Validate input
        if not all(key in data for key in FEATURE_ORDER):
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Extract and reorder input features
        input_data = np.array([data[key] for key in FEATURE_ORDER]).reshape(1, -1)

        # Scale input data
        scaled_input = scaler.transform(input_data).reshape(1, 1, -1)  # Reshape for LSTM

        # Predict AQI
        with torch.no_grad():
            prediction = model(torch.Tensor(scaled_input)).numpy()

        predicted_aqi = float(prediction[0][0])  # Convert to Python float

        return jsonify({'predicted_AQI': predicted_aqi})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({'error': str(e)}), 400  # Return the error message if something goes wrong

if __name__ == '__main__':
    app.run(debug=True)
