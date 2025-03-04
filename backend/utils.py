import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import LSTM1  # Import the model architecture
import joblib

# Load the trained model
def load_model(model_path):
    # Initialize the model with the correct parameters
    model = LSTM1(num_classes=1, input_size=13, hidden_size=2, num_layers=1, seq_length=1)
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    # Confirm model has been loaded properly
    print("Model loaded successfully!")
    
    return model

# Preprocess input data for model prediction
def preprocess_input(input_data):
    # Load the scaler used during training
    sc = joblib.load('backend\scaler.pkl')  # Loading the saved scaler
    
    # Scale the incoming data
    scaled_input = sc.transform(input_data)
    
    # Reshape the data for LSTM
    scaled_input = scaled_input.reshape(1, 1, -1)  # Reshaping for LSTM input
    return scaled_input