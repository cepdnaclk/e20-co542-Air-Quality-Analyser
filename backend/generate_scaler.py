# generate_scaler.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset (assuming 'data.csv' or similar)
data = pd.read_csv('data/AQI_prediction_dataset.csv')

# Select the relevant features (excluding the target column like 'AQI')
X_train = data[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'SO2', 'O3', 'temp', 'max_temp', 'min_temp', 'humid', 'visible', 'wind']]

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler to a .pkl file
joblib.dump(scaler, 'scaler.pkl')
