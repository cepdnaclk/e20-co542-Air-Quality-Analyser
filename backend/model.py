# -*- coding: utf-8 -*-
"""LSTM1.ipynb - AQI Prediction using LSTM"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.autograd import Variable
import joblib

# Load dataset
data = pd.read_csv('data/AQI_prediction_dataset.csv')
print("Data overview:")
print(data.info())

# Plot the first few rows of the dataset
print(data.head())

# Line plot of NOx values
sns.lineplot(x=data.index, y=data['NOx'])
plt.xlabel('Index')
plt.title('NOx Levels Over Time')
plt.show()

# Plot all features
data.plot(subplots=True, layout=(5, 3), figsize=(22, 18))
plt.show()

# Drop the 'Date' column since it is not needed for modeling
data = data.drop(columns=['Date'])

# Separate features (X) and target (y)
target_column = 'AQI'
X = data.drop(columns=[target_column])
y = data[target_column]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the features and target
sc = StandardScaler()
sc_target = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)

# Save the scaler for later use in prediction
joblib.dump(sc, 'scaler.pkl')

y_train_scaled = sc_target.fit_transform(y_train.values.reshape(-1, 1))
X_test_scaled = sc.transform(X_test)
y_test_scaled = sc_target.transform(y_test.values.reshape(-1, 1))

# Convert data to PyTorch tensors
X_train_tensors = Variable(torch.Tensor(X_train_scaled))
y_train_tensors = Variable(torch.Tensor(y_train_scaled))
X_test_tensors = Variable(torch.Tensor(X_test_scaled))
y_test_tensors = Variable(torch.Tensor(y_test_scaled))

# Reshape tensors to 3D (batch_size, sequence_length, input_size)
X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

# Define LSTM Model
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)  # Fully connected layer 1
        self.fc = nn.Linear(128, num_classes)    # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size))  # Hidden state
        c_0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size))  # Cell state

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # LSTM
        hn = hn.view(-1, self.hidden_size)           # Reshape hidden state
        out = self.relu(hn)
        out = self.fc_1(out)  # First fully connected layer
        out = self.relu(out)
        out = self.fc(out)    # Output layer
        return out

# Hyperparameters
num_epochs = 1000
learning_rate = 0.001
input_size = 13
hidden_size = 2
num_layers = 1
num_classes = 1

# Initialize model, loss function, and optimizer
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(num_epochs):
    lstm1.train()  # Set model to training mode
    outputs = lstm1(X_train_tensors_final)  # Forward pass
    optimizer.zero_grad()  # Clear gradients

    # Compute loss
    loss = criterion(outputs, y_train_tensors)
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model weights

    # Store and print loss at every 100 epochs
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.5f}")

# After the training loop
torch.save(lstm1.state_dict(), 'lstm_aqi_model.pth')
print("Model saved successfully!")

# Plot training loss
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Evaluate the model
lstm1.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient calculation during evaluation
    predictions = lstm1(X_test_tensors_final).detach().numpy()
    predictions = sc_target.inverse_transform(predictions)  # Inverse transform predictions to original scale

# Compare predictions with actual values
y_test_actual = sc_target.inverse_transform(y_test_scaled)

# Visualize predictions vs actual AQI
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='True AQI')
plt.plot(predictions, label='Predicted AQI')
plt.title('True vs Predicted AQI')
plt.legend()
plt.show()

# Calculate R-squared score
r2 = r2_score(y_test_actual, predictions)
print(f'R-squared score: {r2:.4f}')

# Heatmap of feature correlations
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Custom data testing
custom_data = pd.DataFrame([{
    "PM2.5": 47.56, "PM10": 116.59, "NO2": 36.31, "NOx": 32.77, "CO": 0.85, "SO2": 4.07,
    "O3": 40.04, "temp": 27.9, "max_temp": 33.6, "min_temp": 22.1, "humid": 73, "visible": 6.3, "wind": 10.7
}])

custom_data_scaled = sc.transform(custom_data)
custom_data_tensors = Variable(torch.Tensor(custom_data_scaled)).reshape(1, 1, -1)

lstm1.eval()
with torch.no_grad():
    predicted_aqi = lstm1(custom_data_tensors).detach().numpy()
    predicted_aqi = sc_target.inverse_transform(predicted_aqi)
    print(f'Predicted AQI for custom data: {predicted_aqi[0][0]:.2f}')
