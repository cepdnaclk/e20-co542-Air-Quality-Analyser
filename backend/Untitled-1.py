# %%
!pip install torch

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# %%
# -------------------------------
# Data Loading & Preprocessing
# -------------------------------
data = pd.read_csv('AQI_prediction_dataset.csv')
data = data.drop(columns=['Date'])
target_column = 'AQI'
X = data.drop(columns=[target_column])
y = data[target_column]


# %%
# Train-test split (preserving time order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# %%
# Standardize features and target
sc = StandardScaler()
sc_target = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
y_train_scaled = sc_target.fit_transform(y_train.values.reshape(-1, 1))
X_test_scaled = sc.transform(X_test)
y_test_scaled = sc_target.transform(y_test.values.reshape(-1, 1))


# %%
# Convert to tensors and reshape to (batch_size, seq_length, num_features)
X_train_tensor = torch.Tensor(X_train_scaled).unsqueeze(1)  # seq_length = 1
y_train_tensor = torch.Tensor(y_train_scaled)
X_test_tensor  = torch.Tensor(X_test_scaled).unsqueeze(1)
y_test_tensor  = torch.Tensor(y_test_scaled)


# %%
# -------------------------------
# Model 1: Single LSTM layer, one FC layer + output
# -------------------------------

# %%
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        # Use last hidden state (reshape if needed)
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

# %%
# -------------------------------
# Model 2: Two LSTM layers, one FC layer + output; ReLU after LSTM & FC
# -------------------------------

# %%
class LSTM2(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, seq_length):
        super(LSTM2, self).__init__()
        # Two LSTM layers (num_layers=2)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        # One FC layer and an output layer (two dense layers)
        self.fc = nn.Linear(hidden_size, 128)
        self.out = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.randn(2, x.size(0), self.hidden_size)
        c0 = torch.randn(2, x.size(0), self.hidden_size)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        # Use the last hidden state from the final LSTM layer
        out = self.relu(hn[-1])
        out = self.fc(out)
        out = self.relu(out)
        out = self.out(out)
        return out

# %%
# -------------------------------
# Model 3: Two LSTM layers, two FC layers and an output layer; ReLU after each layer
# -------------------------------

# %%
class LSTM3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, fc_hidden_size, seq_length):
        super(LSTM3, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.out = nn.Linear(fc_hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.randn(2, x.size(0), self.hidden_size)
        c0 = torch.randn(2, x.size(0), self.hidden_size)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.relu(hn[-1])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.out(out)
        return out

# %%
# -------------------------------
# Model 4: Model 3 with Dropout (dropout rate = 0.5)
# -------------------------------

# %%
class LSTM4(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, fc_hidden_size, seq_length, dropout_rate=0.5):
        super(LSTM4, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.out = nn.Linear(fc_hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h0 = torch.randn(2, x.size(0), self.hidden_size)
        c0 = torch.randn(2, x.size(0), self.hidden_size)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.relu(hn[-1])
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.out(out)
        return out


# %%
# -------------------------------
# Training Function
# -------------------------------
def train_model(model, X_train, y_train, num_epochs=200, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.5f}")
    return losses

# %%
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    # Inverse-transform predictions and true values
    predictions_inv = sc_target.inverse_transform(predictions)
    y_actual = sc_target.inverse_transform(y_test.numpy())
    r2 = r2_score(y_actual, predictions_inv)
    mae = mean_absolute_error(y_actual, predictions_inv)
    rmse = np.sqrt(mean_squared_error(y_actual, predictions_inv))
    mape = np.mean(np.abs((y_actual - predictions_inv) / y_actual)) * 100
    return r2, mae, rmse, mape, predictions_inv, y_actual

# %%
# -------------------------------
# Hyperparameters & Model Instantiation
# -------------------------------
num_classes = 1
input_size = X_train_tensor.shape[2]  # should be 13
seq_length = 1

# For Models 1 & 2, we use hidden_size=2
hidden_size_m1_m2 = 2

# For Models 3 & 4, we add an extra FC layer so we use a larger hidden size in FC layers
fc_hidden_size = 128
hidden_size_m3_m4 = 2  # you can also experiment with larger hidden sizes here if desired


# %%
# Instantiate each model
model1 = LSTM1(num_classes, input_size, hidden_size_m1_m2, num_layers=1, seq_length=seq_length)
model2 = LSTM2(num_classes, input_size, hidden_size_m1_m2, seq_length=seq_length)
model3 = LSTM3(num_classes, input_size, hidden_size_m3_m4, fc_hidden_size, seq_length=seq_length)
model4 = LSTM4(num_classes, input_size, hidden_size_m3_m4, fc_hidden_size, seq_length=seq_length, dropout_rate=0.5)

models = {
    "Model 1 (1 LSTM, 1 FC + output)": model1,
    "Model 2 (2 LSTM, 1 FC + output)": model2,
    "Model 3 (2 LSTM, 2 FC + output)": model3,
    "Model 4 (Model 3 + Dropout 0.5)": model4
}

# %%
# -------------------------------
# Train and Evaluate All Models
# -------------------------------
num_epochs = 200
lr = 0.001
results = {}
losses_dict = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")

    # Train the model and store the loss curve
    losses = train_model(model, X_train_tensor, y_train_tensor, num_epochs=200, lr=lr)
    losses_dict[name] = losses

    # Evaluate the model and unpack all evaluation metrics
    r2, mae, rmse, mape, preds, y_actual = evaluate_model(model, X_test_tensor, y_test_tensor)

    # Store all metrics in a dictionary for this model
    results[name] = {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}

    print(f"{name} Metrics:")
    print(f"R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

# %%
# -------------------------------
# Compare Model Performance (All Metrics)
# -------------------------------
print("\n--- Model Comparison ---")
for name, metrics in results.items():
    print(f"{name}: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.4f}, "
          f"RMSE = {metrics['rmse']:.4f}, MAPE = {metrics['mape']:.4f}%")

# %%
# -------------------------------
# Visualize predictions for each model
# Plot Actual vs Predicted AQI for Each Model
# -------------------------------
plt.figure(figsize=(15, 10))

# Loop over the models dictionary
for i, (name, model) in enumerate(models.items()):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        preds = model(X_test_tensor).numpy()
    # Inverse-transform the predictions and true targets
    preds_inv = sc_target.inverse_transform(preds)
    y_actual = sc_target.inverse_transform(y_test_tensor.numpy())

    plt.subplot(2, 2, i + 1)
    plt.plot(y_actual, label='True AQI', color='blue', linestyle='--')
    plt.plot(preds_inv, label='Predicted AQI', color='red')
    plt.title(name)
    plt.xlabel('Test Sample Index')
    plt.ylabel('AQI')
    plt.legend()

plt.tight_layout()
plt.show()

# %%
# -------------------------------
# Plot Loss Variation for Each Model
# -------------------------------
plt.figure(figsize=(15, 10))
for i, (name, loss_values) in enumerate(losses_dict.items()):
    plt.subplot(2, 2, i + 1)
    plt.plot(loss_values, label='Loss', color='purple')
    plt.title(f"{name} Loss Variation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
plt.tight_layout()
plt.show()


# %%
! pip install mlflow

# %%
# -------------------------------
# MLFlow Integration: Train & Log Metrics for Each Model
# -------------------------------
import mlflow
import mlflow.pytorch

mlflow.set_experiment("AQI_Prediction_Milestone2_MLOps")

# Hyperparameters
num_epochs = 200
lr = 0.001

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Log basic hyperparameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", lr)

        # Depending on the model, log hidden size and layers info
        if "Model 1" in model_name or "Model 2" in model_name:
            mlflow.log_param("hidden_size", hidden_size_m1_m2)
            mlflow.log_param("num_lstm_layers", 1 if "Model 1" in model_name else 2)
        else:
            mlflow.log_param("hidden_size", hidden_size_m3_m4)
            mlflow.log_param("fc_hidden_size", fc_hidden_size)
            mlflow.log_param("num_lstm_layers", 2)
            if "Dropout" in model_name:
                mlflow.log_param("dropout_rate", 0.5)

        # Train the model
        print(f"\nTraining {model_name} ...")
        losses = train_model(model, X_train_tensor, y_train_tensor, num_epochs=num_epochs, lr=lr)
        mlflow.log_metric("final_loss", losses[-1])

        # Save and log the loss curve plot
        plt.figure()
        plt.plot(losses, label="Training Loss", color="purple")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{model_name} Training Loss")
        plt.legend()
        plt.savefig("loss_curve.png")
        mlflow.log_artifact("loss_curve.png")
        plt.close()

        # Evaluate the model and log evaluation metrics
        r2, mae, rmse, mape, preds_inv, y_actual = evaluate_model(model, X_test_tensor, y_test_tensor)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)

        print(f"{model_name} Metrics:")
        print(f"R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

        # Plot Actual vs. Predicted AQI and log the plot
        plt.figure(figsize=(10,6))
        plt.plot(y_actual, label="Actual AQI", linestyle="--", color="blue")
        plt.plot(preds_inv, label="Predicted AQI", color="red")
        plt.title(f"{model_name}: Actual vs Predicted AQI")
        plt.xlabel("Test Sample Index")
        plt.ylabel("AQI")
        plt.legend()
        plt.savefig("predictions_plot.png")
        mlflow.log_artifact("predictions_plot.png")
        plt.close()

        # Log the trained model
        mlflow.pytorch.log_model(model, "model")

# %%
!pip install pyngrok

# %%
from pyngrok import ngrok
!ngrok authtoken 2tQS3Osnmi3zFJ8YOVe0pK09iFN_6dyBEL1cGVLEiMPmYyPSr

# %%
# Launch MLflow UI in the background on port 5000 using nohup
!nohup mlflow ui --port 5000 &

# %%
# Allow some time for MLflow UI to start
import time
time.sleep(10)

# %%
# Set up the ngrok tunnel to port 5000
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print("MLflow UI is accessible at:", public_url)


