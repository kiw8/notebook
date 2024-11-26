# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Seed fixing for reproducibility
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

# Load dataset
dataraw = pd.read_csv('data/BTC-USD.csv', index_col='Date', parse_dates=['Date'])
dataset = pd.DataFrame(dataraw['Close'])

# Normalize the data
scaler = MinMaxScaler()
dataset_norm = dataset.copy()
dataset_norm['Close'] = scaler.fit_transform(dataset[['Close']])

# Data splitting: 70% train, 10% validation, 20% test
totaldata = dataset_norm.values
totaldatatrain = int(len(totaldata) * 0.7)
totaldataval = int(len(totaldata) * 0.1)

training_set = totaldata[:totaldatatrain]
val_set = totaldata[totaldatatrain:totaldatatrain + totaldataval]
test_set = totaldata[totaldatatrain + totaldataval:]

# Sliding window creation
def create_sliding_windows(data, len_data, lag):
    x, y = [], []
    for i in range(lag, len_data):
        x.append(data[i - lag:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Hyperparameters
lag = 2

# Create sliding windows for train, validation, and test sets
x_train, y_train = create_sliding_windows(training_set, len(training_set), lag)
x_val, y_val = create_sliding_windows(val_set, len(val_set), lag)
x_test, y_test = create_sliding_windows(test_set, len(test_set), lag)

# Convert datasets to PyTorch tensors
x_train, y_train = (torch.tensor(x_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32))
x_val, y_val = (torch.tensor(x_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32))
x_test, y_test = (torch.tensor(x_test, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.float32))

# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out

# Initialize model, criterion, and optimizer
input_size = 1
hidden_size = 64
output_size = 1
model = GRUModel(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train.unsqueeze(-1))
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        val_outputs = model(x_val.unsqueeze(-1))
        val_loss = criterion(val_outputs.squeeze(), y_val)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Testing and evaluation
model.eval()
y_pred = model(x_test.unsqueeze(-1)).detach().numpy()
y_pred_inver_norm = scaler.inverse_transform(y_pred)

# Original test set values for comparison
dataset_test = dataset.values[totaldatatrain + totaldataval + lag:][:len(y_pred)]

# RMSE and MAPE calculation
def rmse(dataset, datapred):
    return np.sqrt(np.mean((datapred - dataset) ** 2))

def mape(dataset, datapred):
    return np.mean(np.abs((dataset - datapred) / dataset)) * 100

print('RMSE:', rmse(dataset_test, y_pred_inver_norm))
print('MAPE:', mape(dataset_test, y_pred_inver_norm))

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(dataset_test, label="Actual Test Data", color='red')
plt.plot(y_pred_inver_norm, label="Predictions", color='blue')
plt.title('BTC-USD Price Prediction')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()

