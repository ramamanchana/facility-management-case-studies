import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample Data
data = {
    'timestamp': pd.date_range(start='1/1/2022', periods=100, freq='H'),
    'energy_usage': np.random.rand(100) * 100
}

# Create DataFrame
df = pd.DataFrame(data)

# Prepare data for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# Normalize the dataset
df['energy_usage'] = df['energy_usage'] / df['energy_usage'].max()

# Create dataset
look_back = 10
X, Y = create_dataset(df['energy_usage'].values.reshape(-1, 1), look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], look_back, 1))

# LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train Model
model.fit(X, Y, epochs=10, batch_size=1, verbose=2)

# Make predictions
train_predict = model.predict(X)

# Shift train predictions for plotting
train_predict_plot = np.empty_like(df['energy_usage'])
train_predict_plot[:] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back] = train_predict.reshape(-1)

# Plot baseline and predictions
plt.plot(df['timestamp'], df['energy_usage'], label='Actual')
plt.plot(df['timestamp'], train_predict_plot, label='Predicted', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Energy Usage')
plt.title('Real-time Energy Consumption Monitoring')
plt.legend()
plt.show()
