import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

data = pd.read_csv('your_data.csv', parse_dates=True, index_col='Date')

plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

stepwise_fit = auto_arima(data, start_p=1, start_q=1, max_p=3, max_q=3, m=12, seasonal=True, trace=True)
p, d, q = stepwise_fit.order


train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit()

forecast = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

rmse = sqrt(mean_squared_error(test_data, forecast))
mae = mean_absolute_error(test_data, forecast)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')

plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Test Data')
plt.plot(forecast, label='Forecast')
plt.title('Actual vs. Forecasted Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
