"""Consider the problem at hand, the data characteristics, and the available expertise when choosing between these methods. It may also be beneficial to experiment with both approaches and compare their performance on your specific problem.
Can you write me out one in Python with an SPY.csv as input
Certainly! Below, I'll provide you with a Python script that uses an LSTM model paired with ARIMA for time series forecasting. We'll use the SPY.csv file as input, assuming it contains historical stock price data for SPY (S&P 500 E-mini futures).
First, make sure you have the required libraries installed:"""
#pip install pandas numpy matplotlib scikit-learn statsmodels tensorflow


#Python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load the data
df = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)
df = df['Close'].dropna()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))

# Create training and testing datasets
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(scaled_data, time_step)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Train LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict using LSTM
lstm_pred = model.predict(X_train)
lstm_pred = scaler.inverse_transform(lstm_pred)

# Fit ARIMA model
arima_model = ARIMA(df, order=(5,1,2))
arima_result = arima_model.fit()
arima_pred = arima_result.forecast(steps=len(y_train))[0]
arima_pred = scaler.inverse_transform(arima_pred.reshape(-1, 1))

# Combine LSTM and ARIMA predictions
combined_pred = (lstm_pred + arima_pred) / 2

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_train, combined_pred))
print(f'RMSE: {rmse}')

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df.index[-len(y_train):], y_train, label='Actual')
plt.plot(df.index[-len(y_train):], combined_pred, label='Predicted')
plt.title('SPY Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#ADD Straddle long at assumed Strike Price

import numpy as np
import pandas as pd
from datetime import datetime


# Load historical option data (this should be a DataFrame with columns: 'Strike', 'Expiry', 'CallPrice', 'PutPrice')
# For example, you might load this from a CSV file:
# df_options = pd.read_csv('historical_option_prices.csv')


# Define the straddle long strategy
def straddle_long_strategy(df_options, current_price, strike_price, expiry_date):
    # Filter options with the same expiry and strike price
    options_at_price = df_options[(df_options['Expiry'] == expiry_date) & (df_options['Strike'] == strike_price)]
    
    # Calculate the intrinsic value of the call and put options
    call_intrinsic_value = np.maximum(current_price - strike_price, 0)
    put_intrinsic_value = np.maximum(strike_price - current_price, 0)
    
    # Calculate the premium paid for the call and put options
    call_premium = options_at_price['CallPrice'].values[0]
    put_premium = options_at_price['PutPrice'].values[0]
    
    # Calculate the net cost of the straddle
    straddle_cost = call_premium + put_premium
    
    # Check if the strategy is profitable
    if call_intrinsic_value + put_intrinsic_value > straddle_cost:
        return True, straddle_cost, call_intrinsic_value, put_intrinsic_value
    else:
        return False, straddle_cost, call_intrinsic_value, put_intrinsic_value


# Example usage
current_stock_price = 100  # Current stock price
strike_price = 100  # Strike price for the options
expiry_date = '2023-12-17'  # Expiry date for the options


# Assuming you have a DataFrame with historical option prices
# df_options = pd.read_csv('historical_option_prices.csv')


# Check if the straddle long strategy is profitable
is_profitable, cost, call_value, put_value = straddle_long_strategy(df_options, current_stock_price, strike_price, expiry_date)


if is_profitable:
    print(f"Straddle long strategy is profitable. Cost: {cost}, Call intrinsic value: {call_value}, Put intrinsic value: {put_value}")
    # Here you would add the code to execute the trade
else:
    print("Straddle long strategy is not profitable.")


# Note: This script does not include the actual trade execution, risk management, or real-time data fetching.

