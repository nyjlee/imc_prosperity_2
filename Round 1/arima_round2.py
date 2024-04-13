######## THIS ARIMA WILL TRY TO PREDICT PRICES INSTEAD OF MID PRICES ############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import os

script_dir = os.path.dirname(__file__)

csv_file_path_0 = os.path.join(script_dir, 'data', 'trades_round_1_day_0_nn.csv')
csv_file_path_1 = os.path.join(script_dir, 'data', 'trades_round_1_day_-1_nn.csv')
csv_file_path_2 = os.path.join(script_dir, 'data', 'trades_round_1_day_-2_nn.csv')


df_0 = pd.read_csv(csv_file_path_0, sep=';')
df_1 = pd.read_csv(csv_file_path_1, sep=';')
df_2 = pd.read_csv(csv_file_path_2, sep=';')


for i, df in enumerate([df_2, df_1, df_0]):
    df['timestamp'] = df['timestamp'] / 100 + i * 10000


df = pd.concat([df_2, df_1])
df = pd.concat([df, df_0])

df = df.set_index('timestamp')

starfruit_df = df[df['symbol'] == 'STARFRUIT']

starfruit_df = (starfruit_df['price'])
print(starfruit_df)

# Assuming 'series' is your Pandas Series with the time series data
result = adfuller(starfruit_df)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


model = ARIMA(starfruit_df, order=(10,1,3))
model_fit = model.fit()
print(model_fit.summary())

# Get the in-sample residuals
residuals = model_fit.resid
initial_errors = residuals[-2:] 
print(initial_errors)