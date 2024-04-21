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

csv_file_path_0 = os.path.join(script_dir, 'data', 'prices_round_1_day_0.csv')
csv_file_path_1 = os.path.join(script_dir, 'data', 'prices_round_1_day_-1.csv')
csv_file_path_2 = os.path.join(script_dir, 'data', 'prices_round_1_day_-2.csv')
csv_file_path_3 = os.path.join(script_dir, 'data', 'output_round1.csv')
csv_file_path_4 = os.path.join(script_dir, 'data', 'output_round2.csv')
csv_file_path_5 = os.path.join(script_dir, 'data', 'output_round3.csv')



df_0 = pd.read_csv(csv_file_path_0, sep=';')
df_1 = pd.read_csv(csv_file_path_1, sep=';')
df_2 = pd.read_csv(csv_file_path_2, sep=';')
df_3 = pd.read_csv(csv_file_path_3, sep=';')
df_4 = pd.read_csv(csv_file_path_4, sep=';')
df_5 = pd.read_csv(csv_file_path_5, sep=';')


for i, df in enumerate([df_2, df_1, df_0, df_3, df_4, df_5]):
    df['timestamp'] = df['timestamp'] / 100 + i * 10000


df = pd.concat([df_2, df_1])
df = pd.concat([df, df_0])
df = pd.concat([df, df_3])
df = pd.concat([df, df_4])
df = pd.concat([df, df_5])

#df = df[df['timestamp'] < 20000]
df = df.set_index('timestamp')


amethysts_df = df[df['product'] == 'AMETHYSTS']
starfruit_df = df[df['product'] == 'STARFRUIT']


amethysts_df = amethysts_df['mid_price']
starfruit_df = (starfruit_df['mid_price'])
print(starfruit_df)

"""
model = auto_arima(starfruit_df, 
                   start_p=1, start_q=1,
                   test='adf',       # Use adftest to find optimal 'd'
                   max_p=3, max_q=3, # Maximum p and q
                   m=1,              # Frequency of the series
                   d=None,           # Let model determine 'd'
                   seasonal=False,   # No Seasonality
                   start_P=0, 
                   D=0, 
                   trace=True,
                   error_action='ignore',  
                   suppress_warnings=True, 
                   stepwise=True)

# Summary of the model
print(model.summary())
"""

# Assuming 'series' is your Pandas Series with the time series data
result = adfuller(starfruit_df)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


model = ARIMA(starfruit_df, order=(6,1,3))
model_fit = model.fit()
print(model_fit.summary())

# Get the in-sample residuals
residuals = model_fit.resid
initial_errors = residuals[-2:] 
print(initial_errors)