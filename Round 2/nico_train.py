import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

import os

script_dir = os.path.dirname(__file__)


csv_file_path_0 = os.path.join(script_dir, 'data', 'prices_round_2_day_-1.csv')
csv_file_path_1 = os.path.join(script_dir, 'data', 'prices_round_2_day_0.csv')
csv_file_path_2 = os.path.join(script_dir, 'data', 'prices_round_2_day_1.csv')


df_0 = pd.read_csv(csv_file_path_0, sep=';')
df_1 = pd.read_csv(csv_file_path_1, sep=';')
df_2 = pd.read_csv(csv_file_path_2, sep=';')


for i, df in enumerate([df_0, df_1, df_2]):
    df['timestamp'] = df['timestamp'] / 100 + i * 10000


df = pd.concat([df_0, df_1])
orchids_df = pd.concat([df, df_2])
orchids_df = orchids_df.set_index('timestamp')

def plot_prices(df):

   
    df['81'] = df['ORCHIDS'].rolling(window=81).mean()
    df['50'] = df['ORCHIDS'].rolling(window=34).mean()

    plt.figure(figsize=(10, 6))  
    plt.plot(df['ORCHIDS'], label='Price', color='blue')  
    
    plt.plot(df['81'], label=f'81', color='blue', linestyle='--')
    plt.plot(df['50'], label=f'50', color='red', linestyle='--') 

    plt.title('Orchids Evolution')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()  
    plt.xticks(rotation=45)  

    plt.show()

    print('Max', df['ORCHIDS'].max())
    print('Min', df['ORCHIDS'].min())


# plot_prices(orchids_df)

"""
orchids_df = (orchids_df['ORCHIDS'])

result = adfuller(orchids_df)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


model = ARIMA(orchids_df, order=(6,1,3))
model_fit = model.fit()
print(model_fit.summary())

# Get the in-sample residuals
residuals = model_fit.resid
initial_errors = residuals[-2:] 
print(initial_errors)
"""


orchids_df = orchids_df.iloc[::100, :]
print(orchids_df.head())

orchids_df = orchids_df.reset_index(drop=True)
print(orchids_df.head())


def check_stationarity(series, name):
    result = adfuller(series, autolag='AIC')
    print(f'Results for {name}:')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] > 0.05:
        print(f"{name} is not stationary")
    else:
        print(f"{name} is stationary")

# Check each variable
for column in ['ORCHIDS', 'SUNLIGHT', 'HUMIDITY']:
    check_stationarity(orchids_df[column], column)


plot_prices(orchids_df)