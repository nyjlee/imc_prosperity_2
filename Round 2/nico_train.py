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
orchids_df['SL_RATIO'] = orchids_df['SUNLIGHT'] / orchids_df['HUMIDITY']
print(orchids_df['SL_RATIO'].mean())

for df in [df_0, df_1, df_2, orchids_df]:
    df['SL_RATIO'] = df['SUNLIGHT'] / (df['HUMIDITY'])
    df['XM_RATIO'] = df['EXPORT_TARIFF'] / abs(df['IMPORT_TARIFF'])

    df['HUM'] = (df['HUMIDITY'].shift(-1) - df['HUMIDITY']) 
    df['SUN'] = (df['SUNLIGHT'].shift(-1) - df['SUNLIGHT']) 
    df['SLRC'] = (df['SL_RATIO'].shift(5) / df['SL_RATIO']) -1
    # Calculate the first derivative of the ratio
    df['First_Derivative'] = df['SLRC'].diff()

    df['XM_Derivative'] = df['XM_RATIO'].diff()

    # Calculate the second derivative of the ratio
    df['Second_Derivative'] = df['First_Derivative'].diff()

    df['Third_Derivative'] = df['Second_Derivative'].diff()

    # Assuming the intervals are uniform and h=1 for simplicity
    # First Derivative (Central Difference)
    df['First_Derivative_Central'] = (df['SL_RATIO'].shift(-1) - df['SL_RATIO'].shift(1)) / 2

    # Second Derivative (Central Difference)
    df['Second_Derivative_Central'] = (df['SL_RATIO'].shift(-1) - 2 * df['SL_RATIO'] + df['SL_RATIO'].shift(1))

    df['EM_DERIVATIVE'] = df['Second_Derivative'].ewm(span=8, adjust=False).mean()




def plot_prices(df):

    plt.figure(figsize=(10, 6))  
    plt.plot(df['ORCHIDS'], label='Price', color='blue')  
    
    plt.title('Price Evolution')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()  
    plt.xticks(rotation=45)  

    plt.show()

    print('Max', df['ORCHIDS'].max())
    print('Min', df['ORCHIDS'].min())


#plot_prices(orchids_df)

def plot_orchids(data):
    # Create a dual-axis time series plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting orchid prices on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Time (timestamp)')
    ax1.set_ylabel('Orchid Prices ($)', color=color)
    ax1.plot(data.index, data['ORCHIDS'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the sunlight to humidity ratio
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Sunlight to Humidity Ratio', color=color)
    ax2.plot(data.index, data['First_Derivative'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a title and grid
    plt.title('Orchid Prices and Sunlight/Humidity Ratio Over Time')
    fig.tight_layout()  # adjust subplots to fit into figure area.
    plt.grid(True)
    plt.show()

    

def plot_orchids2(data):
    # Calculate the ratio and its derivatives
    # Normalize derivatives for better visualization
    data['First_Derivative_Norm'] = (data['First_Derivative'] - data['First_Derivative'].min()) / (data['First_Derivative'].max() - data['First_Derivative'].min())
    data['Second_Derivative_Norm'] = (data['Second_Derivative'] - data['Second_Derivative'].min()) / (data['Second_Derivative'].max() - data['Second_Derivative'].min())

    # Create a dual-axis time series plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting orchid prices on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Time (timestamp)')
    ax1.set_ylabel('Orchid Prices ($)', color=color)
    ax1.plot(data['timestamp'], data['ORCHIDS'], color=color, label='Orchid Prices')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the sunlight to humidity ratio and its derivatives
    ax2 = ax1.twinx()
    color_ratio = 'tab:blue'
    color_first_derivative = 'tab:orange'
    color_second_derivative = 'tab:green'
    ax2.set_ylabel('Metrics', color=color_ratio)
    ax2.plot(data['timestamp'], data['SL_RATIO'], color=color_ratio, label='Sunlight/Humidity Ratio')
    ax2.plot(data['timestamp'], data['First_Derivative_Norm'], color=color_first_derivative, label='First Derivative (Normalized)')
    ax2.plot(data['timestamp'], data['Second_Derivative_Norm'], color=color_second_derivative, label='Second Derivative (Normalized)')
    ax2.tick_params(axis='y', labelcolor=color_ratio)

    # Add legends and title
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Orchid Prices, Sunlight/Humidity Ratio, and Derivatives Over Time')
    fig.tight_layout()  # adjust subplots to fit into figure area.
    plt.grid(True)
    plt.show()

#plot_orchids(df_0)
#plot_orchids(df_1)
#plot_orchids(df_2)


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


#orchids_df = orchids_df.iloc[::100, :]
#print(orchids_df.head())

#orchids_df = orchids_df.reset_index(drop=True)
#print(orchids_df.head())

orchids_df['OPT SUNLIGHT'] = (orchids_df['SUNLIGHT'] > 2500).astype(int)
orchids_df['OPT HUMIDITY'] = orchids_df['HUMIDITY'].between(60, 80).astype(int)

orchids_df = orchids_df[['HUMIDITY', 'SUNLIGHT', 'ORCHIDS', 'SUN', 'HUM']]

#model = VAR(df)
#results = model.fit(maxlags=5, ic='aic')
#print(results.summary())


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


#plot_prices(orchids_df)


