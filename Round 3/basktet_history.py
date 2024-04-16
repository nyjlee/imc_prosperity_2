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
import statsmodels.api as sm

import os

script_dir = os.path.dirname(__file__)


csv_file_path_0 = os.path.join(script_dir, 'data', 'prices_round_3_day_0.csv')
csv_file_path_1 = os.path.join(script_dir, 'data', 'prices_round_3_day_1.csv')
csv_file_path_2 = os.path.join(script_dir, 'data', 'prices_round_3_day_2.csv')


df_0 = pd.read_csv(csv_file_path_0, sep=';')
df_1 = pd.read_csv(csv_file_path_1, sep=';')
df_2 = pd.read_csv(csv_file_path_2, sep=';')


for i, df in enumerate([df_0, df_1, df_2]):
    df['timestamp'] = df['timestamp'] / 100 + i * 10000


df = pd.concat([df_0, df_1])
df = pd.concat([df, df_2])
df = df.set_index('timestamp')
print(df)

def basket_spread(products_df):


    chocolates = products_df[products_df['product'] == 'CHOCOLATE']['mid_price']
    strawberries = products_df[products_df['product'] == 'STRAWBERRIES']['mid_price']
    roses = products_df[products_df['product'] == 'ROSES']['mid_price']
    gift_basket = products_df[products_df['product'] == 'GIFT_BASKET']['mid_price']

    spread = gift_basket - (4*chocolates + 6*strawberries + roses)

    print(spread)
    print('SPREAD AVG:', spread.mean())
    print('SPREAD STD:', spread.std())

basket_spread(df.iloc[20000:30000])