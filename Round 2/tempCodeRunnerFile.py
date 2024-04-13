import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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


plot_prices(orchids_df)