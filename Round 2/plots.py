import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

script_dir = os.path.dirname(__file__)


csv_file_path = os.path.join(script_dir, 'data', 'sample.csv')

df = pd.read_csv(csv_file_path, sep=';')

df = df.set_index('timestamp')

df['spread'] = df['ask_price_1'] - df['bid_price_1']

amethysts_df = df[df['product'] == 'AMETHYSTS']
starfruit_df = df[df['product'] == 'STARFRUIT']


def plot_prices(df):

   
    df['81'] = df['mid_price'].rolling(window=81).mean()
    df['50'] = df['mid_price'].rolling(window=50).mean()
    df['20'] = df['mid_price'].rolling(window=20).mean()
    df['9'] = df['mid_price'].rolling(window=9).mean()

    plt.figure(figsize=(10, 6))  
    plt.plot(df['bid_price_1'], label='Bid', color='yellow')  
    plt.plot(df['ask_price_1'], label='Ask', color='blue')  
    #plt.plot(df['bid_price_2'], label='Bid', color='orange')  
    #plt.plot(df['ask_price_2'], label='Ask', color='orange')  
    #plt.plot(df['bid_price_3'], label='Bid', color='orange')  
    #plt.plot(df['ask_price_3'], label='Ask', color='red')  
    plt.plot(df['mid_price'], label='Mid Price', color='green') 
    plt.plot(df['81'], label=f'81', color='blue', linestyle='--')
    plt.plot(df['50'], label=f'50', color='red', linestyle='--') 
    plt.plot(df['20'], label=f'20', color='purple', linestyle='--') 
    plt.plot(df['9'], label=f'9', color='black', linestyle='--') 

    plt.title('Mid Price Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()  
    plt.xticks(rotation=45)  

    plt.show()

    print('Max ask', df['ask_price_1'].max())
    print('Min bid', df['bid_price_1'].min())
    print('Avg spread:', df['spread'].mean())

#plot_prices(amethysts_df)
plot_prices(starfruit_df)