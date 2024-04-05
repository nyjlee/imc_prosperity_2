import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)

csv_file_path = os.path.join(script_dir, 'data', 'tutorial_round.csv')

df = pd.read_csv(csv_file_path, sep=';')

df = df.set_index('timestamp')

amethysts_df = df[df['product'] == 'AMETHYSTS']
starfruit_df = df[df['product'] == 'STARFRUIT']

def plot_prices(df):

    plt.figure(figsize=(10, 6))  
    plt.plot(df['bid_price_1'], label='Bid', color='blue')  
    plt.plot(df['ask_price_1'], label='Ask', color='blue')  
    plt.plot(df['bid_price_2'], label='Bid', color='orange')  
    plt.plot(df['ask_price_2'], label='Ask', color='orange')  
    plt.plot(df['bid_price_3'], label='Bid', color='red')  
    plt.plot(df['ask_price_3'], label='Ask', color='red')  
    plt.plot(df['mid_price'], label='Mid Price', color='green')  

    plt.title('Bid, Ask, and Mid Price Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()  
    plt.xticks(rotation=45)  

    plt.show()


#plot_prices(amethysts_df)
#plot_prices(starfruit_df)


amethysts_df['spread'] = amethysts_df['ask_price_1'] - amethysts_df['bid_price_1']
starfruit_df['spread'] = starfruit_df['ask_price_1'] - starfruit_df['bid_price_1']

print('Average Spread Amethysts:', amethysts_df['spread'].mean())
print('Average Spread Starfruit:', starfruit_df['spread'].mean())