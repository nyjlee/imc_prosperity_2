import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

script_dir = os.path.dirname(__file__)


csv_file_path_0 = os.path.join(script_dir, 'data', 'prices_round_1_day_0.csv')
csv_file_path_1 = os.path.join(script_dir, 'data', 'prices_round_1_day_-1.csv')
csv_file_path_2 = os.path.join(script_dir, 'data', 'prices_round_1_day_-2.csv')


df_0 = pd.read_csv(csv_file_path_0, sep=';')
df_1 = pd.read_csv(csv_file_path_1, sep=';')
df_2 = pd.read_csv(csv_file_path_2, sep=';')


for i, df in enumerate([df_2, df_1, df_0]):
    df['timestamp'] = df['timestamp'] / 100 + i * 10000


df = pd.concat([df_2, df_1])
df = pd.concat([df, df_0])
"""
csv_file_path = os.path.join(script_dir, 'data', 'sample.csv')

df = pd.read_csv(csv_file_path, sep=';')
"""
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

def plot_spread(product_df):
    plt.figure(figsize=(10, 6))  
    plt.plot(product_df['bid_price_1'] - product_df['ask_price_1'], label='Spread', color='blue')  
    plt.title('Bid, Ask, and Mid Price Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()  
    plt.xticks(rotation=45)  

    plt.show()

#plot_spread(starfruit_df)
    

############# LINEAR REGRESSION OF LOG RETURNS #################


def regression_log_returns(df):
    df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1))

    # Create 4 lagged features
    for lag in range(1, 4):  # Creates lag_1, lag_2, lag_3, and lag_4
        df[f'lag_{lag}'] = df['log_return'].shift(lag)

    df = df.dropna()

    X = df[['lag_1', 'lag_2', 'lag_3']]
    y = df['log_return']

    # Split the data into training and test sets
    # Using the timestamp index to split the dataset
    train_index = 24000
    # Ensure that your index is sorted if it's not already
    X_train = X[X.index <= train_index]
    y_train = y[y.index <= train_index]

    X_test = X[X.index > train_index]
    y_test = y[y.index > train_index]

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse}')

    # After fitting the model
    print(f'Intercept: {model.intercept_}')
    print('Coefficients:')
    for i, coef in enumerate(model.coef_):
        print(f'lag_{i+1}: {coef}')

    #last_train_price = df.loc[y_train.index[-1], 'price']

    # Generate a Series for predicted values aligned with the test dataset index
    y_pred_series = pd.Series(y_pred, index=y_test.index)

    # Combine actual and predicted values into a single DataFrame
    combined_df = pd.DataFrame({
        'Actual': df['log_return'],
        'Predicted': pd.concat([pd.Series([None]*len(y_train)), y_pred_series])  # Placeholder None for train period
        })

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(combined_df.index, combined_df['Actual'], label='Actual Mid Price', color='blue')
    plt.plot(combined_df.index, combined_df['Predicted'], label='Predicted Mid Price', linestyle='--', color='red')
    plt.xlabel('Timestamp')
    plt.ylabel('Mid Price')
    plt.title('Actual vs Predicted Mid Prices')
    plt.legend()
    plt.show()

#regression_log_returns(starfruit_df)
    

############# LINEAR REGRESSION PRICES #############
    
def linear_regression(product_df):

    # Assuming 'df' is your DataFrame, and it's indexed by timestamp.
    # Step 1: Prepare the data with lagged features
    for i in range(1, 7):
        product_df[f'lag_{i}'] = product_df['mid_price'].shift(i)

    print(product_df)

    product_df = product_df.dropna()

    # Step 2: Manually split your data into features (X) and target (y), then into training and testing sets
    X = product_df[['lag_1', 'lag_2','lag_3', 'lag_4', 'lag_5']]
    y = product_df['mid_price']

    # Using the timestamp index to split the dataset
    train_index = 1000

    # Ensure that your index is sorted if it's not already
    X_train = X[X.index >= train_index]
    y_train = y[y.index >= train_index]

    X_test = X[X.index < train_index]
    y_test = y[y.index < train_index]

    # Step 3: Train your model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Evaluate your model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # After fitting the model
    print(f'Intercept: {model.intercept_}')
    print('Coefficients:')
    for i, coef in enumerate(model.coef_):
        print(f'lag_{i+1}: {coef}')

    # Generate a Series for predicted values aligned with the test dataset index
    y_pred_series = pd.Series(y_pred, index=y_test.index)

    # Combine actual and predicted values into a single DataFrame
    combined_df = pd.DataFrame({
        'Actual': product_df['mid_price'],
        'Predicted': pd.concat([pd.Series([None]*len(y_train)), y_pred_series])  # Placeholder None for train period
    })

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(combined_df.index, combined_df['Actual'], label='Actual Mid Price', color='blue')
    plt.plot(combined_df.index, combined_df['Predicted'], label='Predicted Mid Price', linestyle='--', color='red')
    plt.xlabel('Timestamp')
    plt.ylabel('Mid Price')
    plt.title('Actual vs Predicted Mid Prices')
    plt.legend()
    plt.show()

#linear_regression(starfruit_df)
