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


csv_file_path_0 = os.path.join(script_dir, 'data', 'prices_round_2_day_0.csv')
csv_file_path_1 = os.path.join(script_dir, 'data', 'prices_round_2_day_1.csv')
csv_file_path_2 = os.path.join(script_dir, 'data', 'prices_round_2_day_2.csv')


df_0 = pd.read_csv(csv_file_path_0, sep=';')
df_1 = pd.read_csv(csv_file_path_1, sep=';')
df_2 = pd.read_csv(csv_file_path_2, sep=';')


for i, df in enumerate([df_0, df_1, df_2]):
    df['timestamp'] = df['timestamp'] / 100 + i * 10000


df = pd.concat([df_0, df_1])
df = pd.concat([df, df_2])
df = df.set_index('timestamp')

def linear_regression(products_df):


    # Step 2: Manually split your data into features (X) and target (y), then into training and testing sets
    chocolates = [df['product'] == 'CHOCOLATES']['mid_price']
    strawberries = [df['product'] == 'STRAWBERRIES']['mid_price']
    strawberries = [df['product'] == 'ROSES']['mid_price']
    X = products_df
    y = [df['product'] == 'GIFT_BASKET']['mid_price']

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
        'Actual': products_df['mid_price'],
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
