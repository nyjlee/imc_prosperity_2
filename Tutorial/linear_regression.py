import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(__file__)

csv_file_path = os.path.join(script_dir, 'data', 'tutorial_round.csv')

df = pd.read_csv(csv_file_path, sep=';')

df = df.set_index('timestamp')

amethysts_df = df[df['product'] == 'AMETHYSTS']
starfruit_df = df[df['product'] == 'STARFRUIT']

#print(starfruit_df)

def linear_regression(product_df):

    # Assuming 'df' is your DataFrame, and it's indexed by timestamp.
    # Step 1: Prepare the data with lagged features
    for i in range(1, 6):
        product_df[f'lag_{i}'] = product_df['mid_price'].shift(i)

    print(product_df)


    # Remove the rows with NaN values that were created by shifting
    product_df  = product_df.iloc[3:].copy()



    # Step 2: Manually split your data into features (X) and target (y), then into training and testing sets
    X = product_df[['lag_1', 'lag_2','lag_3']]
    y = product_df['mid_price']
    product_df['mid_price_2'] = product_df['mid_price'].iloc[::-1].values
    #y = product_df['mid_price_2']

    # Using the timestamp index to split the dataset
    train_index = 160000

    # Ensure that your index is sorted if it's not already
    X_train = X[X.index <= train_index]
    y_train = y[y.index <= train_index]

    y = product_df['mid_price_2']
    X_test = X[X.index > train_index]
    y_test = y[y.index > train_index]

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


linear_regression(starfruit_df)