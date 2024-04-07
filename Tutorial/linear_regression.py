import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
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
    X = product_df[['lag_1', 'lag_2','lag_3', 'lag_4', 'lag_5']]
    y = product_df['mid_price']

    # Using the timestamp index to split the dataset
    train_index = 160000

    # Ensure that your index is sorted if it's not already
    X_train = X[X.index <= train_index]
    y_train = y[y.index <= train_index]

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

#linear_regression(starfruit_df)


def linear_regression_with_cv_and_parameters(product_df):
    # Copy the DataFrame to avoid any potential SettingWithCopyWarning when modifying
    product_df = product_df.copy()
    
    # Add lagged features safely using .loc to avoid SettingWithCopyWarning
    for i in range(1, 6):
        product_df.loc[:, f'lag_{i}'] = product_df['mid_price'].shift(i)

    # Clean rows with NaN values that were created by shifting
    product_df.dropna(inplace=True)

    # Prepare features and target variable
    X = product_df[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]
    y = product_df['mid_price']

    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []
    intercepts = []
    coefficients = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Store intercepts and coefficients for analysis
        intercepts.append(model.intercept_)
        coefficients.append(model.coef_)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    # Reporting average MSE, intercept, and coefficients
    average_mse = np.mean(mse_scores)
    print(f'Average Mean Squared Error across the folds: {average_mse}')

    avg_intercept = np.mean(intercepts)
    avg_coefficients = np.mean(coefficients, axis=0)

    print(f'Average Intercept: {avg_intercept}')
    for i, coef in enumerate(avg_coefficients):
        print(f'lag_{i+1}: {coef}')

linear_regression_with_cv_and_parameters(starfruit_df)