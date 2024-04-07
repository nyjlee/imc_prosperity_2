"""
This module contains functions for polynomial regression. Regression is done 
using the normal equation method. Functions for plotting and computing R-squared
are also included. Consider using np.polynomial.polynomial.polyfit or 
numpy.polynomial.polynomial.Polynomial.fit (returns a Polynomial object) instead
of the functions in this module.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def polynomial_features(x: pd.Series, degree: int) -> np.ndarray:
    """
    Prepare polynomial features for a given degree.

    Parameters:
        x (pd.Series): The input feature series. Can be any object 
                       convertable to numpy array.
        degree (int): The degree of the polynomial.

    Returns:
        np.ndarray: A matrix of polynomial features.
    """
    x = np.array([x**i for i in range(degree + 1)]).T

    return x


def polynomial_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Perform polynomial regression.

    Parameters:
        x (np.ndarray): The input feature matrix.
        y (np.ndarray): The target variable.

    Returns:
        np.ndarray: The coefficients of the polynomial regression.
    """
    x = np.hstack((np.ones((x.shape[0], 1)), x)) # Bias term (intercept)

    coefficients = np.linalg.inv(x.T @ x) @ x.T @ y

    return coefficients


def model_predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Predict using the coefficients of the model.

    Parameters:
        X (np.ndarray): The input feature matrix.
        coefficients (np.ndarray): The coefficients of the model.

    Returns:
        np.ndarray: The predicted values.
    """
    x = np.hstack((np.ones((x.shape[0], 1)), x))  # Bias term (intercept)

    return x @ coefficients


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R-squared value.

    Parameters:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.

    Returns:
        float: The R-squared value.
    """
    numerator = ((y_true - y_pred)**2).sum()
    denominator = ((y_true - y_true.mean())**2).sum()
    return 1 - (numerator / denominator)

def model_plot(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot the model.

    Parameters:
        x (np.ndarray): The input feature matrix.
        y (np.ndarray): The target variable.
        y_pred (np.ndarray): The predicted values.
    """

    plt.scatter(x, y, color='blue', label='True values')
    plt.plot(x, y_pred, color='red', label='Predicted values')
    plt.legend()
    plt.show()