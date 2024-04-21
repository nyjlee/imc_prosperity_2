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
from statsmodels.tsa.vector_ar.vecm import VECM, select_order
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import math
import os
from arch import arch_model


forecasted = np.sqrt(1.0489105974177813 / 1e+08)
annualized_volatility = forecasted * np.sqrt(252)
#annualized_volatility = annualized_volatility * np.sqrt(10000)
print(f"Annualized Forecasted Volatility: {annualized_volatility}")


script_dir = os.path.dirname(__file__)


csv_file_path_0 = os.path.join(script_dir, 'data', 'prices_round_4_day_1.csv')
csv_file_path_1 = os.path.join(script_dir, 'data', 'prices_round_4_day_2.csv')
csv_file_path_2 = os.path.join(script_dir, 'data', 'prices_round_4_day_3.csv')


df_0 = pd.read_csv(csv_file_path_0, sep=';')
df_1 = pd.read_csv(csv_file_path_1, sep=';')
df_2 = pd.read_csv(csv_file_path_2, sep=';')


for i, df in enumerate([df_0, df_1, df_2]):
    df['timestamp'] = df['timestamp'] / 100 + i * 10000


df = pd.concat([df_0, df_1])
df = pd.concat([df, df_2])
df = df.set_index('timestamp')

coconut = df[df['product'] == 'COCONUT']['mid_price']
coconut_coupon = df[df['product'] == 'COCONUT_COUPON']['mid_price']
sampled_coconut_prices = coconut.iloc[::10]
sampled_coconut_prices = sampled_coconut_prices.reset_index(drop=True)
"""
model = ARIMA(sampled_coconut_prices, order=(3,1,3))
model_fit = model.fit()
print(model_fit.summary())

# Get the in-sample residuals
residuals = model_fit.resid
initial_errors = residuals[-10:] 
print(initial_errors)

df = pd.DataFrame({
    'coconut_price': coconut,
    'coconut_call': coconut_coupon
})

df['coconut_returns'] = df['coconut_price'].pct_change().fillna(0)

df['scaled_returns'] = df['coconut_returns'] * 1e+04


# Fit the GARCH model using scaled returns
garch_model = arch_model(df['coconut_returns'], vol='Garch', p=1, q=1, dist='Normal')
model_result = garch_model.fit(update_freq=5)
# Retrieve the conditional variances
conditional_variances = model_result.conditional_volatility ** 2
# Get the last observed variance
last_observed_variance = conditional_variances.iloc[-1]
print("Last observed variance according to the GARCH model:", last_observed_variance)

print(model_result.summary())
"""

def norm_cdf(x):
    """
    Calculate the CDF of the standard normal distribution at x using the error function.
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes_call(S, K, T, sigma, r=0.0):
    """
    Calculate the Black-Scholes price of a European call option.

    Parameters:
    - S (float): Current stock price
    - K (float): Strike price of the option
    - T (float): Time to expiration in years (250/365 for your case)
    - sigma (float): Annualized volatility of the stock
    - r (float): Risk-free rate, default to 0.0 as per your situation

    Returns:
    - float: Price of the call option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))
    return call_price


def implied_volatility(market_price, S, K, T, initial_sigma=0.2, tolerance=1e-5, max_iterations=100):
    """
    Estimate the implied volatility of a European call option using the bisection method.
    
    Parameters:
    - market_price (float): The observed market price of the option.
    - S (float): Current stock price.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - initial_sigma (float): Initial guess for volatility.
    - tolerance (float): Tolerance level for stopping the search.
    - max_iterations (int): Maximum number of iterations to perform.
    
    Returns:
    - float: Estimated implied volatility.
    """
    sigma_low = 0.001
    sigma_high = 1.0
    for i in range(max_iterations):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = black_scholes_call(S, K, T, sigma_mid, r=0.0)
        
        if price_mid > market_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
            
        if abs(price_mid - market_price) < tolerance:
            return sigma_mid
    
    return (sigma_low + sigma_high) / 2  # Return the best estimate if convergence criterion not met


vol = implied_volatility(637.63, 10000, 10000, 246/250)
print('iv', vol)

call_price = black_scholes_call(10000, 10000, 246/250, vol)
print('call price', call_price)

# Assuming the following values
sigma = 0.195  # Estimated volatility (20% annualized)
T = 246 / 250  # Time to expiration in years
K =10000

df = pd.DataFrame({
    'coconut_price': coconut,
    'coconut_call': coconut_coupon
})

# Calculate the call prices using the Black-Scholes formula
#df['call_price'] = df['coconut_price'].apply(lambda S: black_scholes_call(S, 10000, T, sigma))


def calculate_iv(row):
    S = row['coconut_price']
    market_price = row['coconut_call']
    return implied_volatility(market_price, S, K, T)

# Apply the function to each row
df['implied_vol'] = df.apply(calculate_iv, axis=1)

max = df['implied_vol'].max()
min = df['implied_vol'].min()

print('max vol',max)
print('min vol',min)

print(df)