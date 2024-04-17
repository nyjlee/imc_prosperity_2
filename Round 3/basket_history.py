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
    nav = (4*chocolates + 6*strawberries + roses)
    ratio = gift_basket/nav

    print(spread)
    print('ratio AVG:', spread.mean())
    print('spread min:', spread.min())
    print('spread max:', spread.max())
    print('spread STD:', spread.std())

#basket_spread(df)


chocolates = df[df['product'] == 'CHOCOLATE']['mid_price']
strawberries = df[df['product'] == 'STRAWBERRIES']['mid_price']
roses = df[df['product'] == 'ROSES']['mid_price']
gift_basket = df[df['product'] == 'GIFT_BASKET']['mid_price']

# Combine them into a single DataFrame
df = pd.DataFrame({
    'chocolates': chocolates,
    'strawberries': strawberries,
    'roses': roses,
    'gift_basket': gift_basket
})

# Ensure all are aligned by time and drop any rows with missing data
df = df.dropna()

def check_stationarity(series, name):
    result = adfuller(series, autolag='AIC')
    print(f'Results for {name}:')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] > 0.05:
        print(f"{name} is not stationary")
    else:
        print(f"{name} is stationary")

#check_stationarity(chocolates, 'chocolates')
#check_stationarity(strawberries, 'strawberries')
#check_stationarity(roses, 'roses')
#check_stationarity(gift_basket, 'gift_basket')


# Differencing non-stationary series
df['chocolates_diff'] = df['chocolates'].diff().dropna()
df['strawberries_diff'] = df['strawberries'].diff().dropna()
df['gift_basket_diff'] = df['gift_basket'].diff().dropna()

# Since 'roses' are already stationary, we do not difference them, but let's align all series by dropping any NaN values
df = df.dropna()

# Re-check stationarity after differencing
check_stationarity(df['chocolates_diff'], 'chocolates_diff')
check_stationarity(df['strawberries_diff'], 'strawberries_diff')
check_stationarity(df['gift_basket_diff'], 'gift_basket_diff')


data_for_johansen = df[['roses', 'chocolates_diff', 'strawberries_diff', 'gift_basket_diff']]

# Perform the Johansen cointegration test
# k_ar_diff = number of lags minus 1 used in the test, here we use 1 lag
johansen_test = coint_johansen(data_for_johansen, det_order=0, k_ar_diff=1)

# Print the results
print("Eigenvalues:", johansen_test.eig)  # Eigenvalues of the test
print("Cointegration Test Statistic (Trace):", johansen_test.lr1)  # Test statistics
print("Critical Values (Trace):", johansen_test.cvt)  # Critical values for the test statistics at different significance levels
print("Cointegration Test Statistic (Max Eigen):", johansen_test.lr2)  # Maximum eigenvalue statistics
print("Critical Values (Max Eigen):", johansen_test.cvm)  # Critical values for max eigen statistic


# Fit VAR on levels of data to determine optimal lags
var_model = VAR(data_for_johansen)
var_result = var_model.select_order(8)  # Checks up to 15 lags, adjust as necessary
print(var_result.summary())

"""
# Assuming 'series' is your Pandas Series with the time series data
result = adfuller(chocolates)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


model = ARIMA(chocolates, order=(6,1,3))
model_fit = model.fit()
print(model_fit.summary())

# Get the in-sample residuals
residuals = model_fit.resid
initial_errors = residuals[-2:] 
print(initial_errors)
"""

"""

else:
            spread_chocolate = chocolates_ask - chocolates_bid
            if len(self.mid_p_diff_history['CHOCOLATE']) >= 6:
                AR_L1 = self.mid_p_diff_history['CHOCOLATE'][-1]
                AR_L2 = self.mid_p_diff_history['CHOCOLATE'][-2]
                AR_L3 = self.mid_p_diff_history['CHOCOLATE'][-3]
                AR_L4 = self.mid_p_diff_history['CHOCOLATE'][-4]
                AR_L5 = self.mid_p_diff_history['CHOCOLATE'][-5]
                AR_L6 = self.mid_p_diff_history['CHOCOLATE'][-6]
        
            if len(self.forecasted_diff_history['CHOCOLATE']) > 0:
                forecasted_error = self.forecasted_diff_history['CHOCOLATE'][-1] - self.mid_p_diff_history['CHOCOLATE'][-1]
                self.errors_history['CHOCOLATE'].append(forecasted_error)

            if len(self.errors_history['CHOCOLATE']) < 2:
          
                self.errors_history['CHOCOLATE'].extend([0.015003, -0.505039, -0.010648])

    
            else:
                MA_L1 = self.errors_history['CHOCOLATE'][-1]
                MA_L2 = self.errors_history['CHOCOLATE'][-2]
                MA_L3 = self.errors_history['CHOCOLATE'][-3]
            

            forecasted_diff = (AR_L1 * -0.2603) + (AR_L2 * -0.7993) + (AR_L3 * 0.2902) + (AR_L4 * 0.0098)
            + (AR_L5 * 0.0074) + (AR_L6 * -0.0030)+ (MA_L1 *  0.2433) + (MA_L2 * 0.7976) + (MA_L3 * -0.2965)

            self.forecasted_diff_history['CHOCOLATE'].append(forecasted_diff)

            forecasted_price = chocolates_mid_price + forecasted_diff  

            #play with diff comb
            if forecasted_price > chocolates_bid+2:
                chocolates_orders.append(Order('CHOCOLATE', math.floor(chocolates_bid+1), buy_volume_chocolates/2))

                chocolates_orders.append(Order('CHOCOLATE', math.floor(forecasted_price+spread_chocolate/2), int(math.floor(sell_volume_chocolates/2))))
                chocolates_orders.append(Order('CHOCOLATE', math.floor(forecasted_price+spread_chocolate/3), int(math.ceil(sell_volume_chocolates/2))))
            elif forecasted_price < chocolates_ask-2:
                chocolates_orders.append(Order('CHOCOLATE', math.floor(chocolates_ask-1), sell_volume_chocolates/2))

                chocolates_orders.append(Order('CHOCOLATE', math.ceil(forecasted_price-spread_chocolate/2), int(math.floor(buy_volume_chocolates/2))))
                chocolates_orders.append(Order('CHOCOLATE', math.ceil(forecasted_price-spread_chocolate/3), int(math.ceil(buy_volume_chocolates/2))))
"""


"""
            elif z_score > 2.5: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                qt_basket = min(abs(sell_volume_basket*10), basket_bid_vol) 
                print(qt_basket)
                qt_chocolates = min(buy_volume_chocolates*10, abs(chocolates_ask_vol)) 
                print(qt_chocolates)
                qt_strawberries = min(buy_volume_strawberries*10, abs(strawberries_ask_vol)) 
                print(qt_strawberries)
                qt_roses = min(buy_volume_roses*10, abs(roses_ask_vol)) 
                print(qt_roses)

                n_tradable = min(qt_basket, math.floor(qt_chocolates/4), math.floor(qt_strawberries/6), qt_roses)
                print(n_tradable)

                if n_tradable > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_bid, - n_tradable))
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, n_tradable * 4))
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, n_tradable * 6))
                    roses_orders.append(Order('ROSES', roses_ask, n_tradable))

            elif z_score < -2.5: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket*10, abs(basket_ask_vol)) 
                print(qt_basket)
                qt_chocolates = min(abs(sell_volume_chocolates*10), chocolates_bid_vol) 
                print(qt_chocolates)
                qt_strawberries = min(abs(sell_volume_strawberries*10), strawberries_bid_vol) 
                print(qt_strawberries)
                qt_roses = min(abs(sell_volume_roses*10), roses_bid_vol) 
                print(qt_roses)

                n_tradable = min(qt_basket, math.floor(qt_chocolates / 4), math.floor(qt_strawberries / 6), qt_roses)
                print(n_tradable)
                if n_tradable > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_ask-2, n_tradable))
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_bid+2, -n_tradable * 4))
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid+2, -n_tradable * 6))
                    roses_orders.append(Order('ROSES', roses_bid+2, -n_tradable))
            """

import statsmodels.api as sm

# y and x are your datasets
x = sm.add_constant(strawberries)  # adding a constant for the regression intercept
model = sm.OLS(roses, strawberries).fit()
residuals = model.resid


from statsmodels.tsa.stattools import adfuller

result = adfuller(residuals)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])