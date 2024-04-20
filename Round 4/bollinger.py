from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade
import math
import pandas as pd
import numpy as np
import statistics

class Trader:
    PRODUCTS = [
        'AMETHYSTS', 'STARFRUIT', 'ORCHIDS', 'GIFT_BASKET', 'CHOCOLATE',
        'STRAWBERRIES', 'ROSES', 'COCONUT', 'COCONUT_COUPON'
    ]

    DEFAULT_PRICES = {
        'AMETHYSTS': 10000, 'STARFRUIT': 5000, 'ORCHIDS': 1000, 'GIFT_BASKET': 70000,
        'CHOCOLATE': 8000, 'STRAWBERRIES': 4000, 'ROSES': 15000, 'COCONUT': 10000,
        'COCONUT_COUPON': 600
    }

    POSITION_LIMITS = {
        'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'GIFT_BASKET': 60,
        'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'COCONUT': 300,
        'COCONUT_COUPON': 600
    }

    def __init__(self):
        self.prices_history = {product: [] for product in self.PRODUCTS}
        self.mid_prices_history = {product: [] for product in self.PRODUCTS}
        self.p_diff_history = {product: [] for product in self.PRODUCTS}
        self.errors_history = {product: [] for product in self.PRODUCTS}
        self.forecasted_diff_history = {product: [] for product in self.PRODUCTS}
        self.current_pnl = {product: 0 for product in self.PRODUCTS}
        self.current_position = {product: 0 for product in self.PRODUCTS}
        self.qt_traded = {product: 0 for product in self.PRODUCTS}
        self.pnl_tracker = {product: [] for product in self.PRODUCTS}

    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 30, num_std: float = 2):
        mean = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = mean + (std * num_std)
        lower_band = mean - (std * num_std)
        return lower_band, upper_band

    def simulate_bollinger_strategy(self, prices: pd.Series) -> List[Order]:
        lower_band, upper_band = self.calculate_bollinger_bands(prices)
        orders = []
        for i, price in enumerate(prices):
            if price < lower_band[i]:
                orders.append(Order('COCONUT', price, 'BUY'))
            elif price > upper_band[i]:
                orders.append(Order('COCONUT', price, 'SELL'))
        return orders

    def calculate_vwap(self, trades: pd.DataFrame):
        vwap = (trades['price'] * trades['quantity']).cumsum() / trades['quantity'].cumsum()
        return vwap

    def simulate_vwap_strategy(self, trades: pd.DataFrame) -> List[Order]:
        vwap = self.calculate_vwap(trades)
        orders = []
        for i, trade in trades.iterrows():
            if trade['price'] < vwap[i]:
                orders.append(Order('COCONUT_COUPON', trade['price'], 'BUY'))
            elif trade['price'] > vwap[i]:
                orders.append(Order('COCONUT_COUPON', trade['price'], 'SELL'))
        return orders

    def run(self, state: TradingState):
        coconut_prices = pd.Series([t.price for t in state.market_trades['COCONUT'] if t is not None])
        coconut_trades = pd.DataFrame({
            'price': [t.price for t in state.market_trades['COCONUT'] if t is not None],
            'quantity': [t.quantity for t in state.market_trades['COCONUT'] if t is not None]
        })
        coconut_coupon_trades = pd.DataFrame({
            'price': [t.price for t in state.market_trades['COCONUT_COUPON'] if t is not None],
            'quantity': [t.quantity for t in state.market_trades['COCONUT_COUPON'] if t is not None]
        })

        coconut_orders = self.simulate_bollinger_strategy(coconut_prices)
        coconut_coupon_orders = self.simulate_vwap_strategy(coconut_coupon_trades)

        result = {
            'COCONUT': coconut_orders,
            'COCONUT_COUPON': coconut_coupon_orders
        }
        return result

# Example usage
# state = TradingState(...) 
# trader = Trader()
# results = trader.run(state)
