from typing import Dict, List
import pandas as pd
import numpy as np
import math
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade


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
        self.data = {}

    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 30, num_std: float = 2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return rolling_mean, upper_band, lower_band

    def simulate_bollinger_strategy(self, prices: pd.Series) -> List[Order]:
        mean, upper_band, lower_band = self.calculate_bollinger_bands(prices)
        signals = pd.Series(index=prices.index, data=0)
        signals[prices < lower_band] = 1  # Buy
        signals[prices > upper_band] = -1  # Sell
        return signals.tolist()  # For the purpose of order creation, return list of signals

    def calculate_vwap(self, trades: pd.DataFrame):
        vwap = (trades['price'] * trades['quantity']).cumsum() / trades['quantity'].cumsum()
        return vwap

    def simulate_vwap_strategy(self, trades: pd.DataFrame) -> List[Order]:
        vwap = self.calculate_vwap(trades)
        signals = pd.Series(index=trades.index, data=0)
        signals[trades['price'] < vwap] = 1  # Buy
        signals[trades['price'] > vwap] = -1  # Sell
        return signals.tolist()  # For the purpose of order creation, return list of signals

    def coconut_strategy(self, state: TradingState) -> (List[Order], List[Order]):
        coconut_prices = pd.Series([o.price for o in state.market_trades['COCONUT'] if o is not None])
        coconut_trades = pd.DataFrame({
            'price': [o.price for o in state.market_trades['COCONUT'] if o is not None],
            'quantity': [o.quantity for o in state.market_trades['COCONUT'] if o is not None]
        })
        coconut_coupon_trades = pd.DataFrame({
            'price': [o.price for o in state.market_trades['COCONUT_COUPON'] if o is not None],
            'quantity': [o.quantity for o in state.market_trades['COCONUT_COUPON'] if o is not None]
        })

        coconut_signals = self.simulate_bollinger_strategy(coconut_prices)
        coconut_coupon_signals = self.simulate_vwap_strategy(coconut_coupon_trades)

        coconut_orders = [Order('COCONUT', p, q) for p, q in zip(coconut_prices, coconut_signals)]
        coconut_coupon_orders = [Order('COCONUT_COUPON', p, q) for p, q in
                                 zip(coconut_coupon_trades['price'], coconut_coupon_signals)]

        return coconut_orders, coconut_coupon_orders

    def run(self, state: TradingState):
        result, conversions, traderData = {}, 0, "SAMPLE"

        # Simulate strategies for Coconut and Coconut Coupon
        try:
            result['COCONUT'], result['COCONUT_COUPON'] = self.coconut_strategy(state)
        except Exception as e:
            print(f"Error in Coconut or Coconut Coupon strategy: {e}")

        return result, conversions, traderData

# Usage
# trader = Trader()
# state = TradingState(...)  # This needs to be defined based on the simulation environment
# results, conversions, traderData = trader.run(state)
