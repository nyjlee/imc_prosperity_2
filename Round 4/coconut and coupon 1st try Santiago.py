from typing import Dict, List
import pandas as pd
import numpy as np

class Trader:
    def __init__(self, prices: pd.DataFrame, trades: pd.DataFrame):
        self.prices = prices
        self.trades = trades

    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 30, num_std: float = 2) -> Dict[str, pd.Series]:
        mean = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = mean + (std * num_std)
        lower_band = mean - (std * num_std)
        return {'mean': mean, 'upper_band': upper_band, 'lower_band': lower_band}

    def simulate_bollinger_strategy(self, prices: pd.Series) -> pd.Series:
        bands = self.calculate_bollinger_bands(prices)
        signals = pd.Series(index=prices.index, data=0)
        signals[prices < bands['lower_band']] = 1  # Buy
        signals[prices > bands['upper_band']] = -1  # Sell
        returns = prices.pct_change()
        strategy_returns = signals.shift(1) * returns
        cumulative_returns = (strategy_returns + 1).cumprod() - 1
        return cumulative_returns

    def calculate_vwap(self, trades: pd.DataFrame) -> pd.Series:
        vwap = (trades['price'] * trades['quantity']).cumsum() / trades['quantity'].cumsum()
        return vwap

    def simulate_vwap_strategy(self, trades: pd.DataFrame) -> pd.Series:
        vwap = self.calculate_vwap(trades)
        signals = pd.Series(index=trades.index, data=0)
        signals[trades['price'] < vwap] = 1  # Buy
        signals[trades['price'] > vwap] = -1  # Sell
        returns = trades['price'].pct_change()
        strategy_returns = signals.shift(1) * returns
        cumulative_returns = (strategy_returns + 1).cumprod() - 1
        return cumulative_returns


trader = Trader(prices=pd.DataFrame(), trades=pd.DataFrame())
coconut_returns = trader.simulate_bollinger_strategy(coconut_prices)
coconut_coupon_returns = trader.simulate_vwap_strategy(coconut_coupon_trades)
