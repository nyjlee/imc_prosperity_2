from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade
import math
import pandas as pd
import numpy as np
import statistics

class Trader:
    PRODUCTS = [
        'AMETHYSTS',
        'STARFRUIT',
        'ORCHIDS',
        'GIFT_BASKET',
        'CHOCOLATE',
        'STRAWBERRIES',
        'ROSES',
        'COCONUT',
        'COCONUT_COUPON'
    ]

    DEFAULT_PRICES = {
        'AMETHYSTS': 10000,
        'STARFRUIT': 5000,
        'ORCHIDS': 1000,
        'GIFT_BASKET': 70000,
        'CHOCOLATE': 8000,
        'STRAWBERRIES': 4000,
        'ROSES': 15000,
        'COCONUT': 10000,
        'COCONUT_COUPON': 600
    }

    POSITION_LIMITS = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
        'ORCHIDS': 100,
        'GIFT_BASKET': 60,
        'CHOCOLATE': 250,
        'STRAWBERRIES': 350,
        'ROSES': 60,
        'COCONUT': 300,
        'COCONUT_COUPON': 600
    }

    def __init__(self):
        self.ema_prices = {product: None for product in self.PRODUCTS}
        self.ema_param = 0.1
        self.window_size = 21
        self.current_pnl = {product: 0 for product in self.PRODUCTS}
        self.qt_traded = {product: 0 for product in self.PRODUCTS}
        self.pnl_tracker = {product: [] for product in self.PRODUCTS}

    def run(self, state: TradingState):
        result = {product: [] for product in self.PRODUCTS}

        # Process each product
        for product in self.PRODUCTS:
            trades = state.market_trades.get(product, [])
            if trades:
                self.process_trades(trades, product)
                result[product] = self.generate_orders(product, trades)
            else:
                print(f"No market trades for {product}")

        conversions = 0  # Placeholder for conversion logic
        traderData = "SAMPLE"  # Placeholder for trader-specific data that needs to be returned

        return result, conversions, traderData

    def process_trades(self, trades, product):
        prices = pd.Series([t.price for t in trades if t is not None])
        if not prices.empty:
            # Update EMA prices
            current_ema = self.ema_prices[product]
            if current_ema is None:
                self.ema_prices[product] = prices.mean()
            else:
                self.ema_prices[product] = self.ema_param * prices.iloc[-1] + (1 - self.ema_param) * current_ema

    def generate_orders(self, product, trades):
        # This is a placeholder for the logic to generate orders based on strategy
        last_price = trades[-1].price if trades else self.DEFAULT_PRICES[product]
        # Example order generation logic
        order = Order(symbol=product, price=last_price, quantity=5)  # Simplified example
        return [order]

# Example usage
# Assuming 'state' is an instance of TradingState with relevant market data populated
trader = Trader()
results, conversions, traderData = trader.run(state)
print(results)
