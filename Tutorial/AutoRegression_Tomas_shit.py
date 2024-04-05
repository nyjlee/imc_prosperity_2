from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade
import math
import pandas as pd


class Trader:
    PRODUCTS = [
        'AMETHYSTS',
        'STARFRUIT',
    ]

    DEFAULT_PRICES = {
        'AMETHYSTS': 10000,
        'STARFRUIT': 5000,
    }

    POSITION_LIMITS = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
    }

    prices_history = {"AMETHYSTS": [], "STARFRUIT": []}

    def __init__(self) -> None:

        self.ema_prices = dict()
        for product in self.PRODUCTS:
            self.ema_prices[product] = None
        self.ema_param = 0.2

        self.window_size = 2000

    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)

    def get_best_bid(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders
        best_bid = max(market_bids)

        return best_bid

    def get_best_ask(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        best_ask = min(market_asks)

        return best_ask

    def get_mid_price(self, product, state: TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = self.DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price

        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask) / 2

    def get_last_price(self, symbol, own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]]):
        recent_trades = []
        if symbol in own_trades:
            recent_trades.extend(own_trades[symbol])
        if symbol in market_trades:
            recent_trades.extend(market_trades[symbol])
        recent_trades.sort(key=lambda trade: trade.timestamp)
        last_trade = recent_trades[-1]
        return last_trade.price

    def update_prices_history(self, own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]]):
        for symbol in self.PRODUCTS:
            recent_trades = []
            if symbol in own_trades:
                recent_trades.extend(own_trades[symbol])
            if symbol in market_trades:
                recent_trades.extend(market_trades[symbol])

            recent_trades.sort(key=lambda trade: trade.timestamp)

            for trade in recent_trades:
                self.prices_history[symbol].append(trade.price)

            while len(self.prices_history[symbol]) > self.window_size:
                self.prices_history[symbol].pop(0)

    def update_ema_prices(self, state: TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in self.PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1 - self.ema_param) * self.ema_prices[product]

        print(self.ema_prices)

    def calculate_sma(self, product, window_size):
        sma = None
        prices = pd.Series(self.prices_history[product])
        if len(prices) >= window_size:
            window_sum = prices.iloc[-window_size:].sum()
            sma = window_sum / window_size
        return sma

    def cal_next_price_starfruit(self):
        coef = [0.3213974, 0.22728502, 0.23948268, 0.21125028]
        intercept = 2.85793298
        # Initialize next price with intercept
        nxt_price = intercept
        # Get the last four recorded prices for 'STARFRUIT'
        prices = self.prices_history.get('STARFRUIT', [])
        if len(prices) >= 4:
            # Calculate the next predicted price based on the coefficients and the last four recorded prices
            for i, val in enumerate(prices[-4:]):
                nxt_price += float(val) * coef[i]
            print('Next price for STARFRUIT:', int(round(nxt_price)))
            return int(round(nxt_price))
        else:
            print('Insufficient data to predict next price for STARFRUIT')
            return None




    def amethysts_strategy(self, state: TradingState) -> List[Order]:

        # Returns a list of orders with trades of amethysts.

        orders = []

        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        ask_volume = - self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts

        if position_amethysts == 0:
            # Not long nor short
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] - 1), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 1), ask_volume))

        if position_amethysts > 0:
            # Long position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] - 2), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 1), ask_volume))

        if position_amethysts < 0:
            # Short position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] - 1), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 2), ask_volume))

        print(orders)

        return orders

    def starfruit_strategy(self, state: TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of starfruit.
        """
        orders = []

        position_starfruit = self.get_position('STARFRUIT', state)

        bid_volume = self.POSITION_LIMITS['STARFRUIT'] - position_starfruit
        ask_volume = - self.POSITION_LIMITS['STARFRUIT'] - position_starfruit

        mid_price = self.get_mid_price('STARFRUIT', state)

        last_price = self.get_last_price('STARFRUIT', state.own_trades, state.market_trades)
        print('Last Price:', last_price)

        best_bid = self.get_best_bid('STARFRUIT', state)
        best_ask = self.get_best_ask('STARFRUIT', state)

        # Calculate the next predicted price for STARFRUIT
        nxt_price = self.cal_next_price_starfruit()

        # Buying condition: current price is inferior to next predicted price
        if last_price < nxt_price:
            orders.append(Order('STARFRUIT', best_ask, bid_volume))

        # Selling condition: current price is superior to next predicted price
        elif last_price > nxt_price:
            orders.append(Order('STARFRUIT', best_bid, ask_volume))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS': [], 'STARFRUIT': []}

        self.update_ema_prices(state)

        # PRICE HISTORY
        self.update_prices_history(state.own_trades, state.market_trades)
        # print(self.prices_history)

        """
        # AMETHYSTS STRATEGY
        try:
            result['AMETHYSTS'] = self.amethysts_strategy(state)
        except Exception as e:
            print("Error in AMETHYSTS strategy")
            print(e)
        """

        # STARFRUIT STRATEGY

        try:
            result['STARFRUIT'] = self.starfruit_strategy(state)
        except Exception as e:
            print("Error in STARFRUIT strategy")
            print(e)

        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
