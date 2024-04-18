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
    'ROSES'
    ]

    DEFAULT_PRICES = {
        'AMETHYSTS' : 10000,
        'STARFRUIT' : 5000,
        'ORCHIDS': 1000,
        'GIFT_BASKET' : 70000,
        'CHOCOLATE' : 8000,
        'STRAWBERRIES': 4000,
        'ROSES': 15000,
    }

    POSITION_LIMITS = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
        'ORCHIDS': 100,
        'GIFT_BASKET': 60,
        'CHOCOLATE': 250,
        'STRAWBERRIES': 350,
        'ROSES': 60,
    }

    prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": []}
    mid_prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": []}
    mid_p_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": []}
    p_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": []}
    errors_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": []}
    forecasted_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": []}

    current_signal = {"AMETHYSTS": "", "STARFRUIT": "None", "ORCHIDS": "None"}

    export_tariffs = {"Min": 1000, "Max": 0, "Second Max": 0}

    spreads_basket = []
    ratios_basket = []

    etf_prices = []
    etf_returns = []
    nav_prices = []
    nav_returns = []


    def __init__(self) -> None:

        self.ema_prices = dict()
        for product in self.PRODUCTS:
            self.ema_prices[product] = None
        self.ema_param = 0.

        self.window_size = 21

        self.current_pnl = dict()
        self.qt_traded = dict()
        self.pnl_tracker = dict()

        for product in self.PRODUCTS:
            self.current_pnl[product] = 0
            self.qt_traded[product] = 0
            self.pnl_tracker[product] = []


    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    
    
    def get_order_book(self, product, state: TradingState):
        market_bids = list((state.order_depths[product].buy_orders).items())
        market_asks = list((state.order_depths[product].sell_orders).items())

        if len(market_bids) > 1:
            bid_price_1, bid_amount_1 = market_bids[0]
            bid_price_2, bid_amount_2 = market_bids[1]

        if len(market_asks) > 1:
            ask_price_1, ask_amount_1 = market_asks[0]
            ask_price_2, ask_amount_2 = market_asks[1]


        bid_price, ask_price = bid_price_1, ask_price_1

        if bid_amount_1 < 5:
            bid_price = bid_price_2
        else:
            bid_price = bid_price_1 + 1
        
        if ask_amount_1 < 5:
            ask_price = ask_price_2
        else:
            ask_price = ask_price_1 - 1

        return bid_price, ask_price
    

    def get_best_bid(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders
        #best_bid = max(market_bids)
        best_bid, best_bid_amount = list(market_bids.items())[0]

        return best_bid, best_bid_amount

    def get_best_ask(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        best_ask, best_ask_amount = list(market_asks.items())[0]

        return best_ask, best_ask_amount

    def get_bid2(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 2:
            bid2, bid2_amount = list(market_bids.items())[1]
        else:
            bid2, bid2_amount = float('-inf'), 0

        return bid2, bid2_amount
    
    def get_bid3(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 3:
            bid3, bid3_amount = list(market_bids.items())[2]
        else:
            bid3, bid3_amount = float('-inf'), 0

        return bid3, bid3_amount
    
    def get_ask2(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        if len(market_asks) == 2:
            ask2, ask2_amount = list(market_asks.items())[1]
        else:
            ask2, ask2_amount = float('inf'), 0

        return ask2, ask2_amount
    
    def get_ask3(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        if len(market_asks) == 2:
            ask3, ask3_amount = list(market_asks.items())[2]
        else:
            ask3, ask3_amount = float('inf'), 0

        return ask3, ask3_amount
    
    def get_mid_price(self, product, state : TradingState):

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
        return (best_bid + best_ask)/2   

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

    def update_mid_prices_history(self, state):
            for symbol in self.PRODUCTS:
                mid_price = self.get_mid_price(symbol, state)

                self.mid_prices_history[symbol].append(mid_price)

                while len(self.mid_prices_history[symbol]) > self.window_size:
                    self.mid_prices_history[symbol].pop(0)

    def update_diff_history(self, diff_history, p_history):
        for symbol in self.PRODUCTS:
            if len(p_history[symbol]) >=2:
                diff = p_history[symbol][-1] - p_history[symbol][-2]
                
                diff_history[symbol].append(diff)

            while len(diff_history[symbol]) > 8:
                diff_history[symbol].pop(0)


    def update_ema_prices(self, state : TradingState):
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
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]

        #print(self.ema_prices)

    def calculate_sma(self, product, window_size):
        sma = None
        prices = pd.Series(self.mid_prices_history[product])
        if len(prices) >= window_size:
            window_sum = prices.iloc[-window_size:].sum()
            sma = window_sum / window_size
        return sma
    
    def calculate_ema(self, product, window_size):
        ema = None
        prices = pd.Series(self.mid_prices_history[product])
        if len(prices) >= window_size:
            ema = prices.ewm(span=window_size, adjust=False).mean().iloc[-1]
        return ema
    

    def calculate_vwap(self, symbol, own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]]):
        vwap = None
        recent_trades = []
        prices = []
        volumes = []
        if symbol in own_trades:
            recent_trades.extend(own_trades[symbol])
        if symbol in market_trades:
            recent_trades.extend(market_trades[symbol])

        recent_trades.sort(key=lambda trade: trade.timestamp)

        for trade in recent_trades:
            prices.append(trade.price)
            volumes.append(trade.quantity)

        data = pd.DataFrame({'prices': prices, 'volumes': volumes})
        vwap = (data['prices'] * data['volumes']).sum() / data['volumes'].sum()
        return vwap

    def calculate_standard_deviation(self, values: List[float]) -> float:
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]

        variance = sum(squared_diffs) / len(values)

        std_dev = math.sqrt(variance)

        return std_dev

    def calculate_order_book_imbalance(self, symbol, state: TradingState):
        if symbol not in state.order_depths:
            return None
        order_book = state.order_depths[symbol]
        bid_volume = sum(order_book.buy_orders.values())
        ask_volume = sum(order_book.sell_orders.values())

        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
            return imbalance
        else:
            print(total_volume)
            return 0
        

    def basket_strategy(self, state : TradingState) -> List[Order]:
        """
        Buying and Selling based on last trade price vs mean price (ceiling floor version)
        """
        basket_orders = []
        chocolates_orders = []
        strawberries_orders = []
        roses_orders = []

        #### POSITIONS ####
        position_basket = self.get_position('GIFT_BASKET', state)
        position_chocolates = self.get_position('CHOCOLATE', state)
        position_strawberries = self.get_position('STRAWBERRIES', state)
        position_roses = self.get_position('ROSES', state)

        current_position_all = abs(position_basket) + abs(position_chocolates) + abs(position_strawberries) + abs(position_roses)

        #### QUANTITIES WE ARE ALLOWED TO TRADE ####
        buy_volume_basket = (self.POSITION_LIMITS['GIFT_BASKET'] - position_basket) 
        sell_volume_basket = (- self.POSITION_LIMITS['GIFT_BASKET'] - position_basket) 
        buy_volume_chocolates = (self.POSITION_LIMITS['CHOCOLATE'] - position_chocolates) 
        sell_volume_chocolates = (- self.POSITION_LIMITS['CHOCOLATE'] - position_chocolates) 
        buy_volume_strawberries = (self.POSITION_LIMITS['STRAWBERRIES'] - position_strawberries) 
        sell_volume_strawberries = (- self.POSITION_LIMITS['STRAWBERRIES'] - position_strawberries) 
        buy_volume_roses = (self.POSITION_LIMITS['ROSES'] - position_roses) 
        sell_volume_roses= (- self.POSITION_LIMITS['ROSES'] - position_roses) 
        
        #### MID PRICES ####
        basket_mid_price = self.get_mid_price('GIFT_BASKET', state)
        chocolates_mid_price = self.get_mid_price('CHOCOLATE', state)
        strawberries_mid_price = self.get_mid_price('STRAWBERRIES', state)
        roses_mid_price = self.get_mid_price('ROSES', state)

        #### BIDS, ASKS, VOLUMES ####
        basket_bid, basket_bid_vol = self.get_best_bid('GIFT_BASKET', state)
        basket_ask, basket_ask_vol = self.get_best_ask('GIFT_BASKET', state)
        chocolates_bid, chocolates_bid_vol = self.get_best_bid('CHOCOLATE', state)
        chocolates_ask, chocolates_ask_vol = self.get_best_ask('CHOCOLATE', state)
        strawberries_bid, strawberries_bid_vol = self.get_best_bid('STRAWBERRIES', state)
        strawberries_ask, strawberries_ask_vol = self.get_best_ask('STRAWBERRIES', state)
        roses_bid, roses_bid_vol = self.get_best_bid('ROSES', state)
        roses_ask, roses_ask_vol = self.get_best_ask('ROSES', state)

        if len(self.mid_p_diff_history) > 5:

            latest_roses = self.mid_prices_history['ROSES'][-1]
            #roses_2 = self.mid_prices_history['ROSES'][-2]

            last_chocolates_diff = self.mid_p_diff_history['CHOCOLATE'][-1]
            #chocolate_diff_2 = self.mid_p_diff_history['CHOCOLATE'][-2]

            last_strawberries_diff = self.mid_p_diff_history['STRAWBERRIES'][-1]
            #strawberries_diff_2 = self.mid_p_diff_history['STRAWBERRIES'][-2]

            last_gift_basket_diff = self.mid_p_diff_history['GIFT_BASKET'][-1]
            #basket_diff_2 = self.mid_p_diff_history['GIFT_BASKET'][-2]

            # Coefficients
            beta1_1, beta1_2, beta1_3, beta1_4 = 1.0000, 2.393e-10, 1.224e-10, 3.82e+05
            beta2_1, beta2_2, beta2_3, beta2_4 = -7.448e-25, 1.0000, -1.015e-16, -0.2159
            beta3_1, beta3_2, beta3_3, beta3_4 = -8.289e-25, 9.444e-19, 1.0000, -0.0110

            # Calculate each ECT
            ec1 = (beta1_1 * latest_roses + beta1_2 * last_chocolates_diff +
                beta1_3 * last_strawberries_diff + beta1_4 * last_gift_basket_diff)

            ec2 = (beta2_1 * latest_roses + beta2_2 * last_chocolates_diff +
                beta2_3 * last_strawberries_diff + beta2_4 * last_gift_basket_diff)

            ec3 = (beta3_1 * latest_roses + beta3_2 * last_chocolates_diff +
                beta3_3 * last_strawberries_diff + beta3_4 * last_gift_basket_diff)
            
            intercept = 0
            predicted_strawberries_diff = (
            intercept +
            -0.0032 * latest_roses +
            0.0074 * last_chocolates_diff +
            0.0282 * last_strawberries_diff +
            -0.0010 * last_gift_basket_diff +
            -3.058e-08 * ec1 +  # Calculated from your provided beta coefficients
            -0.0134 * ec2 +     # and latest values
            -1.1759 * ec3
            )
          
            next_price = strawberries_mid_price + predicted_strawberries_diff
            print("Forecast:", next_price)

            print("Mid Price:", self.mid_prices_history['STRAWBERRIES'][-1])


        return basket_orders, chocolates_orders, strawberries_orders, roses_orders
    

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [], 'GIFT_BASKET' : [], 'CHOCOLATE' : [], 'STRAWBERRIES' : [], 'ROSES' : []}

        self.update_ema_prices(state)

        # PRICE HISTORY
        self.update_prices_history(state.own_trades, state.market_trades)
        self.update_mid_prices_history(state)
        #self.update_diff_history(self.mid_prices_history)
        self.update_diff_history(self.mid_p_diff_history, self.mid_prices_history)
        self.update_diff_history(self.p_diff_history, self.prices_history)
        #print(self.prices_history)

        """
        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp-100:
                    continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
                self.qt_traded[product] += abs(trade.quantity)
                if trade.buyer == "SUBMISSION":
                    self.current_pnl[product] -= trade.quantity * trade.price
                else:
                    self.current_pnl[product] += trade.quantity * trade.price

        
        final_pnl = 0
        for product in state.order_depths.keys():
            product_pnl = 0
            settled_pnl = 0
            best_sell = min(state.order_depths[product].sell_orders.keys())
            best_buy = max(state.order_depths[product].buy_orders.keys())
            mid_price = (best_sell + best_buy) / 2

            if self.get_position(product, state) < 0:
                settled_pnl += self.get_position(product, state) * mid_price
            else:
                settled_pnl += self.get_position(product, state) * mid_price
            product_pnl = settled_pnl + self.current_pnl[product]
            self.pnl_tracker[product].append(product_pnl)
            final_pnl += settled_pnl + self.current_pnl[product]
            print(f'\nFor product {product}, Pnl: {settled_pnl + self.current_pnl[product]}, Qty. Traded: {self.qt_traded[product]}')
        print(f'\nFinal Day Expected Pnl: {round(final_pnl,2)}')
        """
        

        for product in self.pnl_tracker.keys():
            while len(self.pnl_tracker[product]) > 20:
                self.pnl_tracker[product].pop(0)
            while len(self.forecasted_diff_history[product]) > 20:
                self.forecasted_diff_history[product].pop(0)
            while len(self.errors_history[product]) > 20:
                self.errors_history[product].pop(0)
        

        
        # BASKET STRATEGY
        try:
            result['GIFT_BASKET'], result['CHOCOLATE'], result['STRAWBERRIES'], result['ROSES'] = self.basket_strategy(state)
        
        except Exception as e:
            print(e)
        
             
        traderData = "SAMPLE" 
        
		# Sample conversion request. Check more details below. 
        conversions = 0
        return result, conversions, traderData