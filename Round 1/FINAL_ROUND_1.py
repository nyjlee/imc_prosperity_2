from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade
import math
import pandas as pd
import numpy as np


class Trader:

    PRODUCTS = [
    'AMETHYSTS',
    'STARFRUIT',
    ]

    DEFAULT_PRICES = {
        'AMETHYSTS' : 10000,
        'STARFRUIT' : 5000,
    }

    POSITION_LIMITS = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
    }

    prices_history = {"AMETHYSTS": [], "STARFRUIT": []}
    mid_prices_history = {"AMETHYSTS": [], "STARFRUIT": []}
    diff_history = {"AMETHYSTS": [], "STARFRUIT": []}
    errors_history = {"AMETHYSTS": [], "STARFRUIT": []}
    forecasted_diff_history = {"AMETHYSTS": [], "STARFRUIT": []}

    current_signal = {"AMETHYSTS": "", "STARFRUIT": "None"}


    def __init__(self) -> None:

        self.ema_prices = dict()
        for product in self.PRODUCTS:
            self.ema_prices[product] = None
        self.ema_param = 0.

        self.window_size = 81

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

        return best_bid

    def get_best_ask(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        best_ask, best_ask_amount = list(market_asks.items())[0]

        return best_ask
    
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

    def update_diff_history(self, mid_prices_history):
        for symbol in self.PRODUCTS:
            if len(mid_prices_history[symbol]) >=2:
                diff = mid_prices_history[symbol][-1] - mid_prices_history[symbol][-2]
                
                self.diff_history[symbol].append(diff)

            while len(self.diff_history[symbol]) > 10:
                    self.diff_history[symbol].pop(0)


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
    
    def amethysts_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of amethysts.
        """
        orders = []

        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = (self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts) 
        ask_volume = (- self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts)

        """
        if bid_volume != 0:
            bid_volume = (bid_volume - 1) / 3
        else:
            bid_volume = 0
        
        if ask_volume != 0:
            ask_volume = (ask_volume + 1) / 3
        else:
            ask_volume = 0
        """
            
        """
        orders.append(Order('AMETHYSTS', 9996, bid_volume))
        orders.append(Order('AMETHYSTS', 1004, ask_volume))
        """

        best_bid, best_bid_volume = self.get_best_bid('AMETHYSTS', state)
        best_ask, best_ask_volume = self.get_best_ask('AMETHYSTS', state)
        mid_price = self.get_mid_price('AMETHYSTS', state)

        spread = best_ask - best_bid

        ema = self.ema_prices['AMETHYSTS']
        last_price = self.get_last_price('AMETHYSTS', state.own_trades, state.market_trades)

        print('EMA: ', ema)
        print('Last Price:', last_price)

        if bid_volume > best_bid_volume:
            second_bid_volume = bid_volume - best_bid_volume
            bid_volume = best_bid_volume
            orders.append(Order('AMETHYSTS', 9998, int(math.floor((bid_volume)))))
            orders.append(Order('AMETHYSTS', 9997, int(math.floor((second_bid_volume)))))
        else:
            orders.append(Order('AMETHYSTS', 9998, int(math.floor((bid_volume)))))
    
        if ask_volume < -best_ask_volume:
            second_ask_volume = ask_volume + best_ask_volume
            ask_volume = -best_ask_volume
            orders.append(Order('AMETHYSTS', 10002, int(math.floor((ask_volume))))) 
            orders.append(Order('AMETHYSTS', 10003, int(math.floor((second_ask_volume))))) 
        else:
            orders.append(Order('AMETHYSTS', 10002, int(math.floor((ask_volume)))))
        
        
        
       
        
        """
        if position_amethysts == 0:
            # Not long nor short
            orders.append(Order('AMETHYSTS', max(9995, math.floor(self.ema_prices['AMETHYSTS'] - 2)), bid_volume))
            orders.append(Order('AMETHYSTS', min(10005,math.ceil(self.ema_prices['AMETHYSTS'] + 2)), ask_volume))

        if position_amethysts > 0:
            # Long position
            orders.append(Order('AMETHYSTS', max(9995, math.floor(self.ema_prices['AMETHYSTS'] - 2)), bid_volume))
            orders.append(Order('AMETHYSTS', min(10005,math.ceil(self.ema_prices['AMETHYSTS'] + 1)), ask_volume))
            #orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] - 3), bid_volume))
            #orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 2), ask_volume))

        if position_amethysts < 0: 
            # Short position
            orders.append(Order('AMETHYSTS', max(9995, math.floor(self.ema_prices['AMETHYSTS'] -1)), bid_volume))
            orders.append(Order('AMETHYSTS', min(10005,math.ceil(self.ema_prices['AMETHYSTS'] + 2)), ask_volume))
            #orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] - 2), bid_volume))
            #orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 3), ask_volume))
        """
        #print(orders)

        return orders

    def amethysts_strategy2(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of amethysts.
        """
        orders = []

        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        ask_volume = - self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts

        mid_price = self.get_mid_price('AMETHYSTS', state)

        #bid_price, ask_price = self.get_order_book('AMETHYSTS', state)
        bid_price = self.get_best_bid('AMETHYSTS', state)
        ask_price = self.get_best_ask('AMETHYSTS', state)

        if position_amethysts == 0:
            # Not long nor short
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS']  - 1), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS']  + 1), ask_volume))
        
        elif position_amethysts > 15:
            # Long position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS']  - 3), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] +1), int(math.ceil((ask_volume/2)))))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] +2), int(math.floor((ask_volume/2)))))

        elif position_amethysts < -15:
            # Short position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -2), int(math.ceil((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -1), int(math.floor((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS']  + 3), ask_volume))

        elif position_amethysts > 10:
            # Long position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS']  - 3), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] +1), int(math.ceil((ask_volume/2)))))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] +2), int(math.floor((ask_volume/2)))))

        elif position_amethysts < -10:
            # Short position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -2), int(math.ceil((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -1), int(math.floor((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS']  + 3), ask_volume))

        elif position_amethysts > 0:
            # Long position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS']  - 2), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 1), int(math.ceil((ask_volume/2)))))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 2), int(math.floor((ask_volume/2)))))

        elif position_amethysts < 0:
            # Short position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -2), int(math.ceil((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -1), int(math.floor((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS']  + 2), ask_volume))


        #print(orders)

        return orders
    
    def amethysts_strategy3(self, state : TradingState) -> List[Order]:
        """
        Buying and Selling based on last trade price vs mean price
        """
        orders = []
        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        ask_volume = - self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        last_price = self.get_last_price('AMETHYSTS', state.own_trades, state.market_trades)

        if len(self.prices_history['AMETHYSTS']) > 5:
            upper_band = self.calculate_sma('AMETHYSTS', 10) + 0.3 * self.calculate_standard_deviation(self.prices_history['AMETHYSTS'][-5:])
            lower_band = self.calculate_sma('AMETHYSTS', 10) - 0.3 * self.calculate_standard_deviation(self.prices_history['AMETHYSTS'][-5:])
            
            if last_price > upper_band:
                orders.append(Order('AMETHYSTS', math.floor(last_price - 1), ask_volume))
            elif last_price < lower_band:
                orders.append(Order('AMETHYSTS', math.ceil(last_price + 1), bid_volume))
        else:
            if last_price > 10000:
                orders.append(Order('AMETHYSTS', math.floor(last_price - 1), ask_volume))
            elif last_price < 10000:
                orders.append(Order('AMETHYSTS', math.ceil(last_price + 1), bid_volume))
            
            #print(orders, upper_band, lower_band, last_price)

        return orders


    def amethysts_strategy4(self, state : TradingState) -> List[Order]:
        """
        Buying and Selling based on last trade price vs mean price (ceiling floor version)
        """
        orders = []
        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        ask_volume = - self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        last_price = self.get_last_price('AMETHYSTS', state.own_trades, state.market_trades)

        if last_price > 10000:
            orders.append(Order('AMETHYSTS', math.floor(last_price), ask_volume))
            if len(self.prices_history["AMETHYSTS"]) > 5:

                if len([i for i in self.prices_history["AMETHYSTS"][-20:] if i > 10000])>18:
                    orders.append(Order('AMETHYSTS', math.ceil(10000), math.floor(bid_volume*0.3)))
                else:
                    for number in reversed(self.prices_history["AMETHYSTS"]):
                        if number < 10000:
                            orders.append(Order('AMETHYSTS', math.floor(number), bid_volume))
                            break

        elif last_price < 10000:
            orders.append(Order('AMETHYSTS', math.ceil(last_price), bid_volume))

            if len(self.prices_history["AMETHYSTS"]) > 5:
                if len([i for i in self.prices_history["AMETHYSTS"][-20:] if i < 10000])>18:
                    orders.append(Order('AMETHYSTS', math.floor(10000), math.floor(ask_volume*0.3)))
                else:
                    for number in reversed(self.prices_history["AMETHYSTS"]):
                        if number > 10000:
                            orders.append(Order('AMETHYSTS', math.floor(number), ask_volume))
                            break
        
        #print(orders, last_price)
        
        return orders

    def amethysts_strategy3(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of amethysts.
        """
        orders = []

        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = (self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts ) / 10 -1
        ask_volume = (- self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts) / 10 +1

        """
        if bid_volume != 0:
            bid_volume = (bid_volume - 1) / 3
        else:
            bid_volume = 0
        
        if ask_volume != 0:
            ask_volume = (ask_volume + 1) / 3
        else:
            ask_volume = 0
        """
            
        """
        orders.append(Order('AMETHYSTS', 9996, bid_volume))
        orders.append(Order('AMETHYSTS', 1004, ask_volume))
        """

        best_bid = self.get_best_bid('AMETHYSTS', state)
        best_ask = self.get_best_ask('AMETHYSTS', state)
        mid_price = int(math.floor(self.get_mid_price('AMETHYSTS', state)))

        spread = best_ask - best_bid

        ema = self.ema_prices['AMETHYSTS']
        last_price = self.get_last_price('AMETHYSTS', state.own_trades, state.market_trades)

        vwap = self.calculate_vwap('AMETHYSTS', state.own_trades, state.market_trades)

        print('EMA: ', ema)
        print('Last Price:', last_price)

        imbalance = self.calculate_order_book_imbalance('AMETHYSTS', state)
        
        #trade at 9996, 9998 and 1002, 1004
        

        if last_price >= 1003:
            orders.append(Order('AMETHYSTS', 9998, math.floor(bid_volume * 7)))
            orders.append(Order('AMETHYSTS', 10002, math.floor(ask_volume * 7)))

            orders.append(Order('AMETHYSTS', 9997, math.floor(bid_volume * 2)))
            orders.append(Order('AMETHYSTS', 10003, math.floor(ask_volume * 3)))

            orders.append(Order('AMETHYSTS', 10004, -1))

        elif last_price <= 9997:
            orders.append(Order('AMETHYSTS', 9998, math.floor(bid_volume * 7)))
            orders.append(Order('AMETHYSTS', 10002, math.floor(ask_volume * 7)))

            orders.append(Order('AMETHYSTS', 9997, math.floor(bid_volume * 2)))
            orders.append(Order('AMETHYSTS', 10003, math.floor(ask_volume * 3)))

            orders.append(Order('AMETHYSTS', 9996, 1))

        else:
            orders.append(Order('AMETHYSTS', 9998, math.floor(bid_volume * 9)))
            orders.append(Order('AMETHYSTS', 10002, math.floor(ask_volume * 9)))

            orders.append(Order('AMETHYSTS', 9997, math.floor(bid_volume * 1)))
            orders.append(Order('AMETHYSTS', 10003, math.floor(ask_volume *1 )))

        return orders

    def amethysts_strategy4(self, state : TradingState) -> List[Order]:
        """
        Buying and Selling based on last trade price vs mean price (ceiling floor version)
        """
        orders = []
        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        ask_volume = - self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        last_price = self.get_last_price('AMETHYSTS', state.own_trades, state.market_trades)

        ema = self.ema_prices['AMETHYSTS']

        if True:
            if True:
                spread = 1
                open_spread = 3
                position_limit = 20
                position_spread = 15
                current_position = state.position.get("AMETHYSTS",0)
                best_ask = 0
                best_bid = 0
                
                order_depth_ame: OrderDepth = state.order_depths["AMETHYSTS"]
                
                if len(order_depth_ame.sell_orders) > 0:
                    best_ask = min(order_depth_ame.sell_orders.keys())

                    if best_ask <= 10000-spread:
                        best_ask_volume = order_depth_ame.sell_orders[best_ask]
                    else:
                        best_ask_volume = 0
                else:
                    best_ask_volume = 0

                if len(order_depth_ame.buy_orders) > 0:
                    best_bid = max(order_depth_ame.buy_orders.keys())

                    if best_bid >= 10000+spread:
                        best_bid_volume = order_depth_ame.buy_orders[best_bid]
                    else:
                        best_bid_volume = 0 
                else:
                    best_bid_volume = 0

                if current_position - best_ask_volume > position_limit:
                    best_ask_volume = current_position - position_limit
                    open_ask_volume = 0
                else:
                    open_ask_volume = current_position - position_spread - best_ask_volume

                if current_position - best_bid_volume < -position_limit:
                    best_bid_volume = current_position + position_limit
                    open_bid_volume = 0
                else:
                    open_bid_volume = current_position + position_spread - best_bid_volume

                if -open_ask_volume < 0:
                    open_ask_volume = 0         
                if open_bid_volume < 0:
                    open_bid_volume = 0

                if best_ask == 10000-open_spread and -best_ask_volume > 0:
                    orders.append(Order("AMETHYSTS", 10000-open_spread, -best_ask_volume-open_ask_volume))
                else:
                    if -best_ask_volume > 0:
                        orders.append(Order("AMETHYSTS", best_ask, -best_ask_volume))
                    if -open_ask_volume > 0:
                        orders.append(Order("AMETHYSTS", 10000-open_spread, -open_ask_volume))

                if best_bid == 10000+open_spread and best_bid_volume > 0:
                    orders.append(Order("AMETHYSTS", 10000+open_spread, -best_bid_volume-open_bid_volume))
                else:
                    if best_bid_volume > 0:
                        orders.append(Order("AMETHYSTS", best_bid, -best_bid_volume))
                    if open_bid_volume > 0:
                        orders.append(Order("AMETHYSTS", 10000+open_spread, -open_bid_volume))
       

        
        print(orders, last_price)
        
        return orders

    def starfruit_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of starfruit.
        """

        orders = []
    
        position_starfruit = self.get_position('STARFRUIT', state)

        bid_volume = self.POSITION_LIMITS['STARFRUIT'] - position_starfruit
        ask_volume = - self.POSITION_LIMITS['STARFRUIT'] - position_starfruit

        best_bid = self.get_best_bid('STARFRUIT', state)
        best_ask = self.get_best_ask('STARFRUIT', state)
        mid_price = self.get_mid_price('STARFRUIT', state)
        spread = best_ask - best_bid

        last_price = self.get_last_price('STARFRUIT', state.own_trades, state.market_trades)
        #print('Last Price:', last_price)
        sma_20 = self.calculate_sma('STARFRUIT', 20)
        #print('SMA:', sma)
        ema_8 = self.calculate_ema('STARFRUIT', 8)
        ema_21 = self.calculate_ema('STARFRUIT', 21)
        #print('EMA:', ema)
        vwap = self.calculate_vwap('STARFRUIT', state.own_trades, state.market_trades)
        #print('VWAP:', vwap)

        lags = self.mid_prices_history['STARFRUIT'][-5:]

        #lags = self.prices_history['STARFRUIT'][-4:]
        #coef = [0.3176191343791975,  0.22955395157579261 ,  0.24255751299309652,  0.20773853347672797]
        #intercept = 12.56819767718207

        coef = [0.28730546050106137,  0.19457488964477182 ,  0.20933726944154457,  0.16398511050940906, 0.14295445771547163]
        intercept = 9.132055913210934

        forecasted_price = intercept
        for i, val in enumerate(lags):
            forecasted_price += val * coef[i]

        """
        if last_price > ema:
            orders.append(Order('STARFRUIT', math.floor(mid_price - 2.5), bid_volume))
            orders.append(Order('STARFRUIT', math.floor(mid_price - 2.5), bid_volume))
        elif last_price < ema:
            orders.append(Order('STARFRUIT', math.ceil(mid_price + 2.5), ask_volume))
        """

        sd = self.calculate_standard_deviation(self.prices_history['STARFRUIT'][:])
        upper_limit = sma_20 + 1.5 * sd
        lower_limit = sma_20 - 1.5 * sd

        
        if last_price < lower_limit:
            orders.append(Order('STARFRUIT', math.floor(mid_price-2), bid_volume))
            #orders.append(Order('STARFRUIT', math.floor(mid_price+3), int(math.floor(ask_volume/2))))
            #orders.append(Order('STARFRUIT', math.floor(mid_price+4), int(math.ceil(ask_volume/2))))
            
        elif last_price > upper_limit:
            orders.append(Order('STARFRUIT', math.ceil(mid_price+2), ask_volume))
            #orders.append(Order('STARFRUIT', math.ceil(mid_price-3), int(math.floor(bid_volume/2))))
            #orders.append(Order('STARFRUIT', math.ceil(mid_price-4), int(math.ceil(bid_volume/2))))
        
        
        if last_price > ema_8:
            orders.append(Order('STARFRUIT', math.floor(best_bid+1), bid_volume))
            #orders.append(Order('STARFRUIT', math.floor(mid_price+2), int(math.floor(ask_volume/2))))
            #orders.append(Order('STARFRUIT', math.floor(mid_price+3), int(math.ceil(ask_volume/2))))
        elif last_price < ema_8:
            orders.append(Order('STARFRUIT', math.ceil(best_ask - 1 ), ask_volume))
            #orders.append(Order('STARFRUIT', math.ceil(mid_price-2), int(math.floor(bid_volume/2))))
            #orders.append(Order('STARFRUIT', math.ceil(mid_price-3), int(math.ceil(bid_volume/2))))
        

        return orders

    def starfruit_strategy2(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of starfruit.
        """

        orders = []
    
        position_starfruit = self.get_position('STARFRUIT', state)

        bid_volume = self.POSITION_LIMITS['STARFRUIT'] - position_starfruit
        ask_volume = - self.POSITION_LIMITS['STARFRUIT'] - position_starfruit

        best_bid = self.get_best_bid('STARFRUIT', state)
        best_ask = self.get_best_ask('STARFRUIT', state)
        mid_price = self.get_mid_price('STARFRUIT', state)

        last_price = self.get_last_price('STARFRUIT', state.own_trades, state.market_trades)
        #print('Last Price:', last_price)

        lags = self.prices_history['STARFRUIT'][-5:]

        coef = [0.28730546050106137,  0.19457488964477182 ,  0.20933726944154457,  0.16398511050940906, 0.14295445771547163]
        intercept = 9.132055913210934

        forecasted_price = intercept
        for i, val in enumerate(lags):
            forecasted_price += val * coef[i]
        
        #print('Forecasted Price:',forecasted_price)
         
        

        print('Last Price:',last_price)
        print('Forecasted Price:',forecasted_price)
        
    
        if forecasted_price > best_bid+1:
            orders.append(Order('STARFRUIT', math.floor(best_bid+1), bid_volume))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+2), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+3), int(math.ceil(ask_volume/2))))
        elif forecasted_price < best_ask-1:
            orders.append(Order('STARFRUIT', math.floor(best_ask-1), ask_volume))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-2), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-3), int(math.ceil(bid_volume/2))))

        return orders

    def starfruit_strategy3(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of starfruit.
        """

        orders = []
    
        position_starfruit = self.get_position('STARFRUIT', state)

        bid_volume = self.POSITION_LIMITS['STARFRUIT'] - position_starfruit
        ask_volume = - self.POSITION_LIMITS['STARFRUIT'] - position_starfruit

        best_bid = self.get_best_bid('STARFRUIT', state)
        best_ask = self.get_best_ask('STARFRUIT', state)
        mid_price = self.get_mid_price('STARFRUIT', state)
        spread = best_ask - best_bid

        last_price = self.get_last_price('STARFRUIT', state.own_trades, state.market_trades)
        #print('Last Price:', last_price)
        sma_20 = self.calculate_sma('STARFRUIT', 20)
        #print('SMA:', sma)
        ema_8 = self.calculate_ema('STARFRUIT', 8)
        ema_21 = self.calculate_ema('STARFRUIT', 21)
        #print('EMA:', ema)
        vwap = self.calculate_vwap('STARFRUIT', state.own_trades, state.market_trades)
        #print('VWAP:', vwap)

        lags = self.mid_prices_history['STARFRUIT'][-4:]
        log_returns = [np.log(lags[i] / lags[i-1]) for i in range(1, len(lags))]

        coef = [-0.1474520820077168,  -0.6965073247325156,  -0.13187759816668118]
        #coef = list(reversed(coef))
        intercept = 0.0003543517501174694

        predicted_log_return = intercept
        forecasted_price = self.mid_prices_history['STARFRUIT'][-1]
        for i, val in enumerate(log_returns):
            predicted_log_return += val * coef[i]
        
        forecasted_price = forecasted_price * np.exp(predicted_log_return)

        print('Last Price: ', self.mid_prices_history['STARFRUIT'][-1])
        print('Forecasted Price: ',forecasted_price)

        """
        if last_price > ema:
            orders.append(Order('STARFRUIT', math.floor(mid_price - 2.5), bid_volume))
            orders.append(Order('STARFRUIT', math.floor(mid_price - 2.5), bid_volume))
        elif last_price < ema:
            orders.append(Order('STARFRUIT', math.ceil(mid_price + 2.5), ask_volume))
        """

        sd = self.calculate_standard_deviation(self.prices_history['STARFRUIT'][:])
        upper_limit = sma_20 + 2 * sd
        lower_limit = sma_20 - 2 * sd

        
        if forecasted_price > best_bid+1:
            orders.append(Order('STARFRUIT', math.floor(best_bid+1), bid_volume))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+2), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+1), int(math.ceil(ask_volume/2))))
        elif forecasted_price < best_ask-1:
            orders.append(Order('STARFRUIT', math.floor(best_ask-1), ask_volume))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-2), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-1), int(math.ceil(bid_volume/2))))

        return orders

    def starfruit_strategy4(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of starfruit.
        """

        orders = []
    
        position_starfruit = self.get_position('STARFRUIT', state)

        bid_volume = self.POSITION_LIMITS['STARFRUIT'] - position_starfruit
        ask_volume = - self.POSITION_LIMITS['STARFRUIT'] - position_starfruit

        best_bid = self.get_best_bid('STARFRUIT', state)
        best_ask = self.get_best_ask('STARFRUIT', state)
        mid_price = self.get_mid_price('STARFRUIT', state)
        spread = (best_ask - best_bid) / 2

        ema_8 = self.calculate_ema('STARFRUIT', 8)
        ema_13 = self.calculate_ema('STARFRUIT', 13)
        ema_21 = self.calculate_ema('STARFRUIT', 21)
        vwap = self.calculate_vwap('STARFRUIT', state.own_trades, state.market_trades)

        # Calculate daily price changes
        price_changes = np.diff(self.prices_history['STARFRUIT'])  # This will give you the difference between each price and the previous one

        # Set the number of days for the EMA period
        d = 20  # You can adjust this value based on your needs

        # Calculate EMA of price changes
        ema = [price_changes[0]]  # Initialize EMA with the first price change
        for change in price_changes[1:]:
            ema.append((change * (2 / (1 + d))) + ema[-1] * (1 - (2 / (1 + d))))
        volatility = ema[-1]

        if len(self.diff_history['STARFRUIT']) >= 6:
            AR_L1 = self.diff_history['STARFRUIT'][-1]
            AR_L2 = self.diff_history['STARFRUIT'][-2]
            AR_L3 = self.diff_history['STARFRUIT'][-3]
            AR_L4 = self.diff_history['STARFRUIT'][-4]
            AR_L5 = self.diff_history['STARFRUIT'][-5]
            AR_L6 = self.diff_history['STARFRUIT'][-6]
        
        if len(self.forecasted_diff_history['STARFRUIT']) > 0:
            forecasted_error = self.forecasted_diff_history['STARFRUIT'][-1] - self.diff_history['STARFRUIT'][-1]
            self.errors_history['STARFRUIT'].append(forecasted_error)

        if len(self.errors_history['STARFRUIT']) < 2:
            self.errors_history['STARFRUIT'].extend([1.682936, -2.797327, -0.480615])
            #self.errors_history['STARFRUIT'].append(-0.491629)
        else:
            MA_L1 = self.errors_history['STARFRUIT'][-1]
            MA_L2 = self.errors_history['STARFRUIT'][-2]
            MA_L3 = self.errors_history['STARFRUIT'][-3]
        

        #forecasted_diff = 0.0017 + (AR_L1 * -0.5872) + (AR_L2 * 0.0029) + (AR_L3 * 0.0016) + (MA_L1 * -0.1208) + (MA_L2 * -0.4189)
        forecasted_diff = (AR_L1 * -1.1102) + (AR_L2 * -0.7276) + (AR_L3 * -0.0854) + (AR_L4 * -0.0674)
        + (AR_L5 * -0.0437) + (AR_L6 * -0.0176)+ (MA_L1 *  0.4021) + (MA_L2 * -0.0587) + (MA_L3 * -0.4357)

        self.forecasted_diff_history['STARFRUIT'].append(forecasted_diff)
        
        forecasted_price = mid_price + forecasted_diff  

        last_price = self.get_last_price('STARFRUIT', state.own_trades, state.market_trades)  

        print('Last Mid Price:', self.mid_prices_history['STARFRUIT'][-1])
        print('Forecasted Price:', forecasted_price)

        """
        if self.current_signal['STARFRUIT'] == 'Triple MA':
            if forecasted_price > mid_price and position_starfruit > 0:
                orders.append(Order('STARFRUIT', math.floor(forecasted_price + 1), -position_starfruit))
            elif forecasted_price < mid_price and position_starfruit < 0:
                orders.append(Order('STARFRUIT', math.floor(forecasted_price - 1), -position_starfruit))
            elif position_starfruit == 0:
                self.current_signal['STARFRUIT'] = 'None'
        
        if last_price > ema_8 and ema_8 > ema_13 and ema_13 > ema_21:
            orders.append(Order('STARFRUIT', math.floor(mid_price - 2), bid_volume))
            self.current_signal['STARFRUIT'] = 'Triple MA'

        if last_price < ema_8 and ema_8 < ema_13 and ema_13 < ema_21:
            orders.append(Order('STARFRUIT', math.floor(mid_price + 2), ask_volume))
            self.current_signal['STARFRUIT'] = 'Triple MA'
        """
        """
        if forecasted_price > best_bid+1:
            orders.append(Order('STARFRUIT', math.floor(best_bid + 1), min(2, bid_volume)))
        elif forecasted_price < best_ask -1:
            orders.append(Order('STARFRUIT', math.ceil(best_ask -1), max(-2, ask_volume)))
        """
        
        """
        if forecasted_price > mid_price -1:
            orders.append(Order('STARFRUIT', math.floor(mid_price-1), bid_volume))
            #orders.append(Order('STARFRUIT', math.floor(forecasted_price + spread), ask_volume))
        elif forecasted_price < mid_price + 1:    
            orders.append(Order('STARFRUIT', math.ceil(mid_price + 1), ask_volume))
            #orders.append(Order('STARFRUIT', math.ceil(forecasted_price - spread), bid_volume))
        """

        if position_starfruit == 0:
            # Not long nor short
            orders.append(Order('STARFRUIT', math.floor(vwap  - 1), bid_volume))
            orders.append(Order('STARFRUIT', math.ceil(vwap  + 1), ask_volume))
        
        elif position_starfruit > 15:
            # Long position
            orders.append(Order('STARFRUIT', math.floor(vwap  - 3), bid_volume))
            orders.append(Order('STARFRUIT', math.ceil(vwap +2), int(math.ceil((ask_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(vwap +1), int(math.ceil((ask_volume/2)))))

        elif position_starfruit < -15:
            # Short position
            orders.append(Order('STARFRUIT', math.floor(vwap -1), int(math.ceil((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.floor(vwap -2), int(math.floor((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(vwap  + 3), ask_volume))

        elif position_starfruit > 10:
            # Long position
            orders.append(Order('STARFRUIT', math.floor(vwap  - 3), bid_volume))
            orders.append(Order('STARFRUIT', math.ceil(vwap +1), int(math.ceil((ask_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(vwap +2), int(math.floor((ask_volume/2)))))

        elif position_starfruit < -10:
            # Short position
            orders.append(Order('STARFRUIT', math.floor(vwap -2), int(math.ceil((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.floor(vwap -1), int(math.floor((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(vwap  + 3), ask_volume))

        elif position_starfruit > 0:
            # Long position
            orders.append(Order('STARFRUIT', math.floor(vwap  - 2), bid_volume))
            orders.append(Order('STARFRUIT', math.ceil(vwap + 1), int(math.ceil((ask_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(vwap + 2), int(math.floor((ask_volume/2)))))

        elif position_starfruit < 0:
            # Short position
            orders.append(Order('STARFRUIT', math.floor(vwap -2), int(math.ceil((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.floor(vwap -1), int(math.floor((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(vwap  + 2), ask_volume))
        

        return orders
    
    def starfruit_strategy5(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of starfruit.
        """

        orders = []
    
        position_starfruit = self.get_position('STARFRUIT', state)

        bid_volume = self.POSITION_LIMITS['STARFRUIT'] - position_starfruit
        ask_volume = - self.POSITION_LIMITS['STARFRUIT'] - position_starfruit

        best_bid = self.get_best_bid('STARFRUIT', state)
        best_ask = self.get_best_ask('STARFRUIT', state)
        mid_price = self.get_mid_price('STARFRUIT', state)
        spread = (best_ask - best_bid) / 2

        sma_81 = self.calculate_sma('STARFRUIT', 81)
        sma_50 = self.calculate_sma('STARFRUIT', 50)
        sma_21 = self.calculate_sma('STARFRUIT', 21)

        # Calculate daily price changes
        price_changes = np.diff(self.prices_history['STARFRUIT'])  # This will give you the difference between each price and the previous one

        # Set the number of days for the EMA period
        d = 20  # You can adjust this value based on your needs

        # Calculate EMA of price changes
        ema = [price_changes[0]]  # Initialize EMA with the first price change
        for change in price_changes[1:]:
            ema.append((change * (2 / (1 + d))) + ema[-1] * (1 - (2 / (1 + d))))
        volatility = ema[-1]

        if len(self.diff_history['STARFRUIT']) >= 6:
            AR_L1 = self.diff_history['STARFRUIT'][-1]
            AR_L2 = self.diff_history['STARFRUIT'][-2]
            AR_L3 = self.diff_history['STARFRUIT'][-3]
            AR_L4 = self.diff_history['STARFRUIT'][-4]
            AR_L5 = self.diff_history['STARFRUIT'][-5]
            AR_L6 = self.diff_history['STARFRUIT'][-6]
        
        if len(self.forecasted_diff_history['STARFRUIT']) > 0:
            forecasted_error = self.forecasted_diff_history['STARFRUIT'][-1] - self.diff_history['STARFRUIT'][-1]
            self.errors_history['STARFRUIT'].append(forecasted_error)

        if len(self.errors_history['STARFRUIT']) < 2:
            #use this!
            self.errors_history['STARFRUIT'].extend([1.682936, -2.797327, -0.480615])
            
            #self.errors_history['STARFRUIT'].append(-0.491629)
            #self.errors_history['STARFRUIT'].extend([1.670939, -2.786309, -0.488883])
            
            #self.errors_history['STARFRUIT'].extend([-0.556338, -1.392315, -1.969127])
        else:
            MA_L1 = self.errors_history['STARFRUIT'][-1]
            MA_L2 = self.errors_history['STARFRUIT'][-2]
            MA_L3 = self.errors_history['STARFRUIT'][-3]
        

        #forecasted_diff = 0.0017 + (AR_L1 * -0.5872) + (AR_L2 * 0.0029) + (AR_L3 * 0.0016) + (MA_L1 * -0.1208) + (MA_L2 * -0.4189)
        
        #use this!
        forecasted_diff = (AR_L1 * -1.1102) + (AR_L2 * -0.7276) + (AR_L3 * -0.0854) + (AR_L4 * -0.0674)
        + (AR_L5 * -0.0437) + (AR_L6 * -0.0176)+ (MA_L1 *  0.4021) + (MA_L2 * -0.0587) + (MA_L3 * -0.4357)

        #forecasted_diff = (AR_L1 * -1.5026) + (AR_L2 * -0.7053) + (AR_L3 * -0.0897) + (AR_L4 * -0.0717)
        #+ (AR_L5 * -0.0472) + (AR_L6 * -0.0196)+ (MA_L1 *  0.7948) + (MA_L2 * -0.3562) + (MA_L3 * -0.4119)

        #forecasted_diff = (AR_L1 * -1.1083) + (AR_L2 * -0.8814) + (AR_L3 * -0.0943) + (AR_L4 * -0.0778)
        #+ (AR_L5 * -0.0468) + (AR_L6 * -0.0207)+ (MA_L1 *  0.3955) + (MA_L2 * 0.0891) + (MA_L3 * -0.5379)

        self.forecasted_diff_history['STARFRUIT'].append(forecasted_diff)
        
        forecasted_price = mid_price + forecasted_diff  

        last_price = self.get_last_price('STARFRUIT', state.own_trades, state.market_trades)  

        print('Last Mid Price:', self.mid_prices_history['STARFRUIT'][-1])
        print('Forecasted Price:', forecasted_price)
        """
        if state.timestamp > 8100:
            if position_starfruit == 0:
                self.current_signal['STARFRUIT'] == 'None'
            if position_starfruit != 0 and self.current_signal['STARFRUIT'] == 'MA':
                if position_starfruit > 0 and sma_21 < sma_50:
                    orders.append(Order('STARFRUIT', math.floor(mid_price + 1), -position_starfruit))
                    self.current_signal['STARFRUIT'] == 'None'
                elif position_starfruit < 0 and sma_21 > sma_50:
                    orders.append(Order('STARFRUIT', math.ceil(mid_price - 1), -position_starfruit))
                    self.current_signal['STARFRUIT'] == 'None'
   
            elif sma_21 > sma_50 and sma_50 > sma_81 and best_bid <= sma_81:
                orders.append(Order('STARFRUIT', math.floor(best_bid + 1), bid_volume))
                self.current_signal['STARFRUIT'] = 'MA'

            elif sma_21 < sma_50 and sma_50 < sma_81 and best_ask >= sma_81:
                orders.append(Order('STARFRUIT', math.floor(best_ask-1), ask_volume))
                self.current_signal['STARFRUIT'] = 'MA'
        """    
        #change conditions based on position 3, 4

 
        
        if forecasted_price > best_bid+2:
            orders.append(Order('STARFRUIT', math.floor(best_bid+1), bid_volume))
            #orders.append(Order('STARFRUIT', math.floor(best_bid+2), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+3), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+4), int(math.ceil(ask_volume/2))))
        elif forecasted_price < best_ask-2:
            orders.append(Order('STARFRUIT', math.floor(best_ask-1), ask_volume))
            #orders.append(Order('STARFRUIT', math.floor(best_ask-2), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-3), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-4), int(math.ceil(bid_volume/2))))
        

        """
        if position_starfruit == 0:
            # Not long nor short
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT']  - 1), bid_volume))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT']  + 1), ask_volume))
        
        elif position_starfruit > 15:
            # Long position
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT']  - 3), bid_volume))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT'] +1), int(math.ceil((ask_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT'] +1), int(math.ceil((ask_volume/2)))))

        elif position_starfruit < -15:
            # Short position
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT'] -1), int(math.ceil((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT'] -1), int(math.floor((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT']  + 3), ask_volume))

        elif position_starfruit > 10:
            # Long position
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT']  - 3), bid_volume))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT'] +1), int(math.ceil((ask_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT'] +2), int(math.floor((ask_volume/2)))))

        elif position_starfruit < -10:
            # Short position
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT'] -2), int(math.ceil((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT'] -1), int(math.floor((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT']  + 3), ask_volume))

        elif position_starfruit > 0:
            # Long position
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT']  - 2), bid_volume))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT'] + 1), int(math.ceil((ask_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT'] + 2), int(math.floor((ask_volume/2)))))

        elif position_starfruit < 0:
            # Short position
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT'] -2), int(math.ceil((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.floor(self.ema_prices['STARFRUIT'] -1), int(math.floor((bid_volume/2)))))
            orders.append(Order('STARFRUIT', math.ceil(self.ema_prices['STARFRUIT']  + 2), ask_volume))
        """

        return orders
    

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS' : [], 'STARFRUIT' : []}

        self.update_ema_prices(state)

        # PRICE HISTORY
        self.update_prices_history(state.own_trades, state.market_trades)
        self.update_mid_prices_history(state)
        self.update_diff_history(self.mid_prices_history)
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
            while len(self.pnl_tracker[product]) > 10:
                self.pnl_tracker[product].pop(0)
            while len(self.forecasted_diff_history[product]) > 10:
                self.forecasted_diff_history[product].pop(0)
            while len(self.errors_history[product]) > 10:
                self.errors_history[product].pop(0)
           


        # AMETHYSTS STRATEGY
        
        try:
            result['AMETHYSTS'] = self.amethysts_strategy4(state)
        except Exception as e:
            print("Error in AMETHYSTS strategy")
            print(e)
        
        # STARFRUIT STRATEGY
        try:
            result['STARFRUIT'] = self.starfruit_strategy5(state)
        except Exception as e:
            print("Error in STARFRUIT strategy")
            print(e)
        
                
        traderData = "SAMPLE" 
        
		# Sample conversion request. Check more details below. 
        conversions = 0
        return result, conversions, traderData