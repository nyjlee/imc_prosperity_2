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

    spreads_roses = []

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
            #bid2, bid2_amount = float('-inf'), 0
            bid2, bid2_amount = 0, 0

        return bid2, bid2_amount
    
    def get_bid3(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 3:
            bid3, bid3_amount = list(market_bids.items())[2]
        else:
            #bid3, bid3_amount = float('-inf'), 0
            bid3, bid3_amount = 0, 0

        return bid3, bid3_amount
    
    def get_ask2(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        if len(market_asks) == 2:
            ask2, ask2_amount = list(market_asks.items())[1]
        else:
            #ask2, ask2_amount = float('inf'), 0
            ask2, ask2_amount = 500000, 0

        return ask2, ask2_amount
    
    def get_ask3(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        if len(market_asks) == 3:
            ask3, ask3_amount = list(market_asks.items())[2]
        else:
            #ask3, ask3_amount = float('inf'), 0
            ask3, ask3_amount = 500000, 0

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

        #### SECOND BIDS AND ASKS ####
        roses_bid_2, roses_bid_vol_2 = self.get_bid2('ROSES', state)
        roses_ask_2, roses_ask_vol_2 = self.get_ask2('ROSES', state)


        #### THIRD BIDS AND ASKS ####
        roses_bid_3, roses_bid_vol_3 = self.get_bid3('ROSES', state)
        roses_ask_3, roses_ask_vol_3 = self.get_ask3('ROSES', state)


        spread_roses = roses_mid_price - 1.3427 * chocolates_mid_price

        print('SPREAD:', spread_roses)
        #print('SPREADS:', self.spreads_basket)
        #print('RATIO:', ratio)

        #if len(self.etf_prices) >= 1:
            #etf_return = basket_mid_price / self.etf_prices[-1] - 1
        #if len(self.nav_prices) >= 1:
            #nav_return = nav / self.nav_prices[-1] - 1


        #if len(self.spreads_basket) == 150:
        if len(self.spreads_roses) == 200:
            mean = statistics.mean(self.spreads_roses)
            std = statistics.stdev(self.spreads_roses)

            #mean = statistics.mean(self.ratios_basket)
            #std = statistics.stdev(self.ratios_basket)

            #mean_etf_return = statistics.mean(self.etf_returns)
            #std_etf_return = statistics.stdev(self.etf_returns)

            #mean_nav_return = statistics.mean(self.nav_returns)
            #std_nav_return = statistics.stdev(self.nav_returns)

            z_score = (spread_roses - mean) / std
            #z_score = (ratio - mean) / std
            print('Z-SCORE: ',z_score)

            
            if z_score > -0.5 and z_score < 0.5 and current_position_all != 0:
                if position_roses > 0: #EXIT LONG POSITION
                    position1 = math.ceil(position_roses/2)
                    position2 = math.floor(position_roses/2)
                    roses_orders.append(Order('ROSES',  int(math.floor(self.ema_prices['ROSES']+1)), -position1))
                    roses_orders.append(Order('ROSES',  int(math.floor(self.ema_prices['ROSES']+2)), -position2))
                    #roses_orders.append(Order('ROSES',  basket_bid, max(-position_roses, -basket_bid_vol)))
                    #if basket_bid_vol < position_roses:
                        #roses_orders.append(Order('ROSES',  basket_bid_2, max(-(position_roses-basket_bid_vol), -basket_bid_vol_2)))
                elif position_roses < 0: #EXIT SHORT POSITION
                    position1 = math.ceil(position_roses/2)
                    position2 = math.floor(position_roses/2)
                    roses_orders.append(Order('ROSES',  int(math.floor(self.ema_prices['ROSES']-1)), -position1))
                    roses_orders.append(Order('ROSES',  int(math.floor(self.ema_prices['ROSES']-2)), -position2))
                    #roses_orders.append(Order('ROSES', int(math.ceil(self.ema_prices['ROSES']-2)), -position_roses))
                    #if abs(basket_ask_vol) < position_roses:
                        #roses_orders.append(Order('GIFT_BASKET',  basket_ask_2, min(abs(position_basket-basket_ask_vol), abs(basket_ask_vol_2))))

                if position_chocolates > 0: #EXIT LONG POSITION
                    position1 = math.ceil(position_chocolates/2)
                    position2 = math.floor(position_chocolates/2)
                    chocolates_orders.append(Order('CHOCOLATE',  int(math.floor(self.ema_prices['CHOCOLATE']+1)), -position1))
                    chocolates_orders.append(Order('CHOCOLATE',  int(math.floor(self.ema_prices['CHOCOLATE']+2)), -position2))
                    #chocolates_orders.append(Order('CHOCOLATE',  basket_bid, max(-position_chocolates, -basket_bid_vol)))
                    #if basket_bid_vol < position_chocolates:
                        #chocolates_orders.append(Order('CHOCOLATE',  basket_bid_2, max(-(position_chocolates-basket_bid_vol), -basket_bid_vol_2)))
                elif position_chocolates < 0: #EXIT SHORT POSITION
                    position1 = math.ceil(position_chocolates/2)
                    position2 = math.floor(position_chocolates/2)
                    chocolates_orders.append(Order('CHOCOLATE',  int(math.floor(self.ema_prices['CHOCOLATE']-1)), -position1))
                    chocolates_orders.append(Order('CHOCOLATE',  int(math.floor(self.ema_prices['CHOCOLATE']-2)), -position2))
                    #chocolates_orders.append(Order('CHOCOLATE', int(math.ceil(self.ema_prices['CHOCOLATE']-2)), -position_chocolates))
                    #if abs(basket_ask_vol) < position_chocolates:
                        #chocolates_orders.append(Order('GIFT_BASKET',  basket_ask_2, min(abs(position_basket-basket_ask_vol), abs(basket_ask_vol_2))))
                """
                if position_chocolates > 0: #EXIT LONG POSITION
                    chocolates_orders.append(Order('CHOCOLATE', int(self.ema_prices['CHOCOLATE']+2), -position_chocolates))
                elif position_chocolates < 0: #EXIT SHORT POSITION
                    chocolates_orders.append(Order('CHOCOLATE', int(self.ema_prices['CHOCOLATE']-2), -position_chocolates))
                if position_strawberries > 0: #EXIT LONG POSITION
                    strawberries_orders.append(Order('STRAWBERRIES', int(self.ema_prices['STRAWBERRIES']+2), -position_strawberries))
                elif position_strawberries < 0: #EXIT SHORT POSITION
                    strawberries_orders.append(Order('STRAWBERRIES', int(self.ema_prices['STRAWBERRIES']-2), -position_strawberries))
                if position_roses > 0: #EXIT LONG POSITION
                    roses_orders.append(Order('ROSES', int(self.ema_prices['ROSES']+2), -position_roses))
                elif position_roses < 0: #EXIT SHORT POSITION
                    roses_orders.append(Order('ROSES', int(self.ema_prices['ROSES']-2), -position_roses))
                """
            
            elif z_score > 2.1: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                qt_basket = min(abs(sell_volume_basket), basket_bid_vol) 
                print(qt_basket)
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol)) 
                print(qt_chocolates)
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol)) 
                print(qt_strawberries)
                qt_roses = min(abs(sell_volume_roses), abs(roses_bid_vol)) 
                print(qt_roses)

                n_tradable = min(math.floor(qt_chocolates/1.3427), qt_roses)
                print(n_tradable)

                if n_tradable >= 0:
                    #basket_orders.append(Order('GIFT_BASKET', int(math.floor(self.ema_prices['GIFT_BASKET']+2)), - sell_volume_basket))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid, - qt_basket))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid, - n_tradable))
                    ##chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, math.floor(n_tradable * 1.3427)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, n_tradable * 6))
                    roses_orders.append(Order('ROSES', roses_bid, qt_roses))
                    if -qt_roses > sell_volume_roses:
                        roses_orders.append(Order('ROSES', roses_bid_2, max(-roses_bid_vol_2, sell_volume_roses+qt_roses)))
                    if -qt_roses - roses_bid_vol_2 >  sell_volume_roses:
                        roses_orders.append(Order('ROSES', roses_ask_3, max(-roses_ask_vol_3, sell_volume_roses+qt_roses+roses_bid_vol_2)))


            elif z_score < -2.1: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol)) 
                print(qt_basket)
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol) 
                print(qt_chocolates)
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol) 
                print(qt_strawberries)
                qt_roses = min(abs(buy_volume_roses), abs(roses_ask_vol)) 
                print(qt_roses)

                n_tradable = min(math.floor(qt_chocolates / 1.3427), qt_roses)
                print(n_tradable)
                if n_tradable >= 0:
                    #basket_orders.append(Order('GIFT_BASKET', int(math.ceil(self.ema_prices['GIFT_BASKET']-2)), buy_volume_basket))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask, qt_basket))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask, n_tradable))
                    ##chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, math.floor(-n_tradable * 1.3427)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid, -n_tradable * 6))
                    roses_orders.append(Order('ROSES', roses_ask, qt_roses))
                    if qt_roses < buy_volume_roses:
                        roses_orders.append(Order('ROSES', roses_ask_2, min(abs(roses_ask_vol_2), buy_volume_roses-qt_roses)))
                    if qt_roses+abs(roses_ask_vol_2) < buy_volume_roses:
                        roses_orders.append(Order('ROSES', roses_ask_3, min(abs(roses_ask_vol_3), buy_volume_roses-qt_roses-abs(roses_ask_vol_2))))


            
        

        self.spreads_roses.append(spread_roses)
        #self.etf_prices.append(basket_mid_price)
        #self.nav_prices.append(nav)
        #self.etf_returns.append(etf_return)
        #self.nav_returns.append(nav_return)

        while len(self.spreads_basket) > 200:
            self.spreads_basket.pop(0)
        while len(self.ratios_basket) > 200:
            self.ratios_basket.pop(0)
        while len(self.spreads_roses) > 200:
            self.spreads_roses.pop(0)
        #while len(self.etf_prices) > 15:
            #self.etf_prices.pop(0)
        #while len(self.nav_prices) > 15:
            #self.nav_prices.pop(0)
        #while len(self.etf_returns) > 15:
            #self.etf_returns.pop(0)
        #while len(self.nav_returns) > 15:
            #self.nav_returns.pop(0)


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