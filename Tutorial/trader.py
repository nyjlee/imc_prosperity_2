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
    spread_history = {"AMETHYSTS": [], "STARFRUIT": []}


    def __init__(self) -> None:

        self.ema_prices = dict()
        for product in self.PRODUCTS:
            self.ema_prices[product] = None
        self.ema_param = 0.2

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
    
    
    def get_spread(self, product, state : TradingState):
        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = self.DEFAULT_PRICES[product]
        
        if len(market_bids) == 0 or len(market_asks) == 0:
            return np.nan
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        spread = best_ask - best_bid
        return spread
        

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

    def update_spread_history(self, state : TradingState):
        for product in self.PRODUCTS:
            spread = self.get_spread(product, state)
            self.spread_history[product].append(spread)

    def calculate_sma(self, product, window_size):
        sma = None
        prices = pd.Series(self.prices_history[product])
        if len(prices) >= window_size:
            window_sum = prices.iloc[-window_size:].sum()
            sma = window_sum / window_size
        return sma
    
    def calculate_ema(self, product, window_size):
        ema = None
        prices = pd.Series(self.prices_history[product])
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
    
    def amethysts_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of amethysts.
        """
        orders = []

        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        ask_volume = - self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts

        if position_amethysts == 0:
            # Not long nor short
            orders.append(Order('AMETHYSTS', max(9996, math.floor(self.ema_prices['AMETHYSTS'] - 2)), bid_volume))
            orders.append(Order('AMETHYSTS', min(10004,math.ceil(self.ema_prices['AMETHYSTS'] + 2)), ask_volume))

        if position_amethysts > 0:
            # Long position
            orders.append(Order('AMETHYSTS', max(9996, math.floor(self.ema_prices['AMETHYSTS'] - 2)), bid_volume))
            orders.append(Order('AMETHYSTS', min(10004,math.ceil(self.ema_prices['AMETHYSTS'] + 1)), ask_volume))
            #orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] - 3), bid_volume))
            #orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 2), ask_volume))

        if position_amethysts < 0:
            # Short position
            orders.append(Order('AMETHYSTS', max(9995, math.floor(self.ema_prices['AMETHYSTS'] - 1)), bid_volume))
            orders.append(Order('AMETHYSTS', min(10005,math.ceil(self.ema_prices['AMETHYSTS'] + 2)), ask_volume))
            #orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] - 2), bid_volume))
            #orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 3), ask_volume))

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
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -1), int(math.ceil((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -2), int(math.floor((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS']  + 3), ask_volume))

        elif position_amethysts > 0:
            # Long position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS']  - 2), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 1), int(math.ceil((ask_volume/2)))))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 2), int(math.floor((ask_volume/2)))))

        elif position_amethysts < 0:
            # Short position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -1), int(math.ceil((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] -2), int(math.floor((bid_volume/2)))))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS']  + 2), ask_volume))


        #print(orders)

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
        upper_limit = sma_20 + 2 * sd
        lower_limit = sma_20 - 2 * sd

        """
        if last_price < lower_limit:
            orders.append(Order('STARFRUIT', math.floor(mid_price-2), bid_volume))
            orders.append(Order('STARFRUIT', math.floor(mid_price+3), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(mid_price+4), int(math.ceil(ask_volume/2))))
            
        elif last_price > upper_limit:
            orders.append(Order('STARFRUIT', math.ceil(mid_price+2), ask_volume))
            orders.append(Order('STARFRUIT', math.ceil(mid_price-3), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(mid_price-4), int(math.ceil(bid_volume/2))))
        """
        
        if last_price > ema_21 and ema_8 > ema_21:
            orders.append(Order('STARFRUIT', math.floor(mid_price - 2.5), bid_volume))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+2), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+3), int(math.ceil(ask_volume/2))))
        elif last_price < ema_21 and ema_8 < ema_21:
            orders.append(Order('STARFRUIT', math.ceil(mid_price + 2.5), ask_volume))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-2), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-3), int(math.ceil(bid_volume/2))))
        

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

        lags = self.prices_history['STARFRUIT'][-4:]

        coef = [0.3176191343791975,  0.22955395157579261 ,  0.24255751299309652,  0.20773853347672797]
        intercept = 12.56819767718207
        forecasted_price = intercept
        for i, val in enumerate(lags):
            forecasted_price += val * coef[i]
        
        #print('Forecasted Price:',forecasted_price)
         
        if mid_price < forecasted_price:
            orders.append(Order('STARFRUIT', math.floor(mid_price-2.5), bid_volume))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price+1), ask_volume))
        elif mid_price > forecasted_price:
            orders.append(Order('STARFRUIT', math.ceil(mid_price+2.5), ask_volume))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price-1), bid_volume))   

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
        upper_limit = sma_20 + 2 * sd
        lower_limit = sma_20 - 2 * sd

        
        if forecasted_price > best_bid+1:
            orders.append(Order('STARFRUIT', math.floor(best_bid+1), bid_volume))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+2), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+3), int(math.ceil(ask_volume/2))))
        elif forecasted_price < best_ask-1:
            orders.append(Order('STARFRUIT', math.floor(best_ask-1), ask_volume))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-2), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-3), int(math.ceil(bid_volume/2))))

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
        self.update_spread_history(state)
        #print(self.prices_history)


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


        for product in self.pnl_tracker.keys():
            while len(self.pnl_tracker[product]) > 20:
                self.pnl_tracker[product].pop(0)


        """
        if state.timestamp == 199900:
            final_pnl = 0
            for product in state.order_depths.keys():
                settled_pnl = 0
                best_sell = min(state.order_depths[product].sell_orders.keys())
                best_buy = max(state.order_depths[product].buy_orders.keys())
                mid_price = (best_sell + best_buy) / 2

                if self.get_position(product, state) < 0:
                    settled_pnl += self.get_position(product, state) * mid_price
                else:
                    settled_pnl += self.get_position(product, state) * mid_price
                final_pnl += settled_pnl + self.current_pnl[product]
                print(f'\nFor product {product}, Pnl: {settled_pnl + self.current_pnl[product]}, Qty. Traded: {self.qt_traded[product]}')

            print(f'\nFinal Day Expected Pnl: {round(final_pnl,2)}')
        """

        # AMETHYSTS STRATEGY
        
        try:
            result['AMETHYSTS'] = self.amethysts_strategy2(state)
        except Exception as e:
            print("Error in AMETHYSTS strategy")
            print(e)
        

        # STARFRUIT STRATEGY
        try:
            result['STARFRUIT'] = self.starfruit_strategy3(state)
        except Exception as e:
            print("Error in STARFRUIT strategy")
            print(e)
        
                
        traderData = "SAMPLE" 
        
		# Sample conversion request. Check more details below. 
        conversions = 0
        return result, conversions, traderData