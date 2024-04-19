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
    spreads_chocolates = []

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

                while len(self.mid_prices_history[symbol]) > 100:
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

    def linear_regression(self, y, x1, x2, x3):
        # Convert lists to NumPy arrays
        Y = np.array(y)
        X1 = np.array(x1)
        X2 = np.array(x2)
        X3 = np.array(x3)
        
        # Stack the independent variables into a matrix X with an intercept column (constant term)
        X = np.column_stack((np.ones(len(Y)), X1, X2, X3))
        
        # Perform the OLS regression using NumPy's least squares function, which returns the coefficients
        # np.linalg.lstsq returns several values, we're interested in the first one (the coefficients)
        coefficients, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        
        # Calculate standard errors of the coefficients
        # First, compute the mean squared error (MSE)
        mse = residuals / (len(Y) - len(coefficients))
        
        # Compute the variance-covariance matrix of the parameter estimates
        cov_b = mse * np.linalg.inv(X.T.dot(X))
        
        # Standard errors are the square roots of the diagonal elements of the covariance matrix
        std_err = np.sqrt(np.diagonal(cov_b))
        
        # Extract intercept, coefficients and their standard errors
        intercept = coefficients[0]
        intercept_std_err = std_err[0]
        betas = coefficients[1:]
        beta_std_errs = std_err[1:]

        print(intercept)
        print(intercept_std_err)
        print(betas)
        print(beta_std_errs)

        return intercept, intercept_std_err, betas, beta_std_errs
        
        

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
        basket_bid_2, basket_bid_vol_2 = self.get_bid2('GIFT_BASKET', state)
        basket_ask_2, basket_ask_vol_2 = self.get_ask2('GIFT_BASKET', state)
        chocolates_bid_2, chocolates_bid_vol_2 = self.get_bid2('CHOCOLATE', state)
        chocolates_ask_2, chocolates_ask_vol_2 = self.get_ask2('CHOCOLATE', state)
        strawberries_bid_2, strawberries_bid_vol_2 = self.get_bid2('STRAWBERRIES', state)
        strawberries_ask_2, strawberries_ask_vol_2 = self.get_ask2('STRAWBERRIES', state)
        roses_bid_2, roses_bid_vol_2 = self.get_bid2('ROSES', state)
        roses_ask_2, roses_ask_vol_2 = self.get_ask2('ROSES', state)


        #### THIRD BIDS AND ASKS ####
        basket_bid_3, basket_bid_vol_3 = self.get_bid3('GIFT_BASKET', state)
        basket_ask_3, basket_ask_vol_3 = self.get_ask3('GIFT_BASKET', state)
        chocolates_bid_3, chocolates_bid_vol_3 = self.get_bid3('CHOCOLATE', state)
        chocolates_ask_3, chocolates_ask_vol_3 = self.get_ask3('CHOCOLATE', state)
        strawberries_bid_3, strawberries_bid_vol_3 = self.get_bid3('STRAWBERRIES', state)
        strawberries_ask_3, strawberries_ask_vol_3 = self.get_ask3('STRAWBERRIES', state)
        roses_bid_3, roses_bid_vol_3 = self.get_bid3('ROSES', state)
        roses_ask_3, roses_ask_vol_3 = self.get_ask3('ROSES', state)

        gift_basket_prices = self.mid_prices_history['GIFT_BASKET']
        chocolates_prices = self.mid_prices_history['CHOCOLATE']
        strawberries_prices = self.mid_prices_history['STRAWBERRIES']
        roses_prices = self.mid_prices_history['ROSES']

        
        #print('SPREAD:', spread)
        #rolling_spreads = self.spreads_basket[-10:]
        #rolling_spread = statistics.mean(rolling_spreads)
   
        
        if len(self.mid_prices_history['GIFT_BASKET']) > 10:
            #intercept, intercept_std, betas, betas_std = self.linear_regression(gift_basket_prices, chocolates_prices, strawberries_prices, roses_prices)
            intercept = 165.3320184325068
            intercept_std = 72.08630428543141
            betas = [3.84317626, 6.17179092, 1.05264383]
            beta_chocolates = betas[0]
            beta_strawberries = betas[1]
            beta_roses = betas[2]

            nav = beta_chocolates * chocolates_mid_price + beta_strawberries *strawberries_mid_price + beta_roses * roses_mid_price + intercept
            #print('regression!')
            #print(beta_chocolates)
            #print(beta_strawberries)
            #print(beta_roses)
            #print(intercept)
            spread = basket_mid_price - nav 
            print('spread', spread)
            self.spreads_basket.append(spread)
            mean = statistics.mean(self.spreads_basket)
            std = statistics.stdev(self.spreads_basket)

            z_score = (spread - mean) / std
  
            print('Z-SCORE: ',z_score)

            print(intercept_std*1.2)
            
            
            #if z_score > -0.4 and z_score < 0.4 and current_position_all != 0:
            if spread < intercept_std * 0.1 and spread > intercept_std * (-0.1) and position_basket != 0:
                if position_basket > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_bid, -min(position_basket, basket_bid_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_2, - max(min(position_basket-basket_bid, basket_bid_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_3, - max(min(abs(-position_basket)-basket_bid_vol-basket_bid_vol_2, basket_bid_vol_3), 0)))
                elif position_basket < 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_ask, min(-position_basket, -basket_ask_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_2, max(min(-position_basket+basket_ask_vol, -basket_ask_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_3, max(min(-position_basket+basket_ask_vol+basket_ask_vol_2, -basket_ask_vol_3), 0)))
                """
                if position_chocolates > 0:
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, -min(position_chocolates, chocolates_bid_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_2,- max(min(position_chocolates-chocolates_bid_vol, chocolates_bid_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_3, -max(min(position_chocolates*4-chocolates_bid_vol-chocolates_bid_vol_2, chocolates_bid_vol_3),0)))
                elif position_chocolates < 0:
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, min(-position_chocolates, -chocolates_ask_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_2, max(min(-position_chocolates+chocolates_ask_vol, -chocolates_ask_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_3, max(min(-position_chocolates*4+chocolates_ask_vol+chocolates_ask_vol_2, -chocolates_ask_vol_3),0)))

                if position_strawberries > 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid, -max(min(position_strawberries, strawberries_bid_vol),0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_2,- max(min(position_strawberries-strawberries_bid_vol, strawberries_bid_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_3, -max(min(position_strawberries-strawberries_bid_vol-strawberries_bid_vol_2, strawberries_bid_vol_3),0)))
                elif position_strawberries < 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, max(min(-position_strawberries, -strawberries_ask_vol),0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_2, max(min(-position_strawberries+strawberries_ask_vol, -strawberries_ask_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_3, max(min(-position_strawberries+strawberries_ask_vol+strawberries_ask_vol_2, -strawberries_ask_vol_3),0)))

                if position_roses > 0:
                    roses_orders.append(Order('ROSES', roses_bid, -max(min(position_roses, roses_bid_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_2, -max(min(position_roses-roses_bid_vol, roses_bid_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_3, -max(min(position_roses-roses_bid_vol-roses_bid_vol_2, roses_bid_vol_3),0)))
                elif position_roses < 0:
                    roses_orders.append(Order('ROSES', roses_ask, max(min(-position_roses, -roses_ask_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_2, max(min(-position_roses+roses_ask_vol, -roses_ask_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_3, max(min(-position_roses+roses_ask_vol+roses_ask_vol_2, -roses_ask_vol_3),0)))            
                """

            #if z_score > 2: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                qt_basket = min(abs(sell_volume_basket), basket_bid_vol) 
                print(qt_basket)
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol)) 
                print(qt_chocolates)
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol)) 
                print(qt_strawberries)
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol)) 
                print(qt_roses)

                #n_tradable = min(qt_basket, math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries), math.floor(qt_roses/beta_roses ))
                #print(n_tradable)
                n_tradable = 1
                if n_tradable > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_bid, sell_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_bid, - min(abs(n_tradable), basket_bid_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_2, - max(min(abs(n_tradable)-basket_bid, basket_bid_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_3, - max(min(abs(n_tradable)-basket_bid_vol-basket_bid_vol_2, basket_bid_vol_3), 0)))
                    
                    ##chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, min(math.floor(n_tradable*beta_chocolates), -chocolates_ask_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_2, max(min(math.floor(n_tradable*beta_chocolates)+chocolates_ask_vol, -chocolates_ask_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_3, max(min(n_tradable*4+chocolates_ask_vol+chocolates_ask_vol_2, -chocolates_ask_vol_3),0)))

                    ##strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, max(min(math.floor(n_tradable*beta_strawberries), -strawberries_ask_vol),0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_2, max(min(math.floor(n_tradable*beta_strawberries)+strawberries_ask_vol, -strawberries_ask_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_3, max(min(n_tradable*6+strawberries_ask_vol+strawberries_ask_vol_2, -strawberries_ask_vol_3),0)))

                    ##roses_orders.append(Order('ROSES', roses_ask, max(min(math.floor(n_tradable*beta_roses ), -roses_ask_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_2, max(min(math.floor(n_tradable*beta_roses )+roses_ask_vol, -roses_ask_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_3, max(min(n_tradable+roses_ask_vol+roses_ask_vol_2, -roses_ask_vol_3),0)))
                    
                    
            #elif z_score < -2: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            if spread < intercept_std * (-1):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol)) 
                print(qt_basket)
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol) 
                print(qt_chocolates)
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol) 
                print(qt_strawberries)
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol) 
                print(qt_roses)

                #n_tradable = min(qt_basket, math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries), math.floor(qt_roses / beta_roses ))
                #print(n_tradable)
                n_tradable = 1
                if n_tradable > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_ask, buy_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_ask, min(n_tradable, -basket_ask_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_2, max(min(n_tradable+basket_ask_vol, -basket_ask_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_3, max(min(n_tradable+basket_ask_vol+basket_ask_vol_2, -basket_ask_vol_3), 0)))
                    
                    ##chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, -min(math.floor(n_tradable*beta_chocolates), chocolates_bid_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_2,- max(min(math.floor(n_tradable*beta_chocolates)-chocolates_bid_vol, chocolates_bid_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_3, -max(min(n_tradable*4-chocolates_bid_vol-chocolates_bid_vol_2, chocolates_bid_vol_3),0)))

                    ##strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid, -max(min(math.floor(n_tradable*beta_strawberries), strawberries_bid_vol),0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_2,- max(min(math.floor(n_tradable*beta_strawberries)-strawberries_bid_vol, strawberries_bid_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_3, -max(min(n_tradable*6-strawberries_bid_vol-strawberries_bid_vol_2, strawberries_bid_vol_3),0)))

                    ##roses_orders.append(Order('ROSES', roses_bid, -max(min(math.floor(n_tradable*beta_roses ), roses_bid_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_2, -max(min(math.floor(n_tradable*beta_roses )-roses_bid_vol, roses_bid_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_3, -max(min(n_tradable-roses_bid_vol-roses_bid_vol_2, roses_bid_vol_3),0)))
           

        
        if len(self.mid_prices_history['ROSES']) > 10:
            #intercept, intercept_std, betas, betas_std = self.linear_regression(gift_basket_prices, chocolates_prices, strawberries_prices, roses_prices)
            intercept = 24820
            intercept_std = 163.617
            beta_roses = 3.1629

            nav = beta_roses * roses_mid_price + intercept
            #print('regression!')
            #print(beta_chocolates)
            #print(beta_strawberries)
            #print(beta_roses)
            #print(intercept)
            spread = basket_mid_price - nav 
            print('spread', spread)
            self.spreads_basket.append(spread)
            mean = statistics.mean(self.spreads_basket)
            std = statistics.stdev(self.spreads_basket)

            z_score = (spread - mean) / std
  
            print('Z-SCORE: ',z_score)

            print(intercept_std*1.2)
            
            
            #if z_score > -0.4 and z_score < 0.4 and current_position_all != 0:
            if spread < intercept_std * 0.1 and spread > intercept_std * (-0.1) and position_roses != 0:
                """
                if position_basket > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_bid, - min(position_basket, basket_bid_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_2, - max(min(position_basket-basket_bid, basket_bid_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_3, - max(min(abs(-position_basket)-basket_bid_vol-basket_bid_vol_2, basket_bid_vol_3), 0)))
                elif position_basket < 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_ask, min(-position_basket, -basket_ask_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_2, max(min(-position_basket+basket_ask_vol, -basket_ask_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_3, max(min(-position_basket+basket_ask_vol+basket_ask_vol_2, -basket_ask_vol_3), 0)))

                if position_chocolates > 0:
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, -min(position_chocolates, chocolates_bid_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_2,- max(min(position_chocolates-chocolates_bid_vol, chocolates_bid_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_3, -max(min(position_chocolates*4-chocolates_bid_vol-chocolates_bid_vol_2, chocolates_bid_vol_3),0)))
                elif position_chocolates < 0:
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, min(-position_chocolates, -chocolates_ask_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_2, max(min(-position_chocolates+chocolates_ask_vol, -chocolates_ask_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_3, max(min(-position_chocolates*4+chocolates_ask_vol+chocolates_ask_vol_2, -chocolates_ask_vol_3),0)))

                if position_strawberries > 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid, -max(min(position_strawberries, strawberries_bid_vol),0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_2,- max(min(position_strawberries-strawberries_bid_vol, strawberries_bid_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_3, -max(min(position_strawberries-strawberries_bid_vol-strawberries_bid_vol_2, strawberries_bid_vol_3),0)))
                elif position_strawberries < 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, max(min(-position_strawberries, -strawberries_ask_vol),0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_2, max(min(-position_strawberries+strawberries_ask_vol, -strawberries_ask_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_3, max(min(-position_strawberries+strawberries_ask_vol+strawberries_ask_vol_2, -strawberries_ask_vol_3),0)))
                """
                if position_roses > 0:
                    roses_orders.append(Order('ROSES', roses_bid, -max(min(position_roses, roses_bid_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_2, -max(min(position_roses-roses_bid_vol, roses_bid_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_3, -max(min(position_roses-roses_bid_vol-roses_bid_vol_2, roses_bid_vol_3),0)))
                elif position_roses < 0:
                    roses_orders.append(Order('ROSES', roses_ask, max(min(-position_roses, -roses_ask_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_2, max(min(-position_roses+roses_ask_vol, -roses_ask_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_3, max(min(-position_roses+roses_ask_vol+roses_ask_vol_2, -roses_ask_vol_3),0)))            
            
            #if z_score > 2: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                qt_basket = min(abs(sell_volume_basket), basket_bid_vol) 
                print(qt_basket)
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol)) 
                print(qt_chocolates)
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol)) 
                print(qt_strawberries)
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol)) 
                print(qt_roses)

                #n_tradable = min(qt_basket, math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries), math.floor(qt_roses/beta_roses ))
                n_tradable = 1
                print(n_tradable)

                if n_tradable > 0:
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid, sell_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_bid, - min(abs(n_tradable), basket_bid_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_2, - max(min(abs(n_tradable)-basket_bid, basket_bid_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_3, - max(min(abs(n_tradable)-basket_bid_vol-basket_bid_vol_2, basket_bid_vol_3), 0)))
                    
                    roses_orders.append(Order('ROSES', roses_ask, buy_volume_roses))
                    ##chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, min(math.floor(n_tradable*beta_chocolates), -chocolates_ask_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_2, max(min(math.floor(n_tradable*beta_chocolates)+chocolates_ask_vol, -chocolates_ask_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_3, max(min(n_tradable*4+chocolates_ask_vol+chocolates_ask_vol_2, -chocolates_ask_vol_3),0)))

                    ##strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, max(min(math.floor(n_tradable*beta_strawberries), -strawberries_ask_vol),0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_2, max(min(math.floor(n_tradable*beta_strawberries)+strawberries_ask_vol, -strawberries_ask_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_3, max(min(n_tradable*6+strawberries_ask_vol+strawberries_ask_vol_2, -strawberries_ask_vol_3),0)))

                    ##roses_orders.append(Order('ROSES', roses_ask, max(min(math.floor(n_tradable*beta_roses ), -roses_ask_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_2, max(min(math.floor(n_tradable*beta_roses )+roses_ask_vol, -roses_ask_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_3, max(min(n_tradable+roses_ask_vol+roses_ask_vol_2, -roses_ask_vol_3),0)))
                    
                    
            #elif z_score < -2: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            if spread < intercept_std * (-1):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol)) 
                print(qt_basket)
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol) 
                print(qt_chocolates)
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol) 
                print(qt_strawberries)
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol) 
                print(qt_roses)

                #n_tradable = min(qt_basket, math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries), math.floor(qt_roses / beta_roses ))
                n_tradable = 1
                print(n_tradable)
                if n_tradable > 0:
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask, buy_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_ask, min(n_tradable, -basket_ask_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_2, max(min(n_tradable+basket_ask_vol, -basket_ask_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_3, max(min(n_tradable+basket_ask_vol+basket_ask_vol_2, -basket_ask_vol_3), 0)))
                    
                    roses_orders.append(Order('ROSES', roses_bid, sell_volume_roses))
                    ##chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, -min(math.floor(n_tradable*beta_chocolates), chocolates_bid_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_2,- max(min(math.floor(n_tradable*beta_chocolates)-chocolates_bid_vol, chocolates_bid_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_3, -max(min(n_tradable*4-chocolates_bid_vol-chocolates_bid_vol_2, chocolates_bid_vol_3),0)))

                    ##strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid, -max(min(math.floor(n_tradable*beta_strawberries), strawberries_bid_vol),0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_2,- max(min(math.floor(n_tradable*beta_strawberries)-strawberries_bid_vol, strawberries_bid_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_3, -max(min(n_tradable*6-strawberries_bid_vol-strawberries_bid_vol_2, strawberries_bid_vol_3),0)))

                    ##roses_orders.append(Order('ROSES', roses_bid, -max(min(math.floor(n_tradable*beta_roses ), roses_bid_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_2, -max(min(math.floor(n_tradable*beta_roses )-roses_bid_vol, roses_bid_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_3, -max(min(n_tradable-roses_bid_vol-roses_bid_vol_2, roses_bid_vol_3),0)))   
        

        intercept = 25410
        intercept_std = 88.407
        beta_chocolates = 5.7223

        nav = beta_chocolates * chocolates_mid_price #+ intercept
        spread_chocolate = basket_mid_price - nav
        self.spreads_chocolates.append(spread_chocolate)
        

        if len(self.spreads_chocolates) >= 10:
            rolling_spreads = self.spreads_chocolates[-10:]
            rolling_spread = statistics.mean(rolling_spreads)
            #print('regression!')
            #print(beta_chocolates)
            #print(beta_strawberries)
            #print(beta_roses)
            #print(intercept)
            mean = statistics.mean(self.spreads_chocolates)
            std = statistics.stdev(self.spreads_chocolates)

            z_score = (rolling_spread - mean) / std
  
            print('Z-SCORE: ',z_score)

            print(intercept_std*1.2)

            intercept = 26800
            intercept_std = 441.474
            #beta_chocolates = 5.2957
            beta_strawberries = 10.9044

            nav = beta_strawberries * strawberries_mid_price  + intercept
            spread = basket_mid_price - nav
            
            #if z_score > -0.4 and z_score < 0.4 and current_position_all != 0:
            if spread < intercept_std * 0.1 and spread > intercept_std * (-0.1) and (position_strawberries !=0):
                
            
                if position_strawberries > 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid, -max(min(position_strawberries, strawberries_bid_vol),0)))
                    #strawberries_orders.append(Order('strawberries', strawberries_bid_2, -max(min(position_strawberries-strawberries_bid_vol, strawberries_bid_vol_2),0)))
                    #strawberries_orders.append(Order('strawberries', strawberries_bid_3, -max(min(position_strawberries-strawberries_bid_vol-strawberries_bid_vol_2, strawberries_bid_vol_3),0)))
                elif position_strawberries < 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, max(min(-position_strawberries, -strawberries_ask_vol),0)))
                    #strawberries_orders.append(Order('strawberries', strawberries_ask_2, max(min(-position_strawberries+strawberries_ask_vol, -strawberries_ask_vol_2),0)))
                    #strawberries_orders.append(Order('strawberries', strawberries_ask_3, max(min(-position_strawberries+strawberries_ask_vol+strawberries_ask_vol_2, -strawberries_ask_vol_3),0)))            
            
            #if z_score > 1: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                #qt_basket = min(abs(sell_volume_basket), basket_bid_vol) 
                print(qt_basket)
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol)) 
                print(qt_chocolates)
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol)) 
                print(qt_strawberries)
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol)) 
                print(qt_roses)

                #n_tradable = min(math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries))
                n_tradable = 1
                print(n_tradable)

                if n_tradable > 0:
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid, sell_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_bid, - min(abs(n_tradable), basket_bid_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_2, - max(min(abs(n_tradable)-basket_bid, basket_bid_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_3, - max(min(abs(n_tradable)-basket_bid_vol-basket_bid_vol_2, basket_bid_vol_3), 0)))
                    
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, min(math.floor(n_tradable*beta_chocolates), -chocolates_ask_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_2, max(min(math.floor(n_tradable*beta_chocolates)+chocolates_ask_vol, -chocolates_ask_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_3, max(min(n_tradable*4+chocolates_ask_vol+chocolates_ask_vol_2, -chocolates_ask_vol_3),0)))

                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, buy_volume_strawberries))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_2, max(min(math.floor(n_tradable*beta_strawberries)+strawberries_ask_vol, -strawberries_ask_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_3, max(min(n_tradable*6+strawberries_ask_vol+strawberries_ask_vol_2, -strawberries_ask_vol_3),0)))

                    ##roses_orders.append(Order('ROSES', roses_ask, max(min(math.floor(n_tradable*beta_roses ), -roses_ask_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_2, max(min(math.floor(n_tradable*beta_roses )+roses_ask_vol, -roses_ask_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_3, max(min(n_tradable+roses_ask_vol+roses_ask_vol_2, -roses_ask_vol_3),0)))
                    
                    
            #elif z_score < -1: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            elif spread < intercept_std * (-1):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol)) 
                print(qt_basket)
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol) 
                print(qt_chocolates)
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol) 
                print(qt_strawberries)
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol) 
                print(qt_roses)

                #n_tradable = min(math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries))
                n_tradable = 1
                print(n_tradable)
                if n_tradable > 0:
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask, buy_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_ask, min(n_tradable, -basket_ask_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_2, max(min(n_tradable+basket_ask_vol, -basket_ask_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_3, max(min(n_tradable+basket_ask_vol+basket_ask_vol_2, -basket_ask_vol_3), 0)))
                    
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, sell_volume_chocolates))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, -min(math.floor(n_tradable*beta_chocolates), chocolates_bid_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_2,- max(min(math.floor(n_tradable*beta_chocolates)-chocolates_bid_vol, chocolates_bid_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_3, -max(min(n_tradable*4-chocolates_bid_vol-chocolates_bid_vol_2, chocolates_bid_vol_3),0)))

                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid, sell_volume_strawberries))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_2,- max(min(math.floor(n_tradable*beta_strawberries)-strawberries_bid_vol, strawberries_bid_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_3, -max(min(n_tradable*6-strawberries_bid_vol-strawberries_bid_vol_2, strawberries_bid_vol_3),0)))

                    ##roses_orders.append(Order('ROSES', roses_bid, -max(min(math.floor(n_tradable*beta_roses ), roses_bid_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_2, -max(min(math.floor(n_tradable*beta_roses )-roses_bid_vol, roses_bid_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_3, -max(min(n_tradable-roses_bid_vol-roses_bid_vol_2, roses_bid_vol_3),0)))   
        
        
        if len(self.spreads_chocolates) >= 10:

            intercept = 3878.8146
            intercept_std = 41.821
            #beta_chocolates = 5.2957
            beta_chocolates = 1.3427

            nav = beta_chocolates * chocolates_mid_price  + intercept
            spread = roses_mid_price - nav
            
            #if z_score > -0.4 and z_score < 0.4 and current_position_all != 0:
            if spread < intercept_std * 0.1 and spread > intercept_std * (-0.1) and (position_chocolates !=0):
                
                if position_chocolates > 0:
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, -max(min(position_chocolates, chocolates_bid_vol),0)))
                    #chocolates_orders.append(Order('chocolates', chocolates_bid_2, -max(min(position_chocolates-chocolates_bid_vol, chocolates_bid_vol_2),0)))
                    #chocolates_orders.append(Order('chocolates', chocolates_bid_3, -max(min(position_chocolates-chocolates_bid_vol-chocolates_bid_vol_2, chocolates_bid_vol_3),0)))
                elif position_chocolates < 0:
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, max(min(-position_chocolates, -chocolates_ask_vol),0)))
                    #chocolates_orders.append(Order('chocolates', chocolates_ask_2, max(min(-position_chocolates+chocolates_ask_vol, -chocolates_ask_vol_2),0)))
                    #chocolates_orders.append(Order('chocolates', chocolates_ask_3, max(min(-position_chocolates+chocolates_ask_vol+chocolates_ask_vol_2, -chocolates_ask_vol_3),0))) 

            
            #if z_score > 1: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                #qt_basket = min(abs(sell_volume_basket), basket_bid_vol) 
                print(qt_basket)
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol)) 
                print(qt_chocolates)
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol)) 
                print(qt_strawberries)
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol)) 
                print(qt_roses)

                #n_tradable = min(math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries))
                n_tradable = 1
                print(n_tradable)

                if n_tradable > 0:
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid, sell_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_bid, - min(abs(n_tradable), basket_bid_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_2, - max(min(abs(n_tradable)-basket_bid, basket_bid_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_3, - max(min(abs(n_tradable)-basket_bid_vol-basket_bid_vol_2, basket_bid_vol_3), 0)))
                    
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, min(math.floor(n_tradable*beta_chocolates), -chocolates_ask_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_2, max(min(math.floor(n_tradable*beta_chocolates)+chocolates_ask_vol, -chocolates_ask_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_ask_3, max(min(n_tradable*4+chocolates_ask_vol+chocolates_ask_vol_2, -chocolates_ask_vol_3),0)))

                    chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, buy_volume_chocolates))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_2, max(min(math.floor(n_tradable*beta_strawberries)+strawberries_ask_vol, -strawberries_ask_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask_3, max(min(n_tradable*6+strawberries_ask_vol+strawberries_ask_vol_2, -strawberries_ask_vol_3),0)))

                    ##roses_orders.append(Order('ROSES', roses_ask, max(min(math.floor(n_tradable*beta_roses ), -roses_ask_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_2, max(min(math.floor(n_tradable*beta_roses )+roses_ask_vol, -roses_ask_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_3, max(min(n_tradable+roses_ask_vol+roses_ask_vol_2, -roses_ask_vol_3),0)))
                    
                    
            #elif z_score < -1: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            elif spread < intercept_std * (-1):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol)) 
                print(qt_basket)
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol) 
                print(qt_chocolates)
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol) 
                print(qt_strawberries)
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol) 
                print(qt_roses)

                #n_tradable = min(math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries))
                n_tradable = 1
                print(n_tradable)
                if n_tradable > 0:
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask, buy_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_ask, min(n_tradable, -basket_ask_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_2, max(min(n_tradable+basket_ask_vol, -basket_ask_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_3, max(min(n_tradable+basket_ask_vol+basket_ask_vol_2, -basket_ask_vol_3), 0)))
                    
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, sell_volume_chocolates))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, -min(math.floor(n_tradable*beta_chocolates), chocolates_bid_vol)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_2,- max(min(math.floor(n_tradable*beta_chocolates)-chocolates_bid_vol, chocolates_bid_vol_2),0)))
                    #chocolates_orders.append(Order('CHOCOLATE', chocolates_bid_3, -max(min(n_tradable*4-chocolates_bid_vol-chocolates_bid_vol_2, chocolates_bid_vol_3),0)))

                    chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, sell_volume_chocolates))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_2,- max(min(math.floor(n_tradable*beta_strawberries)-strawberries_bid_vol, strawberries_bid_vol_2), 0)))
                    #strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid_3, -max(min(n_tradable*6-strawberries_bid_vol-strawberries_bid_vol_2, strawberries_bid_vol_3),0)))

                    ##roses_orders.append(Order('ROSES', roses_bid, -max(min(math.floor(n_tradable*beta_roses ), roses_bid_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_2, -max(min(math.floor(n_tradable*beta_roses )-roses_bid_vol, roses_bid_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_3, -max(min(n_tradable-roses_bid_vol-roses_bid_vol_2, roses_bid_vol_3),0)))   
        
        

        

        while len(self.spreads_basket) > 300:
            self.spreads_basket.pop(0)
        while len(self.spreads_chocolates) > 200:
            self.spreads_chocolates.pop(0)
        while len(self.ratios_basket) > 300:
            self.ratios_basket.pop(0)
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