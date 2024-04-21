from typing import Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

import math
import pandas as pd
import numpy as np
import statistics
import json
from typing import Any


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()


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
        'AMETHYSTS' : 10000,
        'STARFRUIT' : 5000,
        'ORCHIDS': 1000,
        'GIFT_BASKET' : 70000,
        'CHOCOLATE' : 8000,
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

    prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    mid_prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    mid_p_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    p_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    errors_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    forecasted_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [], 'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}


    current_signal = {"AMETHYSTS": "", "STARFRUIT": "None", "ORCHIDS": "None"}

    export_tariffs = {"Min": 1000, "Max": 0, "Second Max": 0}

    spreads_basket = []
    spreads_roses = []
    spreads_chocolates = []

    ratios_basket = []

    etf_prices = []
    etf_returns = []
    nav_prices = []
    nav_returns = []

    coconuts_returns = []
    last_variances = []


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

        self.omega = 0.2478
        self.alpha = 8.9738e-03
        self.beta = 0.7572

        self.initial_last_variance = 1.0489105974177813


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
        if len(market_bids) != 0:
            best_bid, best_bid_amount = list(market_bids.items())[0]
        else:
            best_bid, best_bid_amount = 0, 0 

        return best_bid, best_bid_amount

    def get_best_ask(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        if len(market_asks) != 0:
            best_ask, best_ask_amount = list(market_asks.items())[0]
        else:
            best_ask, best_ask_amount = 100000, 0

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
    
    def norm_cdf(self, x):
        """
        Calculate the CDF of the standard normal distribution at x.
        """
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes_call(self, S, K, T, sigma, r=0.0):
        """
        Calculate the Black-Scholes price of a European call option using math.erf for the normal CDF.

        Parameters:
        - S (float): Current stock price
        - K (float): Strike price of the option
        - T (float): Time to expiration in years (250/365 for your case)
        - sigma (float): Annualized volatility of the stock
        - r (float): Risk-free rate, set to 0.0 as default

        Returns:
        - float: Price of the call option
        """
        # Calculate d1 and d2 components
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate the call price using the normal CDF approximation
        call_price = (S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2))
        return call_price
    
    def black_scholes_delta(self, S, K, T, sigma, r=0.0):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = self.norm_cdf(d1)  # Call option delta
        return delta
    
    def black_scholes_gamma(self, S, K, T, sigma, r=0.0):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        pdf_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)  # Probability density function at d1
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        return gamma
    
    def black_scholes_vega(self, S, K, T, sigma, r=0.0):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        pdf_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)  # Probability density function at d1
        vega = S * pdf_d1 * np.sqrt(T)  # Vega is typically given per 1% change in volatility
        return vega
    

    def implied_volatility(self, market_price, S, K, T, initial_sigma=0.2, tolerance=1e-5, max_iterations=100):
        """
        Estimate the implied volatility of a European call option using the bisection method.
        
        Parameters:
        - market_price (float): The observed market price of the option.
        - S (float): Current stock price.
        - K (float): Strike price of the option.
        - T (float): Time to expiration in years.
        - initial_sigma (float): Initial guess for volatility.
        - tolerance (float): Tolerance level for stopping the search.
        - max_iterations (int): Maximum number of iterations to perform.
        
        Returns:
        - float: Estimated implied volatility.
        """
        sigma_low = 0.001
        sigma_high = 1.0
        for i in range(max_iterations):
            sigma_mid = (sigma_low + sigma_high) / 2
            price_mid = self.black_scholes_call(S, K, T, sigma_mid, r=0.0)
            
            if price_mid > market_price:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid
                
            if abs(price_mid - market_price) < tolerance:
                return sigma_mid
        
        return (sigma_low + sigma_high) / 2  # Return the best estimate if convergence criterion not met


    def forecast_volatility(self, last_variance, last_return, omega, alpha, beta):
        """
        Forecast the next period's variance using the GARCH(1,1) model.

        Parameters:
        last_variance (float): The last known variance.
        last_return (float): The last observed return.
        omega (float): The GARCH model constant (long-run average variance).
        alpha (float): The coefficient for the last period's squared return.
        beta (float): The coefficient for the last period's variance.

        Returns:
        float: The forecasted variance for the next period.
        """
        scaling_factor = 1e+04
        # Since the coefficients were obtained from scaled returns, we scale the last return
        scaled_last_return = last_return * scaling_factor  # Adjust based on your scaling factor used in GARCH model estimation
        
        # Calculate the next period's variance using the GARCH(1,1) formula
        next_variance = omega + alpha * (scaled_last_return ** 2) + beta * last_variance

        adjusted_vol = np.sqrt(next_variance / 1e+08)
        annualized_volatility = adjusted_vol * np.sqrt(252)
        annualized_volatility = annualized_volatility * np.sqrt(10000)
        
        # Return the forecasted variance, scaled back to the original scale
        return next_variance, annualized_volatility


    def amethysts_strategy(self, state : TradingState) -> List[Order]:
        """
        Buying and Selling based on last trade price vs mean price (ceiling floor version)
        """
        orders = []
        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        ask_volume = - self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        last_price = self.get_last_price('AMETHYSTS', state.own_trades, state.market_trades)

        ema = self.ema_prices['AMETHYSTS']
        
        spread = 1
        open_spread = 3
        position_limit = 20
        position_spread = 16
        current_position = state.position.get("AMETHYSTS",0)
        best_ask = 0
        best_bid = 0
                
        order_depth_ame: OrderDepth = state.order_depths["AMETHYSTS"]
                
        # Check for anyone willing to sell for lower than 10 000 - 1
        if len(order_depth_ame.sell_orders) > 0:
            best_ask = min(order_depth_ame.sell_orders.keys())

            if best_ask <= 10000-spread:
                best_ask_volume = order_depth_ame.sell_orders[best_ask]
            else:
                best_ask_volume = 0
        else:
            best_ask_volume = 0

        # Check for buyers above 10 000 + 1
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
    


    def starfruit_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of starfruit.
        """

        orders = []
    
        position_starfruit = self.get_position('STARFRUIT', state)

        bid_volume = self.POSITION_LIMITS['STARFRUIT'] - position_starfruit
        ask_volume = - self.POSITION_LIMITS['STARFRUIT'] - position_starfruit

        best_bid, best_bid_amount = self.get_best_bid('STARFRUIT', state)
        best_ask, best_ask_amount = self.get_best_ask('STARFRUIT', state)
        mid_price = self.get_mid_price('STARFRUIT', state)
        spread = (best_ask - best_bid) / 2
        last_price = self.get_last_price('STARFRUIT', state.own_trades, state.market_trades)


        if len(self.mid_p_diff_history['STARFRUIT']) >= 6:
            AR_L1 = self.mid_p_diff_history['STARFRUIT'][-1]
            AR_L2 = self.mid_p_diff_history['STARFRUIT'][-2]
            AR_L3 = self.mid_p_diff_history['STARFRUIT'][-3]
            AR_L4 = self.mid_p_diff_history['STARFRUIT'][-4]
            AR_L5 = self.mid_p_diff_history['STARFRUIT'][-5]
            AR_L6 = self.mid_p_diff_history['STARFRUIT'][-6]
        
        if len(self.forecasted_diff_history['STARFRUIT']) > 0:
            forecasted_error = self.forecasted_diff_history['STARFRUIT'][-1] - self.mid_p_diff_history['STARFRUIT'][-1]
            self.errors_history['STARFRUIT'].append(forecasted_error)

        if len(self.errors_history['STARFRUIT']) < 2:
            #use this!
            #self.errors_history['STARFRUIT'].extend([1.682936, -2.797327, -0.480615])
            
            self.errors_history['STARFRUIT'].extend([0.021429, -0.490601, -1.861910]) ### best one

            #self.errors_history['STARFRUIT'].extend([-0.578542, 1.081536, 1.764008])

            #self.errors_history['STARFRUIT'].extend([-3.258368, -3.353484, -3.593285])
      
        else:
            MA_L1 = self.errors_history['STARFRUIT'][-1]
            MA_L2 = self.errors_history['STARFRUIT'][-2]
            MA_L3 = self.errors_history['STARFRUIT'][-3]
        
        #use this!
        #forecasted_diff = (AR_L1 * -1.1102) + (AR_L2 * -0.7276) + (AR_L3 * -0.0854) + (AR_L4 * -0.0674)
        #+ (AR_L5 * -0.0437) + (AR_L6 * -0.0176)+ (MA_L1 *  0.4021) + (MA_L2 * -0.0587) + (MA_L3 * -0.4357)

        #new data
        #forecasted_diff = (AR_L1 * -1.4799) + (AR_L2 * -0.8168) + (AR_L3 * -0.0868) + (AR_L4 * -0.0693)
        #+ (AR_L5 * -0.0492) + (AR_L6 * -0.0221)+ (MA_L1 *  0.7712) + (MA_L2 * -0.2324) + (MA_L3 * -0.4996)

        #new data2 best one yet
        forecasted_diff = (AR_L1 * -1.4799) + (AR_L2 * -0.5811) + (AR_L3 * -0.0511) + (AR_L4 * -0.0402)
        + (AR_L5 * -0.0288) + (AR_L6 * -0.0113)+ (MA_L1 * 0.2766) + (MA_L2 * -0.1251) + (MA_L3 * -0.3693)

        #forecasted_diff = (AR_L1 * -0.8938) + (AR_L2 * -0.6530) + (AR_L3 * -0.0280) + (AR_L4 * -0.0222)
        #+ (AR_L5 * -0.0165) + (AR_L6 * -0.0074)+ (MA_L1 * 0.1823 ) + (MA_L2 * 0.0157) + (MA_L3 * -0.4415)
        

        self.forecasted_diff_history['STARFRUIT'].append(forecasted_diff)

        forecasted_price = mid_price + forecasted_diff  

        #play with diff comb
        if forecasted_price > best_bid+2:
            orders.append(Order('STARFRUIT', math.floor(best_bid+1), bid_volume))
            #orders.append(Order('STARFRUIT', math.floor(best_bid+2), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+spread/2), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+spread/3), int(math.ceil(ask_volume/2))))
        elif forecasted_price < best_ask-2:
            orders.append(Order('STARFRUIT', math.floor(best_ask-1), ask_volume))
            #orders.append(Order('STARFRUIT', math.floor(best_ask-2), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-spread/2), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-spread/3), int(math.ceil(bid_volume/2))))
        

        return orders
    

    def orchids_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of orchids.
        """

        orders = []
    
        position_orchids = self.get_position('ORCHIDS', state)

        bid_volume = self.POSITION_LIMITS['ORCHIDS'] - position_orchids
        ask_volume = - self.POSITION_LIMITS['ORCHIDS'] - position_orchids

        best_bid, best_bid_amount = self.get_best_bid('ORCHIDS', state)
        best_ask, best_ask_amount  = self.get_best_ask('ORCHIDS', state)

        mid_price = self.get_mid_price('ORCHIDS', state)

        spread = (best_ask - best_bid) / 2

        observations = state.observations
        conversion_observations = observations.conversionObservations
        orchid_observations = conversion_observations['ORCHIDS']

        bid_price_south = orchid_observations.bidPrice
        ask_price_south = orchid_observations.askPrice
        transport_fees = orchid_observations.transportFees
        export_tariff = orchid_observations.exportTariff
        import_tariff = orchid_observations.importTariff
        sunlight = orchid_observations.sunlight
        humidity = orchid_observations.humidity

        if len(self.mid_p_diff_history['ORCHIDS']) >= 6:
            AR_L1 = self.mid_p_diff_history['ORCHIDS'][-1]
            AR_L2 = self.mid_p_diff_history['ORCHIDS'][-2]
            AR_L3 = self.mid_p_diff_history['ORCHIDS'][-3]
            AR_L4 = self.mid_p_diff_history['ORCHIDS'][-4]
            AR_L5 = self.mid_p_diff_history['ORCHIDS'][-5]
            AR_L6 = self.mid_p_diff_history['ORCHIDS'][-6]
        else:
            return orders
        
        if len(self.forecasted_diff_history['ORCHIDS']) > 0:
            forecasted_error = self.forecasted_diff_history['ORCHIDS'][-1] - self.p_diff_history['ORCHIDS'][-1]
            self.errors_history['ORCHIDS'].append(forecasted_error)

        if len(self.errors_history['ORCHIDS']) < 2:
            #use this!
            self.errors_history['ORCHIDS'].extend([-0.021682, -2.008885, 0.981433])
    
        else:
            MA_L1 = self.errors_history['ORCHIDS'][-1]
            MA_L2 = self.errors_history['ORCHIDS'][-2]
            MA_L3 = self.errors_history['ORCHIDS'][-3]

        forecasted_diff = (AR_L1 * 0.0003) + (AR_L2 * -0.0036) + (AR_L3 * 0.0011) + (AR_L4 * -0.0039)
        + (AR_L5 * -0.0062) + (AR_L6 * -0.0020)+ (MA_L1 *  0.0003) + (MA_L2 * -0.0036) + (MA_L3 * 0.0011)

        last_price = self.get_last_price('ORCHIDS', state.own_trades, state.market_trades)

        self.forecasted_diff_history['ORCHIDS'].append(forecasted_diff)

        forecasted_price = mid_price + forecasted_diff

        print('Last Price', last_price)  
        print('Forecast:', forecasted_price)

        if forecasted_price > best_bid+1:
            orders.append(Order('ORCHIDS', math.floor(best_bid+1), bid_volume))

            #orders.append(Order('ORCHIDS', math.floor(forecasted_price+spread/2), int(math.floor(ask_volume/2))))
            #orders.append(Order('ORCHIDS', math.floor(forecasted_price+spread/3), int(math.ceil(ask_volume/2))))
        elif forecasted_price < best_ask-1:
            orders.append(Order('ORCHIDS', math.floor(best_ask-1), ask_volume))

            #orders.append(Order('ORCHIDS', math.ceil(forecasted_price-spread/2), int(math.floor(bid_volume/2))))
            #orders.append(Order('ORCHIDS', math.ceil(forecasted_price-spread/3), int(math.ceil(bid_volume/2))))

        return orders
    
    def orchids_strategy2(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of orchids.
        """

        orders = []
    
        position_orchids = self.get_position('ORCHIDS', state)

        bid_volume = (self.POSITION_LIMITS['ORCHIDS'] - position_orchids ) / 10
        ask_volume = (- self.POSITION_LIMITS['ORCHIDS'] - position_orchids ) / 10

        best_bid, best_bid_amount = self.get_best_bid('ORCHIDS', state)
        best_ask, best_ask_amount  = self.get_best_ask('ORCHIDS', state)

        mid_price = self.get_mid_price('ORCHIDS', state)

        spread = (best_ask - best_bid) / 2

        observations = state.observations
        conversion_observations = observations.conversionObservations
        orchid_observations = conversion_observations['ORCHIDS']

        bid_price_south = orchid_observations.bidPrice
        ask_price_south = orchid_observations.askPrice
        transport_fees = orchid_observations.transportFees
        export_tariff = orchid_observations.exportTariff
        import_tariff = orchid_observations.importTariff
        sunlight = orchid_observations.sunlight
        humidity = orchid_observations.humidity

        conversion = 0
        """
        #### KEEP TRACK OF MIN, MAX AND SECOND MAX ####
        if export_tariff < self.export_tariffs['Min']:
            self.export_tariffs['Min'] = export_tariff
        if export_tariff > self.export_tariffs['Max']:
            self.export_tariffs['Second Max'] = self.export_tariffs['Max']
            self.export_tariffs['Max'] = export_tariff
        elif export_tariff > self.export_tariffs['Second Max'] and export_tariff != self.export_tariffs['Max']:
            self.export_tariffs['Second Max'] = export_tariff
        
        #### STRATEGY BASED ON EXPORT TARIFF ####
        if export_tariff >= 1.4 * self.export_tariffs['Min'] and export_tariff < self.export_tariffs['Max'] and export_tariff == self.export_tariffs['Second Max']:
            self.current_signal['ORCHIDS'] = 'Export Tariff Short'
            orders.append(Order('ORCHIDS', best_bid, ask_volume))
            return orders, conversion
        
        elif self.current_signal['ORCHIDS'] == 'Export Tariff Short' and position_orchids < 0 and export_tariff == self.export_tariffs['Min']:
            orders.append(Order('ORCHIDS', best_ask, -position_orchids))
            self.current_signal['ORCHIDS'] = 'Close Short'    
            return orders, conversion
        
        elif self.current_signal['ORCHIDS'] == 'Close Short' and position_orchids == 0:
            self.current_signal['ORCHIDS'] = 'None'  

        elif self.current_signal['ORCHIDS'] == 'Close Short' and position_orchids < 0 and export_tariff == self.export_tariffs['Min']:
            orders.append(Order('ORCHIDS', best_ask, -position_orchids))
            return orders, conversion
        
        elif self.current_signal['ORCHIDS'] == 'Export Tariff Short':
            orders.append(Order('ORCHIDS', best_bid, ask_volume))
            return orders, conversion
        """

        buy_price_south = ask_price_south + transport_fees + import_tariff
        sell_price_south = bid_price_south - transport_fees - export_tariff

        expected_profit_buying_3 = 0
        expected_profit_selling_3 = 0
        expected_profit_buying_2 = 0
        expected_profit_selling_2 = 0


        if position_orchids != 0:
            conversion = -position_orchids
        else:
            conversion = 0


        if best_bid+2 > buy_price_south:
            #orders.append(Order('ORCHIDS', math.floor(best_bid), ask_volume))
            expected_profit_selling_2 = best_bid+2 - buy_price_south
        if best_ask-2 < sell_price_south:
            #orders.append(Order('ORCHIDS', math.floor(best_ask), bid_volume))
            expected_profit_buying_2 = sell_price_south - best_ask-2
        
        if expected_profit_buying_2 > 0 and expected_profit_buying_2 > expected_profit_selling_2:
            orders.append(Order('ORCHIDS', math.floor(best_ask-2), math.floor(3*bid_volume)))
        if expected_profit_selling_2 > 0 and expected_profit_selling_2 > expected_profit_buying_2:
            orders.append(Order('ORCHIDS', math.floor(best_bid+2), math.floor(3*ask_volume)))

        
        if best_bid+3 > buy_price_south:
            #orders.append(Order('ORCHIDS', math.floor(best_bid), ask_volume))
            expected_profit_selling_3 = best_bid+3 - buy_price_south
        if best_ask-3 < sell_price_south:
            #orders.append(Order('ORCHIDS', math.floor(best_ask), bid_volume))
            expected_profit_buying_3 = sell_price_south - best_ask-3

        if expected_profit_buying_3 > 0 and expected_profit_buying_3 > expected_profit_selling_3:
            orders.append(Order('ORCHIDS', math.floor(best_ask-3), math.floor(7*bid_volume)))

        if expected_profit_selling_3 > 0 and expected_profit_selling_3 > expected_profit_buying_3:
            orders.append(Order('ORCHIDS', math.floor(best_bid+3), math.floor(7*ask_volume)))


        return orders, conversion
    
    def orchids_strategy3(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of orchids.
        """

        orders = []
    
        position_orchids = self.get_position('ORCHIDS', state)

        bid_volume = self.POSITION_LIMITS['ORCHIDS'] - position_orchids
        ask_volume = - self.POSITION_LIMITS['ORCHIDS'] - position_orchids

        best_bid, best_bid_amount = self.get_best_bid('ORCHIDS', state)
        best_ask, best_ask_amount  = self.get_best_ask('ORCHIDS', state)

        bid2, bid2_amount = self.get_bid2('ORCHIDS', state)
        ask2, ask2_amount  = self.get_ask2('ORCHIDS', state)

        bid3, bid3_amount = self.get_bid3('ORCHIDS', state)
        ask3, ask3_amount  = self.get_ask3('ORCHIDS', state)

        mid_price = self.get_mid_price('ORCHIDS', state)

        spread = (best_ask - best_bid) / 2

        observations = state.observations
        conversion_observations = observations.conversionObservations
        orchid_observations = conversion_observations['ORCHIDS']

        bid_price_south = orchid_observations.bidPrice
        ask_price_south = orchid_observations.askPrice
        transport_fees = orchid_observations.transportFees
        export_tariff = orchid_observations.exportTariff
        import_tariff = orchid_observations.importTariff
        sunlight = orchid_observations.sunlight
        humidity = orchid_observations.humidity

        buy_price_south = ask_price_south + transport_fees + import_tariff
        sell_price_south = bid_price_south - transport_fees - export_tariff

        expected_profit_buying = 0
        expected_profit_selling = 0
        expected_profit_buying_2 = 0
        expected_profit_selling_2 = 0
        expected_profit_buying_3 = 0
        expected_profit_selling_3 = 0


        if position_orchids != 0:
            conversion = -position_orchids
        else:
            conversion = 0

        if best_ask < sell_price_south:
            expected_profit_buying = sell_price_south - best_ask
        
        if best_bid > buy_price_south:
            expected_profit_selling = best_bid - buy_price_south

        if ask2 < sell_price_south and ask2_amount != 0:
            expected_profit_buying_2 = sell_price_south - ask2
        
        if bid2 > buy_price_south and bid2_amount != 0:
            expected_profit_selling_2 = bid2 - buy_price_south
        
        if ask3 < sell_price_south and ask3_amount != 0:
            expected_profit_buying_3 = sell_price_south - ask3
        
        if bid3 > buy_price_south and bid3_amount != 0:
            expected_profit_selling_3 = bid3 - buy_price_south

        print('orchids!!!1', expected_profit_buying)
        print('orchids!!!1', expected_profit_selling)
        print('orchids!!!2', expected_profit_buying_2)
        print('orchids!!!2', expected_profit_selling_2)
        print('orchids!!!3', expected_profit_buying_3)
        print('orchids!!!3', expected_profit_selling_3)
        
        """
        if expected_profit_buying_2 > 0 and expected_profit_buying_2 > expected_profit_selling_2:
            orders.append(Order('ORCHIDS', math.floor(best_ask),  min(ask2_amount, bid_volume)))

        if expected_profit_selling_2 > 0 and expected_profit_selling_2 > expected_profit_buying_2:
            orders.append(Order('ORCHIDS', math.floor(best_bid), max(ask_volume, -bid2_amount)))

        if expected_profit_buying > 0 and expected_profit_buying > expected_profit_selling:
            orders.append(Order('ORCHIDS', math.floor(best_ask), bid_volume - min(ask2_amount, bid_volume)))

        if expected_profit_selling > 0 and expected_profit_selling > expected_profit_buying:
            orders.append(Order('ORCHIDS', math.floor(best_bid), ask_volume - max(ask_volume, -bid2_amount)))
        """
        if expected_profit_buying > 0 and expected_profit_buying > expected_profit_selling:
            orders.append(Order('ORCHIDS', math.floor(best_ask-2), bid_volume))

        if expected_profit_selling > 0 and expected_profit_selling > expected_profit_buying:
            orders.append(Order('ORCHIDS', math.floor(best_bid+2), ask_volume )) 
        

        return orders, conversion
    
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

        strawberries_book_spread  = strawberries_ask - strawberries_bid

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



        nav = 4 * chocolates_mid_price + 6 *strawberries_mid_price + roses_mid_price
        spread_basket = basket_mid_price - nav

        spread_roses = roses_mid_price - 1.3427 * chocolates_mid_price

        spread_chocolates = basket_mid_price - 5.7223 * chocolates_mid_price

        self.spreads_basket.append(spread_basket)
        self.spreads_roses.append(spread_roses)
        self.spreads_chocolates.append(spread_chocolates)

        

        if len(self.mid_prices_history['GIFT_BASKET']) > 1:
            #intercept, intercept_std, betas, betas_std = self.linear_regression(gift_basket_prices, chocolates_prices, strawberries_prices, roses_prices)
            intercept = 165.3320184325068
            intercept_std = 72.08630428543141
            betas = [3.84317626, 6.17179092, 1.05264383]

            #intercept = -421.4553
            #intercept_std = 47.103
            #betas = [3.8385 , 6.3047, 1.0586]
            
            
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
                
            #if z_score > 2: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1.2:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                qt_basket = min(abs(sell_volume_basket), basket_bid_vol) 
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol)) 
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol)) 
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol)) 

                #n_tradable = min(qt_basket, math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries), math.floor(qt_roses/beta_roses ))
                #print(n_tradable)
                n_tradable = 1
                if n_tradable > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_bid, sell_volume_basket))
                   
                    
            #elif z_score < -2: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            if spread < intercept_std * (-1.2):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol)) 
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol) 
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol) 
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol) 

                #n_tradable = min(qt_basket, math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries), math.floor(qt_roses / beta_roses ))
                #print(n_tradable)
                n_tradable = 1
                if n_tradable > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_ask, buy_volume_basket))
                   
        if len(self.mid_prices_history['ROSES']) > 1:
            #intercept, intercept_std, betas, betas_std = self.linear_regression(gift_basket_prices, chocolates_prices, strawberries_prices, roses_prices)
            intercept = 24820
            intercept_std = 163.617
            beta_roses = 3.1629

            #intercept = 32260
            #intercept_std = 86.827
            #beta_roses = 2.6492

            nav = beta_roses * roses_mid_price + intercept
            #print('regression!')
            #print(beta_chocolates)
            #print(beta_strawberries)
            #print(beta_roses)
            #print(intercept)
            spread = basket_mid_price - nav 
            #print('spread', spread)
            self.spreads_basket.append(spread)
            mean = statistics.mean(self.spreads_basket)
            std = statistics.stdev(self.spreads_basket)

            z_score = (spread - mean) / std
  
            #print('Z-SCORE: ',z_score)

            #print(intercept_std*1.2)
            
            
            #if z_score > -0.4 and z_score < 0.4 and current_position_all != 0:
            if spread < intercept_std * 0.1 and spread > intercept_std * (-0.1) and position_roses != 0:
                
                if position_roses > 0:
                    roses_orders.append(Order('ROSES', roses_bid, -max(min(position_roses, roses_bid_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_2, -max(min(position_roses-roses_bid_vol, roses_bid_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_bid_3, -max(min(position_roses-roses_bid_vol-roses_bid_vol_2, roses_bid_vol_3),0)))
                elif position_roses < 0:
                    roses_orders.append(Order('ROSES', roses_ask, max(min(-position_roses, -roses_ask_vol),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_2, max(min(-position_roses+roses_ask_vol, -roses_ask_vol_2),0)))
                    #roses_orders.append(Order('ROSES', roses_ask_3, max(min(-position_roses+roses_ask_vol+roses_ask_vol_2, -roses_ask_vol_3),0)))            
            
            #if z_score > 2: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1.2:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                qt_basket = min(abs(sell_volume_basket), basket_bid_vol) 
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol)) 
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol)) 
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol)) 

                #n_tradable = min(qt_basket, math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries), math.floor(qt_roses/beta_roses ))
                n_tradable = 1

                if n_tradable > 0:
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid, sell_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_bid, - min(abs(n_tradable), basket_bid_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_2, - max(min(abs(n_tradable)-basket_bid, basket_bid_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_bid_3, - max(min(abs(n_tradable)-basket_bid_vol-basket_bid_vol_2, basket_bid_vol_3), 0)))
                    
                    roses_orders.append(Order('ROSES', roses_ask, buy_volume_roses))
                  
                    
            #elif z_score < -2: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            if spread < intercept_std * (-1.2):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol)) 
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol) 
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol) 
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol) 

                #n_tradable = min(qt_basket, math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries), math.floor(qt_roses / beta_roses ))
                n_tradable = 1
                print(n_tradable)
                if n_tradable > 0:
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask, buy_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_ask, min(n_tradable, -basket_ask_vol)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_2, max(min(n_tradable+basket_ask_vol, -basket_ask_vol_2), 0)))
                    #basket_orders.append(Order('GIFT_BASKET', basket_ask_3, max(min(n_tradable+basket_ask_vol+basket_ask_vol_2, -basket_ask_vol_3), 0)))
                    
                    roses_orders.append(Order('ROSES', roses_bid, sell_volume_roses))

        if len(self.mid_prices_history['STRAWBERRIES']) >1:
            #rolling_spreads = self.spreads_chocolates[-10:]
            #rolling_spread = statistics.mean(rolling_spreads)
            #print('regression!')
            #print(beta_chocolates)
            #print(beta_strawberries)
            #print(beta_roses)
            #print(intercept)
            #mean = statistics.mean(self.spreads_chocolates)
            #std = statistics.stdev(self.spreads_chocolates)

            #z_score = (rolling_spread - mean) / std
  
            #print('Z-SCORE: ',z_score)

            #print(intercept_std*1.2)

            intercept = 26800
            intercept_std = 441.474
            #beta_chocolates = 5.2957
            beta_strawberries = 10.9044

            #intercept= -433.5429
            #intercept_std = 267.348
            #beta_strawberries = 17.6479

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
            if spread > intercept_std * 1.2:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                #qt_basket = min(abs(sell_volume_basket), basket_bid_vol) 
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol)) 
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol)) 
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol)) 

                #n_tradable = min(math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries))
                n_tradable = 1

                if n_tradable > 0:
                    
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, buy_volume_strawberries))
                   
                    
            #elif z_score < -1: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            elif spread < intercept_std * (-1.2):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol)) 
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol) 
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol) 
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol) 

                #n_tradable = min(math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries))
                n_tradable = 1
                if n_tradable > 0:
                    
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid, sell_volume_strawberries))

        if len(self.spreads_chocolates) > 1:

            intercept = 3878.8146
            intercept_std = 41.821
            #beta_chocolates = 5.2957
            beta_chocolates = 1.3427

            #intercept = -1400.8233
            #intercept_std = 74.048
            #beta_chocolates = 5.2957
            #beta_chocolates = 2.0010

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
            if spread > intercept_std * 1.2:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                #qt_basket = min(abs(sell_volume_basket), basket_bid_vol) 
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol)) 
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol)) 
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol)) 

                #n_tradable = min(math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries))
                n_tradable = 1

                if n_tradable > 0:
                    
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, buy_volume_chocolates))
                    
                    
            #elif z_score < -1: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            elif spread < intercept_std * (-1.2):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol)) 
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol) 
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol) 
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol) 

                #n_tradable = min(math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries))
                n_tradable = 1
                if n_tradable > 0:
                    
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, sell_volume_chocolates))
                    

        while len(self.spreads_basket) > 100:
            self.spreads_basket.pop(0)
        while len(self.spreads_roses) > 100:
            self.spreads_roses.pop(0)
        while len(self.spreads_chocolates) > 100:
            self.spreads_chocolates.pop(0)
        while len(self.ratios_basket) > 100:
            self.ratios_basket.pop(0)


        return basket_orders, chocolates_orders, strawberries_orders, roses_orders

    def coconut_strategy(self, state : TradingState) -> List[Order]:
        """
        Buying and Selling based on last trade price vs mean price (ceiling floor version)
        """
        coconut_orders = []
        coconut_coupon_orders = []

        #### POSITIONS ####
        position_coconut = self.get_position('COCONUT', state)
        position_coconut_coupon = self.get_position('COCONUT_COUPON', state)

        #### QUANTITIES WE ARE ALLOWED TO TRADE ####
        buy_volume_coconut = (self.POSITION_LIMITS['COCONUT'] - position_coconut) 
        sell_volume_coconut = (- self.POSITION_LIMITS['COCONUT'] - position_coconut) 
        buy_volume_coconut_coupon = (self.POSITION_LIMITS['COCONUT_COUPON'] - position_coconut_coupon) 
        sell_volume_coconut_coupon = (- self.POSITION_LIMITS['COCONUT_COUPON'] - position_coconut_coupon) 

        #### MID PRICES ####
        coconut_mid_price = self.get_mid_price('COCONUT', state)
        coconut_coupon_mid_price = self.get_mid_price('COCONUT_COUPON', state)

        #print('call price', coconut_coupon_mid_price)

        #### BIDS, ASKS, VOLUMES ####
        coconut_bid, coconut_bid_vol = self.get_best_bid('COCONUT', state)
        coconut_ask, coconut_ask_vol = self.get_best_ask('COCONUT', state)
        coconut_coupon_bid, coconut_coupon_bid_vol = self.get_best_bid('COCONUT_COUPON', state)
        coconut_coupon_ask, coconut_coupon_ask_vol = self.get_best_ask('COCONUT_COUPON', state)

        """
        if len(self.last_variances) == 0:
            self.last_variances.append(self.initial_last_variance)

        last_variance = self.last_variances[-1]
        
        if len(self.mid_prices_history['COCONUT']) < 2:
            return coconut_orders, coconut_coupon_orders
        
        last_return = (self.mid_prices_history['COCONUT'][-1] / self.mid_prices_history['COCONUT'][-2]) - 1
        self.coconuts_returns.append(last_return)

        next_variance, forecasted_volatility = self.forecast_volatility(last_variance, last_return, self.omega, self.alpha, self.beta)

        """

        implied_vol = self.implied_volatility(coconut_coupon_mid_price, coconut_mid_price, 10000, 246/250, initial_sigma=0.2, 
                                              tolerance=1e-5, max_iterations=100)
        
        #self.last_variances.append(next_variance)

        
        delta = self.black_scholes_delta(coconut_mid_price, 10000, 246/250, implied_vol, 0)

        #print('implied vol', implied_vol)
        #print('est vol', forecasted_volatility)
        #print('delta', delta)

        gamma = self.black_scholes_gamma(coconut_mid_price, 10000, 246/250, implied_vol, 0)
        #print('gamma',gamma)

        #portfolio_delta = delta * position_coconut_coupon + position_coconut WRONG
        #print('\n PORT DELTA', portfolio_delta)

        vol = 0.1612962156385183
        #vol = 0.1919
        bs_price = self.black_scholes_call(coconut_mid_price, 10000, 246/365, vol, r=0.0)
        #print('BS PRICE', bs_price)
        
        if bs_price > coconut_coupon_ask+2:
            coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_ask, buy_volume_coconut_coupon))
        elif bs_price < coconut_coupon_bid-2:
            coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_bid, sell_volume_coconut_coupon))

        #max vol 0.1697578300721943
        #min vol 0.14901861264929178

        #### GAMMA SCALPING ####
        """
        if position_coconut_coupon < 500:
            qt_coconut_coupons = min(buy_volume_coconut_coupon, -coconut_coupon_ask_vol)
            qt_coconut = min(abs(sell_volume_coconut), coconut_bid_vol)

            qt_trade = min(qt_coconut_coupons*delta, qt_coconut)

            coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_ask, int(math.floor(qt_trade/delta))))
            coconut_orders.append(Order('COCONUT', coconut_bid, -int(math.floor(qt_trade))))

            portfolio_delta = position_coconut_coupon * delta
            needed_buy_short = - portfolio_delta - position_coconut

        else:
            portfolio_delta = position_coconut_coupon * delta

            needed_buy_short = - portfolio_delta - position_coconut

            if needed_buy_short > 0:
                coconut_orders.append(Order('COCONUT', coconut_ask, int(math.floor(needed_buy_short))))
            elif needed_buy_short < 0:
                coconut_orders.append(Order('COCONUT', coconut_bid, int(math.floor(needed_buy_short))))

        print('delta port', portfolio_delta)
        print('position coconut',position_coconut)
        """

        #### PORTFOLIO DELTA ###
        """
        if portfolio_delta == 0:
            qt_coconut_coupons = min(buy_volume_coconut_coupon, -coconut_coupon_ask_vol)
            qt_coconut = min(abs(sell_volume_coconut), coconut_bid_vol)

            qt_trade = min(qt_coconut_coupons*delta, qt_coconut)

            coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_ask, int(math.floor(qt_trade/delta))))
            coconut_orders.append(Order('COCONUT', coconut_bid, -int(math.floor(qt_trade))))

        elif portfolio_delta > 0:
            qt_coconut_coupons = min(abs(sell_volume_coconut_coupon), coconut_coupon_bid_vol)
            qt_coconut = min(abs(sell_volume_coconut), coconut_bid_vol)

            coconut_orders.append(Order('COCONUT', coconut_bid, -int(math.floor((min(sell_volume_coconut, qt_coconut))))))

            #qt_coconut_coupons = min(buy_volume_coconut_coupon, -coconut_coupon_ask_vol)
            #qt_coconut = max(min(abs(sell_volume_coconut), coconut_bid_vol-portfolio_delta),0)

            #qt_trade = min(qt_coconut_coupons*delta, qt_coconut)

            #coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_ask, int(math.floor(qt_trade/delta))))
            #coconut_orders.append(Order('COCONUT', coconut_bid, -int(math.floor(qt_trade))))

        elif portfolio_delta < 0:
            qt_coconut_coupons = min(buy_volume_coconut_coupon, -coconut_coupon_ask_vol)
            qt_coconut = min(buy_volume_coconut, -coconut_ask_vol)

            coconut_orders.append(Order('COCONUT', coconut_ask, int(math.floor((min(abs(portfolio_delta), qt_coconut))))))

            #qt_coconut_coupons = min(buy_volume_coconut_coupon, -coconut_coupon_ask_vol)
            #qt_coconut = min(abs(sell_volume_coconut), coconut_bid_vol)

            #qt_trade = min(qt_coconut_coupons*delta, qt_coconut)

            #coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_ask, int(math.floor(qt_trade/delta))))
            #coconut_orders.append(Order('COCONUT', coconut_bid, -int(math.floor(qt_trade))))
        """
            
        #### VOLaTILITY STRATEGY BUT FORECAST NOT WORKING WELL YET ####
        """
        if implied_vol < forecasted_volatility:

            #if position_coconut_coupon < 0:
                #coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_ask, -position_coconut_coupon))


            qt_coconut_coupons = min(buy_volume_coconut_coupon, -coconut_coupon_ask_vol)
            qt_coconut = min(abs(sell_volume_coconut), coconut_bid_vol)

            qt_trade = min(qt_coconut_coupons*delta, qt_coconut)

            coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_ask, int(math.floor(qt_trade/delta))))
            coconut_orders.append(Order('COCONUT', coconut_bid, -int(math.floor(qt_trade))))

        elif implied_vol > forecasted_volatility:

            #if position_coconut_coupon > 0:
                #coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_bid, -position_coconut_coupon))

            qt_coconut_coupons = min(abs(sell_volume_coconut_coupon), coconut_coupon_bid_vol)
            qt_coconut = min(buy_volume_coconut, -coconut_ask_vol)

            qt_trade = min(qt_coconut_coupons*delta, qt_coconut)

            coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_bid, -int(math.floor(qt_trade/delta))))
            coconut_orders.append(Order('COCONUT', coconut_ask, int(math.floor(qt_trade))))
        """

        #### ARIMA STRATEGY ####
        """
        if len(self.mid_p_diff_history['COCONUT']) >= 6:
            AR_L1 = self.mid_p_diff_history['COCONUT'][-1]
            AR_L2 = self.mid_p_diff_history['COCONUT'][-2]
            AR_L3 = self.mid_p_diff_history['COCONUT'][-3]
            AR_L4 = self.mid_p_diff_history['COCONUT'][-4]
            AR_L5 = self.mid_p_diff_history['COCONUT'][-5]
            AR_L6 = self.mid_p_diff_history['COCONUT'][-6]
            AR_L7 = self.mid_p_diff_history['COCONUT'][-7]
            AR_L8 = self.mid_p_diff_history['COCONUT'][-8]
            AR_L9 = self.mid_p_diff_history['COCONUT'][-9]
            AR_L10 = self.mid_p_diff_history['COCONUT'][-10]
        
        if len(self.forecasted_diff_history['COCONUT']) > 0:
            forecasted_error = self.forecasted_diff_history['COCONUT'][-1] - self.mid_p_diff_history['COCONUT'][-1]
            self.errors_history['COCONUT'].append(forecasted_error)

        if len(self.errors_history['COCONUT']) < 2:
            
            self.errors_history['COCONUT'].extend([-0.969329, 0.450927, -1.473150])
        else:
            MA_L1 = self.errors_history['COCONUT'][-1]
            MA_L2 = self.errors_history['COCONUT'][-2]
            MA_L3 = self.errors_history['COCONUT'][-3]
        
    
        forecasted_diff = (AR_L1 * -0.7060) + (AR_L2 * 0.5870) + (AR_L3 * 0.8943) + (AR_L4 * 0.0332)
        + (AR_L5 * -0.0005) + (AR_L6 * -0.0052) + (AR_L7 * -0.0102) + (AR_L8 * -0.0058) + (AR_L9 * 0.0050) + (AR_L10 * -0.0013)
        + (MA_L1 * 0.6745) + (MA_L2 * -0.6023) + (MA_L3 * -0.8658)


        self.forecasted_diff_history['COCONUT'].append(forecasted_diff)

        forecasted_price = coconut_mid_price + forecasted_diff  

        print('forecast:', forecasted_price)
        print('mid', coconut_mid_price)

        #play with diff comb
        if forecasted_diff > 0:
            coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_ask, buy_volume_coconut_coupon))
            
        elif forecasted_diff < 0:
            coconut_coupon_orders.append(Order('COCONUT_COUPON', coconut_coupon_bid, sell_volume_coconut_coupon))
        """   

        return coconut_orders, coconut_coupon_orders


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [], 'GIFT_BASKET' : [], 'CHOCOLATE' : [], 'STRAWBERRIES' : [], 'ROSES' : [], 'COCONUT': [], 'COCONUT_COUPON': []}

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
            while len(self.pnl_tracker[product]) > 10:
                self.pnl_tracker[product].pop(0)
            while len(self.forecasted_diff_history[product]) > 10:
                self.forecasted_diff_history[product].pop(0)
            while len(self.errors_history[product]) > 10:
                self.errors_history[product].pop(0)
            while len(self.last_variances) > 10:
                self.last_variances.pop(0)
            while len(self.coconuts_returns) > 10:
                self.coconuts_returns.pop(0)


        
        """
        # AMETHYSTS STRATEGY
        try:
            result['AMETHYSTS'] = self.amethysts_strategy(state)
        except Exception as e:
            print("Error in AMETHYSTS strategy")
            print(e)
        
        # STARFRUIT STRATEGY
        try:
            result['STARFRUIT'] = self.starfruit_strategy(state)
        except Exception as e:
            print("Error in STARFRUIT strategy")
            print(e)
        
        
        # ORCHIDS STRATEGY
        
        try:
            result['ORCHIDS'] = self.orchids_strategy2(state)[0]
        except Exception as e:
            print("Error in ORCHIDS strategy")
            print(e)
        
        
        # BASKET STRATEGY
        try:
            result['GIFT_BASKET'], result['CHOCOLATE'], result['STRAWBERRIES'], result['ROSES'] = self.basket_strategy(state)
        
        except Exception as e:
            print(e)
        """

        # COCONUT AND COCONUT_COUPON STRATEGY
        
        result['COCONUT'], result['COCONUT_COUPON'] = self.coconut_strategy(state)
        

             
        traderData = "SAMPLE" 
        
		# Sample conversion request. Check more details below. 
        #conversions = self.orchids_strategy2(state)[1]
        conversions = 0

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData