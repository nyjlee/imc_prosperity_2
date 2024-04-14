from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade
import math
import pandas as pd
import numpy as np


class Trader:

    PRODUCTS = [
    'AMETHYSTS',
    'STARFRUIT',
    'ORCHIDS',
    ]

    DEFAULT_PRICES = {
        'AMETHYSTS' : 10000,
        'STARFRUIT' : 5000,
        'ORCHIDS': 1000,
    }

    POSITION_LIMITS = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
        'ORCHIDS': 100,
    }

    prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}
    mid_prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}
    mid_p_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}
    p_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}
    errors_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}
    forecasted_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}

    current_signal = {"AMETHYSTS": "", "STARFRUIT": "None", "ORCHIDS": "None"}


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
            
            self.errors_history['STARFRUIT'].extend([-4.218699, -0.945694, 0.298280])

            #self.errors_history['STARFRUIT'].extend([-3.258368, -3.353484, -3.593285])
      
        else:
            MA_L1 = self.errors_history['STARFRUIT'][-1]
            MA_L2 = self.errors_history['STARFRUIT'][-2]
            MA_L3 = self.errors_history['STARFRUIT'][-3]
        
        #use this!
        #forecasted_diff = (AR_L1 * -1.1102) + (AR_L2 * -0.7276) + (AR_L3 * -0.0854) + (AR_L4 * -0.0674)
        #+ (AR_L5 * -0.0437) + (AR_L6 * -0.0176)+ (MA_L1 *  0.4021) + (MA_L2 * -0.0587) + (MA_L3 * -0.4357)

        #new data
        forecasted_diff = (AR_L1 * -1.4799) + (AR_L2 * -0.8168) + (AR_L3 * -0.0868) + (AR_L4 * -0.0693)
        + (AR_L5 * -0.0492) + (AR_L6 * -0.0221)+ (MA_L1 *  0.7712) + (MA_L2 * -0.2324) + (MA_L3 * -0.4996)

        #price instead of mid price
        #forecasted_diff = (AR_L1 * -0.5038) + (AR_L2 * -0.4052) + (AR_L3 * -0.0110) + (AR_L4 * -0.0272)
        #+ (AR_L5 * -0.0209) + (AR_L6 * -0.0107)+ (MA_L1 *  -0.2193) + (MA_L2 * 0.0157) + (MA_L3 * -0.2858)

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

        buy_price_south = ask_price_south + transport_fees + import_tariff
        sell_price_south = bid_price_south - transport_fees - export_tariff

        expected_profit_buying = 0
        expected_profit_selling = 0

        if position_orchids != 0:
            conversion = position_orchids
        else:
            conversion = 0

        if best_ask < sell_price_south:
            #orders.append(Order('ORCHIDS', math.floor(best_ask), bid_volume))
            expected_profit_buying = sell_price_south - best_ask
        
        if best_bid > buy_price_south:
            #orders.append(Order('ORCHIDS', math.floor(best_bid), ask_volume))
            expected_profit_selling = best_bid - buy_price_south

        if expected_profit_buying > 5 and expected_profit_buying > expected_profit_selling:
            orders.append(Order('ORCHIDS', math.floor(best_ask), bid_volume))

        if expected_profit_selling > 5 and expected_profit_selling > expected_profit_buying:
            orders.append(Order('ORCHIDS', math.floor(best_bid), ask_volume))





        return orders, conversion
    

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : []}

        self.update_ema_prices(state)

        # PRICE HISTORY
        self.update_prices_history(state.own_trades, state.market_trades)
        self.update_mid_prices_history(state)
        #self.update_diff_history(self.mid_prices_history)
        self.update_diff_history(self.mid_p_diff_history, self.mid_prices_history)
        self.update_diff_history(self.p_diff_history, self.prices_history)
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
            while len(self.pnl_tracker[product]) > 10:
                self.pnl_tracker[product].pop(0)
            while len(self.forecasted_diff_history[product]) > 10:
                self.forecasted_diff_history[product].pop(0)
            while len(self.errors_history[product]) > 10:
                self.errors_history[product].pop(0)
        


        # AMETHYSTS STRATEGY
        """
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
        
        """
        # ORCHIDS STRATEGY
        try:
            result['ORCHIDS'] = self.orchids_strategy2(state)[0]
        except Exception as e:
            print("Error in ORCHIDS strategy")
            print(e)
        
             
        traderData = "SAMPLE" 
        
		# Sample conversion request. Check more details below. 
        conversions = self.orchids_strategy2(state)[1]

        return result, conversions, traderData