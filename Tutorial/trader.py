from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math



class Trader:

    PRODUCTS = [
    'AMETHYSTS',
    'STARFRUIT',
    ]

    DEFAULT_PRICES = {
        'AMETHYSTS' : 10000,
        'STARFRUIT' : 10000,
    }

    POSITION_LIMITS = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
    }

    def __init__(self) -> None:

        self.ema_prices = dict()
        for product in self.PRODUCTS:
            self.ema_prices[product] = None
        self.ema_param = 0.5

    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    

    def get_mid_price(self, product, state : TradingState):

        if product not in state.order_depths:
            return None

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return None
        
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return None
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
    
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

        print(self.ema_prices)

    

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
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] - 1), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 1), ask_volume))
        
        if position_amethysts > 0:
            # Long position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS'] - 2), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS']), ask_volume))

        if position_amethysts < 0:
            # Short position
            orders.append(Order('AMETHYSTS', math.floor(self.ema_prices['AMETHYSTS']), bid_volume))
            orders.append(Order('AMETHYSTS', math.ceil(self.ema_prices['AMETHYSTS'] + 2), ask_volume))

        print(orders)

        return orders
        

    def starfruit_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of starfruit.
        """
        orders = []
            
        return orders


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS' : [], 'STARFRUIT' : []}

        self.update_ema_prices(state)
        

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
                
        traderData = "SAMPLE" 
        
		# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData