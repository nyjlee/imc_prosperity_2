########## OLD STRATEGIES HERE ##########

from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math


def amethysts_strategy_1(self, state : TradingState) -> List[Order]:
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




def amethysts_strategy_2(self, state : TradingState) -> List[Order]:
    """
    Returns a list of orders with trades of amethysts.

    OBS: terrible performance
    """
    orders = []

    position_amethysts = self.get_position('AMETHYSTS', state)

    bid_volume = self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
    ask_volume = - self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts

    support = 9996
    resistance = 1004

    if position_amethysts == 0:
        # Not long nor short
        orders.append(Order('AMETHYSTS', support, bid_volume))
        orders.append(Order('AMETHYSTS', resistance, ask_volume))

    if position_amethysts > 0:
        # Long position
        orders.append(Order('AMETHYSTS', resistance, ask_volume))

    if position_amethysts < 0:
        # Short position
        orders.append(Order('AMETHYSTS', support, bid_volume))

    print(orders)

    return orders


