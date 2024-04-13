######### THIS IS A SOLUTION FOR A VERY SIMILAR, ALMOST IDENTICAL QUESTION FROM LAST YEAR ###########

import copy
from itertools import permutations


EXCHANGE_MATRIX = [
    [1, 0.48, 1.52, 0.71],
    [2.05, 1, 3.26, 1.56],
    [0.64, 0.30, 1, 0.46],
    [1.41, 0.61, 2.08, 1],
]

# Maximum amount of each asset possible after trades

MAX_AMOUNT = [0, 0, 0, 2_000_000]
BEST_ROUTE = [[], [], [], []]

# There are 5 trades
for _ in range(5):
    NEW_MAX_AMOUNT = copy.deepcopy(MAX_AMOUNT)
    NEW_BEST_ROUTE = copy.deepcopy(BEST_ROUTE)

    for target_product in range(4):
        for origin_product in range(4):
            quantity_target = MAX_AMOUNT[origin_product] * EXCHANGE_MATRIX[origin_product][target_product]
            if quantity_target > NEW_MAX_AMOUNT[target_product]:
                NEW_MAX_AMOUNT[target_product] = quantity_target
                NEW_BEST_ROUTE[target_product] = BEST_ROUTE[origin_product] + [(origin_product, target_product)]

    MAX_AMOUNT = NEW_MAX_AMOUNT
    BEST_ROUTE = NEW_BEST_ROUTE

print(MAX_AMOUNT)
print(BEST_ROUTE)




def maximize_exchange_with_path(exchange_matrix, current_amount, current_currency, transaction_count, max_transactions, start_currency, path):
    # Base case: If we have reached the maximum number of transactions
    if transaction_count == max_transactions:
        # We must end with the start currency (last row currency)
        if current_currency != start_currency:
            return -float('inf'), []  # Impossible path, return negative infinity and empty path
        return current_amount, path
    
    # Recursive case: Try all possible transactions
    best_amount = -float('inf')
    best_path = []
    for next_currency in range(len(exchange_matrix)):
        # Calculate the amount after exchanging to next_currency
        new_amount = current_amount * exchange_matrix[current_currency][next_currency]
        # Recurse to the next transaction
        amount, subpath = maximize_exchange_with_path(
            exchange_matrix, new_amount, next_currency, transaction_count + 1, max_transactions, start_currency, path + [next_currency]
        )
        # Keep track of the best amount we can achieve
        if amount > best_amount:
            best_amount = amount
            best_path = subpath
    
    return best_amount, best_path

# Constants and initial setup
EXCHANGE_MATRIX = [
    [1, 0.48, 1.52, 0.71],
    [2.05, 1, 3.26, 1.56],
    [0.64, 0.30, 1, 0.46],
    [1.41, 0.61, 2.08, 1],
]
START_AMOUNT = 2_000_000
MAX_TRANSACTIONS = 5
START_CURRENCY = 3  # Last row is the start and end currency

# Running the function to find the best transaction sequence
max_end_amount, transaction_path = maximize_exchange_with_path(
    EXCHANGE_MATRIX, START_AMOUNT, START_CURRENCY, 0, MAX_TRANSACTIONS, START_CURRENCY, [START_CURRENCY]
)

print("Maximum Ending Amount:", max_end_amount)
print("Transaction Path:", transaction_path)


import itertools

def maximize_exchange_with_itertools(exchange_matrix, start_amount, start_currency, max_transactions):
    best_amount = -float('inf')
    best_path = []
    num_currencies = len(exchange_matrix)

    # Generate all possible paths of length max_transactions
    all_paths = itertools.product(range(num_currencies), repeat=max_transactions)

    # Evaluate each path
    for path in all_paths:
        current_amount = start_amount
        current_currency = start_currency
        valid_path = True
        transaction_path = [start_currency]

        for next_currency in path:
            if current_currency == next_currency:
                valid_path = False
                break
            new_amount = current_amount * exchange_matrix[current_currency][next_currency]
            current_currency = next_currency
            current_amount = new_amount
            transaction_path.append(current_currency)

        # Must end at the start currency
        if valid_path and current_currency == start_currency and current_amount > best_amount:
            best_amount = current_amount
            best_path = transaction_path

    return best_amount, best_path

# Constants and initial setup
EXCHANGE_MATRIX = [
    [1, 0.48, 1.52, 0.71],
    [2.05, 1, 3.26, 1.56],
    [0.64, 0.30, 1, 0.46],
    [1.41, 0.61, 2.08, 1],
]
START_AMOUNT = 2_000_000
MAX_TRANSACTIONS = 5
START_CURRENCY = 3  # Last row is the start and end currency

# Running the function to find the best transaction sequence
max_end_amount, transaction_path = maximize_exchange_with_itertools(
    EXCHANGE_MATRIX, START_AMOUNT, START_CURRENCY, MAX_TRANSACTIONS
)

print("Maximum Ending Amount:", max_end_amount)
print("Transaction Path:", transaction_path)
