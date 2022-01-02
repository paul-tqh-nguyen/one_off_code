
###########
# Imports #
###########

# TODO make sure these are all used
import sys
from typing import List, Tuple, Dict, Optional, Generator, Iterable, Hashable, Any

# TODO abstract everything out below

###################
# Utility Classes #
###################

class MutableOrderedDictionary:
    '''
    Maintains a mutable custom order rather than
    insertion order (as is done by collections.OrderedDict).
    '''
    
    def __init__(self, key_value_pairs: Iterable[Tuple[Hashable, Any]] = []):
        self.ordered_keys: List[Hashable] = []
        self.mapping: Dict[Hashable, Tuple[Any, int]] = dict()
        for k, v in key_value_pairs:
            self[k] = v
    
    def keys(self) -> List[Hashable]:
        return self.ordered_keys
    
    def values(self) -> Generator[Any, None, None]:
        return (self[k] for k in self.ordered_keys)

    def items(self) -> Generator[Tuple[Hashable, Any], None, None]:
        return ((k, self[k]) for k in self.ordered_keys)

    def __repr__(self) -> str:
        items_string = ', '.join(map(str, self.items()))
        return f'{self.__class__.__name__}([{items_string}])'
    
    def clear(self) -> None:
        self.ordered_keys.clear()
        self.mapping.clear()
        return

    def replace(self, k: Hashable, v: Any, position: int) -> None:
        # TODO is this needed?
        old_key = self.ordered_keys[position]
        del self.mapping[old_key]
        self.mapping[k] = v
        self.ordered_keys[position] = k
        return

    def __contains__(self, key: Hashable) -> bool:
        return key in self.mapping
    
    def __setitem__(self, k: Hashable, v: Any) -> None:
        position = len(self.ordered_keys)
        self.mapping[k] = (v, position)
        self.ordered_keys.append(k) 
        return

    def __getitem__(self, k: Hashable) -> Any:
        return self.mapping[k][0]
 
################################
# Application-Specific Classes #
################################

class Order:
    def __init__(self, order_id: str, price: float, size: int):
        self.order_id = order_id
        self.price = price
        self.size = size

    def reduce_size(self, amount: int) -> None:
        self.size -= amount
        return

    def __repr__(self) -> str:
        attributes_string = ', '.join(f'{k}={repr(self.__dict__[k])}' for k in sorted(self.__dict__.keys()))
        return f'{self.__class__.__name__}({attributes_string})'

class Bid(Order):

    def __lt__(self, other):
        return self.price > other.price


class Ask(Order):

    def __lt__(self, other):
        return self.price < other.price


###########
# Globals #
###########

TARGET_SIZE: Optional[int] = None
BIDS: Dict[str, Bid] = dict()
ASKS: Dict[str, Ask] = dict()

BIDS_SIZE_SUM: int = 0
ASKS_SIZE_SUM: int = 0

LAST_BUY_ORDER_IDS = MutableOrderedDictionary()
LAST_SELL_ORDER_IDS = MutableOrderedDictionary()

LAST_BUY_AMOUNT: Optional[float] = None
LAST_SELL_AMOUNT: Optional[float] = None

################################
# Message Processing Utilities #
################################
    
def calculate_order_amount(
        orders: Dict[str, Order],
        ids_container: MutableOrderedDictionary,
) -> float:
    '''
    Assumes TARGET_SIZE <= sum(e.size for e in orders.values).
    Clears ids_container and adds relevant order ids to it.
    '''
    total_amount = 0
    remaining_shares = TARGET_SIZE
    ids_container.clear()
    # TODO this is O(n log n) open every print, can we do better?
    for order_id, order in sorted(orders.items(), key=lambda pair: pair[1]):
        ids_container[order_id] = order
        if order.size < remaining_shares:
            total_amount += order.size * order.price
            remaining_shares -= order.size
        else:
            total_amount += remaining_shares * order.price
            break
    return total_amount

def recalculate_order_amount_via_last_orders(orders: MutableOrderedDictionary) -> float:
    total_amount = 0
    remaining_shares = TARGET_SIZE
    for order in orders.values():
        if order.size < remaining_shares:
            total_amount += order.size * order.price
            remaining_shares -= order.size
        else:
            total_amount += remaining_shares * order.price
            break
    return total_amount

def process_reduce_message(timestamp: str, order_id: str, size: str) -> None:
    size = int(size)
    if order_id in BIDS:
        BIDS[order_id].reduce_size(size)
        global BIDS_SIZE_SUM
        BIDS_SIZE_SUM -= size
        global LAST_SELL_AMOUNT
        if BIDS_SIZE_SUM < TARGET_SIZE:
            if LAST_SELL_AMOUNT is not None:
                LAST_SELL_AMOUNT = None
                print(f'{timestamp} S NA')
        elif order_id in LAST_SELL_ORDER_IDS:
            # TODO is this optimal?
            size_sum_LAST_SELL_ORDER_IDS = sum(order.size for order in LAST_SELL_ORDER_IDS.values())
            if size_sum_LAST_SELL_ORDER_IDS >= TARGET_SIZE:
                updated_LAST_SELL_AMOUNT = recalculate_order_amount_via_last_orders(LAST_SELL_ORDER_IDS)
            else:
                updated_LAST_SELL_AMOUNT = calculate_order_amount(BIDS, LAST_SELL_ORDER_IDS)
            if updated_LAST_SELL_AMOUNT != LAST_SELL_AMOUNT:
                LAST_SELL_AMOUNT = updated_LAST_SELL_AMOUNT
                print(f'{timestamp} S {LAST_SELL_AMOUNT:.2f}')
    elif order_id in ASKS:
        ASKS[order_id].reduce_size(size)
        global ASKS_SIZE_SUM
        ASKS_SIZE_SUM -= size
        global LAST_BUY_AMOUNT
        if ASKS_SIZE_SUM < TARGET_SIZE:
            if LAST_BUY_AMOUNT is not None:
                LAST_BUY_AMOUNT = None
                print(f'{timestamp} B NA')
        elif order_id in LAST_BUY_ORDER_IDS:
            # TODO is this optimal?
            size_sum_LAST_BUY_ORDER_IDS = sum(order.size for order in LAST_BUY_ORDER_IDS.values())
            if size_sum_LAST_BUY_ORDER_IDS >= TARGET_SIZE:
                updated_LAST_BUY_AMOUNT = recalculate_order_amount_via_last_orders(LAST_BUY_ORDER_IDS)
            else:
                updated_LAST_BUY_AMOUNT = calculate_order_amount(ASKS, LAST_BUY_ORDER_IDS)
            if updated_LAST_BUY_AMOUNT != LAST_BUY_AMOUNT:
                LAST_BUY_AMOUNT = updated_LAST_BUY_AMOUNT
                print(f'{timestamp} B {LAST_BUY_AMOUNT:.2f}')
    else:
        raise RuntimeError(f'Unknown order id {order_id}.')
    return

def process_add_message(timestamp: str, order_id: str, side: str, price: str, size: str) -> None:
    price = float(price)
    size = int(size)
    if side == 'B':
        BIDS[order_id] = Bid(order_id, price, size)
        global BIDS_SIZE_SUM
        BIDS_SIZE_SUM += size
        if BIDS_SIZE_SUM >= TARGET_SIZE:
            global LAST_SELL_AMOUNT
            updated_LAST_SELL_AMOUNT = calculate_order_amount(BIDS, LAST_SELL_ORDER_IDS)
            # TODO is there a smarter way to avoid this check?
            if updated_LAST_SELL_AMOUNT != LAST_SELL_AMOUNT:
                LAST_SELL_AMOUNT = updated_LAST_SELL_AMOUNT
                print(f'{timestamp} S {LAST_SELL_AMOUNT:.2f}')
    elif side == 'S':
        ASKS[order_id] = Ask(order_id, price, size)
        global ASKS_SIZE_SUM
        ASKS_SIZE_SUM += size
        if ASKS_SIZE_SUM >= TARGET_SIZE:
            global LAST_BUY_AMOUNT
            updated_LAST_BUY_AMOUNT = calculate_order_amount(ASKS, LAST_BUY_ORDER_IDS)
            # TODO is there a smarter way to avoid this check?
            if updated_LAST_BUY_AMOUNT != LAST_BUY_AMOUNT:
                LAST_BUY_AMOUNT = updated_LAST_BUY_AMOUNT
                print(f'{timestamp} B {LAST_BUY_AMOUNT:.2f}')
    else:
        raise RuntimeError(f'Unknown side type {repr(side)}.')
    return

def process_message(message: str) -> None:
    timestamp, message_type, *remaining_message_data = message.split()
    if message_type == 'A':
        process_add_message(timestamp, *remaining_message_data)
    elif message_type == 'R':
        process_reduce_message(timestamp, *remaining_message_data)
    else:
        raise RuntimeError(f'Unknown message type {repr(message_type)}.')
    return

##########
# Driver #
##########

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        raise RuntimeError(f'Expected one command line arg, got {len(args)}.')
    [target_size_string] = args
    if not target_size_string.isdigit():
        raise TypeError(f'Target size command line arg must be interpretable as an integer, got {repr(target_size_string)}.')
    
    TARGET_SIZE = int(target_size_string)
    for message in sys.stdin:
        process_message(message)
