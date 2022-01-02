###########
# Imports #
###########

# TODO make sure these are all used
import sys
import heapq
import math
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Optional,
    Generator,
    Iterable,
    Hashable,
    Any,
)

# TODO abstract everything out below
# TODO remove assertions
# TODO sweep all doc strings

#######################
# Debugging Utilities #
#######################

# TODO remove this

import io
from typing import Generator
from contextlib import contextmanager
@contextmanager
def std_out(stream: io.TextIOWrapper) -> Generator:
    import sys
    original_std_out = sys.stdout
    sys.stdout = stream
    yield
    sys.stdout = original_std_out
    return

TRACE_INDENT_LEVEL = 0
TRACE_INDENTATION = '    '
TRACE_VALUE_SIZE_LIMIT = 2000
from typing import Callable
def trace(func: Callable) -> Callable:
    from inspect import signature
    import sys
    import random
    def human_readable_value(value) -> str:
        readable_value = repr(value)
        if len(readable_value) > TRACE_VALUE_SIZE_LIMIT:
            readable_value = readable_value[:TRACE_VALUE_SIZE_LIMIT]+'...'
        return readable_value
    def decorating_function(*args, **kwargs):
        arg_values_string = ', '.join((f'{param_name}={human_readable_value(value)}' for param_name, value in signature(func).bind(*args, **kwargs).arguments.items()))
        probably_unique_id = random.randint(10,99)
        global TRACE_INDENT_LEVEL, TRACE_INDENTATION
        entry_line = f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] {func.__qualname__}({arg_values_string})'
        with std_out(sys.__stdout__):
            print()
            print(entry_line)
            print()
        TRACE_INDENT_LEVEL += 1
        result = func(*args, **kwargs)
        TRACE_INDENT_LEVEL -= 1
        with std_out(sys.__stdout__):
            print()
            print(f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] returned {human_readable_value(result)}')
            print()
        return result
    return decorating_function

################################
# Application-Specific Classes #
################################


class Order:
    def __init__(self, order_id: str, price: float, size: int):
        self.order_id = order_id
        self.price = price
        self.size = size
        # TODO this is a hack/workaround for the fact
        # that Python's heapq doesn't support a max heap.
        # TODO this also forces each instance to only be
        # part of one heap at a time, which is fine for
        # this application but is not ideal.
        self.is_max_heap_friendly = False
        return

    def enable_max_heap_compatibility(self) -> None:
        assert self.is_max_heap_friendly == False
        self.is_max_heap_friendly = True
        return

    def disable_max_heap_compatibility(self) -> None:
        # assert self.is_max_heap_friendly == True # not necessarily true for a pushpop
        self.is_max_heap_friendly = False
        return

    def reduce_size(self, amount: int) -> None:
        assert self.size >= amount
        self.size -= amount
        return

    def __repr__(self) -> str:
        attributes_string = ", ".join(
            f"{k}={repr(self.__dict__[k])}" for k in sorted(self.__dict__.keys())
        )
        return f"{self.__class__.__name__}({attributes_string})"


class Bid(Order):
    def __lt__(self, other) -> bool:
        return (
            self.price < other.price
            if self.is_max_heap_friendly
            else self.price > other.price
        )


class Ask(Order):
    def __lt__(self, other) -> bool:
        return (
            self.price > other.price
            if self.is_max_heap_friendly
            else self.price < other.price
        )


class SimpleMinHeapOfOrders:

    """
    "Min" means least costly towards us, i.e. cheapest ask and highest bid.
    "Simple" means it does not track the total sum of the orders' prices or sizes.
    """

    def __init__(self, orders: Iterable[Order] = ()):
        """
        O(n) where n = len(orders).
        """
        self.heap: List[Order] = list(orders)
        heapq.heapify(self.heap)
        self.order_id_to_order: Dict[str, Order] = {order.order_id: order for order in self.heap}
        return

    def __repr__(self) -> str:
        attributes_string = ", ".join(
            f"{k}={repr(self.__dict__[k])}" for k in sorted(self.__dict__.keys())
        )
        return f"{self.__class__.__name__}({attributes_string})"

    def __contains__(self, item: Union[Order, str]) -> bool:
        """O(1)."""
        assert isinstance(item, (Order, str))
        order_id = item.order_id if isinstance(item, Order) else item
        return order_id in self.order_id_to_order

    def __len__(self) -> int:
        """O(1)."""
        return len(self.order_id_to_order)

    def reduce_order_size(self, order_id: str, reduction_amount: int) -> Order:
        """
        O(1).

        Leaves invalid orders (i.e. orders with size <= 0) in self.heap,
        but removes them from self.order_id_to_order.

        An invalid order will be lazily removed from self.heap elsewhere since
        doing so eagerly (i.e. here) is O(n) and doing so lazily will be O(log n)
        via self.pop or self.peek.
        """
        order = self.order_id_to_order.get(order_id)
        if order is None:  # TODO remove if not needed
            raise KeyError("Unknown order id {order_id}.")
        assert order.size >= reduction_amount
        if order.size == reduction_amount:
            del self.order_id_to_order[order_id]
        elif order.size < reduction_amount:  # TODO remove if not needed
            raise ValueError("Cannot reduce {order} by {reduction_amount}.")
        order.reduce_size(reduction_amount)
        return order

    def push(self, order: Order) -> None:
        """O(log n) where n = len(self.heap)."""
        assert order.size > 0
        assert order.order_id not in self.order_id_to_order
        self.order_id_to_order[order.order_id] = order
        heapq.heappush(self.heap, order)
        return

    def pop(self) -> Order:
        """
        O(log n) where n = len(self.heap) if no invalid orders,
        but can be up to O(num_invalid_orders * log n).

        Only pops valid orders (i.e. orders with size > 0).
        """
        while True:
            assert len(self.heap) > 0
            if len(self.heap) == 0:  # TODO remove if not needed
                raise IndexError("Pop from empty heap.")
            order = heapq.heappop(self.heap)
            if order.order_id in self.order_id_to_order:
                del self.order_id_to_order[order.order_id]
                break
        return order

    def pushpop(self, order: Order) -> Order:
        """
        O(log n) where n = len(self.heap) if no invalid orders,
        but can be up to O(num_invalid_orders * log n).

        Faster than a separate push and pop when the pop returns
        a valid order on first attempt.
        """
        assert order.order_id not in self.order_id_to_order
        self.order_id_to_order[order.order_id] = order
        order = heapq.heappushpop(self.heap, order)
        if order.order_id in self.order_id_to_order:
            del self.order_id_to_order[order.order_id]
        else:
            order = self.pop()
        return order

    def peek(self) -> Order:
        """
        O(1) if no invalid orders, but can cost the same as self.pop + self.push.
        """
        order = self.heap[0]
        if order.order_id in self.order_id_to_order:
            return order
        order = self.pop()
        self.push(order)
        return order


class MaxHeapOfOrders(SimpleMinHeapOfOrders):

    """
    Tracks total price and size.
    """

    def __init__(self, orders: Iterable[Order] = ()):
        """
        Assumes all orders are valid (i.e. have size > 0 and price > 0).
        """
        orders = list(orders)
        for order in orders:
            order.enable_max_heap_compatibility()
        super().__init__(orders)
        self.total_size: int = sum(order.size for order in self.heap)
        self.total_price: float = sum(order.size * order.price for order in self.heap)
        return

    def reduce_order_size(self, order_id: str, reduction_amount: int) -> Order:
        order = super().reduce_order_size(order_id, reduction_amount)
        self.total_size -= reduction_amount
        self.total_price -= reduction_amount * order.price
        assert self.total_size >= 0
        assert round(self.total_price, 2) >= 0
        assert round(self.total_price, 2) == round(sum(e.size*e.price for e in self.heap), 2)
        return order

    def push(self, order: Order) -> None:
        order.enable_max_heap_compatibility()
        super().push(order)
        self.total_size += order.size
        self.total_price += order.size * order.price
        assert self.total_size >= 0
        assert round(self.total_price, 2) >= 0
        assert round(self.total_price, 2) == round(sum(e.size*e.price for e in self.heap), 2)
        return

    def pop(self) -> Order:
        order = super().pop()
        order.disable_max_heap_compatibility()
        self.total_size -= order.size
        self.total_price -= order.size * order.price
        assert self.total_size >= 0
        assert round(self.total_price, 2) >= 0
        assert round(self.total_price, 2) == round(sum(e.size*e.price for e in self.heap), 2)
        return order

    def pushpop(self, input_order: Order) -> Order:
        input_order.enable_max_heap_compatibility()
        assert input_order.order_id not in self.order_id_to_order
        self.order_id_to_order[input_order.order_id] = input_order

        order = heapq.heappushpop(self.heap, input_order)
        self.total_size += input_order.size
        self.total_price += input_order.size * input_order.price
        if order.order_id in self.order_id_to_order:
            del self.order_id_to_order[order.order_id]
        else:
            order = super().pop()
        order.disable_max_heap_compatibility()
        self.total_size -= order.size
        self.total_price -= order.size * order.price
        
        assert self.total_size >= 0
        assert round(self.total_price, 2) >= 0
        assert round(self.total_price, 2) == round(sum(e.size*e.price for e in self.heap), 2)
        return order

    def target_price(self) -> float:
        """
        O(1).

        Assumes TARGET_SIZE <= self.total_size.
        """
        assert TARGET_SIZE <= self.total_size
        excess_num_shares = self.total_size - TARGET_SIZE
        if excess_num_shares == 0:
            return self.total_price
        max_order = self.peek()
        ans = self.total_price - excess_num_shares * max_order.price
        ans = round(ans, 2)
        return ans


###########
# Globals #
###########

TARGET_SIZE: Optional[int] = None

USED_BIDS: MaxHeapOfOrders = MaxHeapOfOrders()
USED_ASKS: MaxHeapOfOrders = MaxHeapOfOrders()

REMAINING_BIDS: SimpleMinHeapOfOrders = SimpleMinHeapOfOrders()
REMAINING_ASKS: SimpleMinHeapOfOrders = SimpleMinHeapOfOrders()

################################
# Message Processing Utilities #
################################


def process_reduce_message(timestamp: str, order_id: str, size: str) -> None:
    size = int(size)
    # TODO abstract this stuff out
    if order_id in USED_BIDS:
        if USED_BIDS.total_size < TARGET_SIZE:
            USED_BIDS.reduce_order_size(order_id, size)
        else:
            before_target_price = USED_BIDS.target_price()
            USED_BIDS.reduce_order_size(order_id, size)
            while (cannot_sell := USED_BIDS.total_size < TARGET_SIZE) and len(REMAINING_BIDS) > 0:
                order = REMAINING_BIDS.pop()
                USED_BIDS.push(order)
            if cannot_sell:
                print(f"{timestamp} S NA")
            elif before_target_price != (after_target_price := USED_BIDS.target_price()):
                print(f"{timestamp} S {after_target_price:.2f}")
    elif order_id in USED_ASKS:
        if USED_ASKS.total_size < TARGET_SIZE:
            USED_ASKS.reduce_order_size(order_id, size)
        else:
            before_target_price = USED_ASKS.target_price()
            USED_ASKS.reduce_order_size(order_id, size)
            while (cannot_buy := USED_ASKS.total_size < TARGET_SIZE) and len(REMAINING_ASKS) > 0:
                order = REMAINING_ASKS.pop()
                USED_ASKS.push(order)
            if cannot_buy:
                print(f"{timestamp} B NA")
            elif before_target_price != (after_target_price := USED_ASKS.target_price()):
                print(f"{timestamp} B {after_target_price:.2f}")
    elif order_id in REMAINING_BIDS:
        REMAINING_BIDS.reduce_order_size(order_id, size)
    elif order_id in REMAINING_ASKS:
        REMAINING_ASKS.reduce_order_size(order_id, size)
    else:
        raise RuntimeError(f"Unknown order id {order_id}.")
    return


def process_add_message(
    timestamp: str, order_id: str, side: str, price: str, size: str
) -> None:
    price = float(price)
    size = int(size)
    # TODO abstract this stuff out
    if side == "B":
        bid = Bid(order_id, price, size)
        changed = False
        if USED_BIDS.total_size < TARGET_SIZE:
            USED_BIDS.push(bid)
            changed = USED_BIDS.total_size >= TARGET_SIZE
            if changed:
                after_target_price = USED_BIDS.target_price()
        else:
            before_target_price = USED_BIDS.target_price()
            worse_bid = USED_BIDS.pushpop(bid)
            REMAINING_BIDS.push(worse_bid)
            after_target_price = USED_BIDS.target_price()
            if bid in USED_BIDS:
                changed = after_target_price != before_target_price
            assert (not changed) or after_target_price > before_target_price
        if changed:
            print(f"{timestamp} S {after_target_price:.2f}")
    elif side == "S":
        ask = Ask(order_id, price, size)
        changed = False
        if USED_ASKS.total_size < TARGET_SIZE:
            USED_ASKS.push(ask)
            changed = USED_ASKS.total_size >= TARGET_SIZE
            if changed:
                after_target_price = USED_ASKS.target_price()
        else:
            before_target_price = USED_ASKS.target_price()
            worse_ask = USED_ASKS.pushpop(ask)
            REMAINING_ASKS.push(worse_ask)
            after_target_price = USED_ASKS.target_price()
            if ask in USED_ASKS:
                changed = after_target_price != before_target_price
            assert (not changed) or after_target_price < before_target_price
        if changed:
            print(f"{timestamp} B {after_target_price:.2f}")
    else:
        raise RuntimeError(f"Unknown side type {repr(side)}.")
    return

def process_message(message: str) -> None:
    timestamp, message_type, *remaining_message_data = message.split()
    if message_type == "A":
        process_add_message(timestamp, *remaining_message_data)
    elif message_type == "R":
        process_reduce_message(timestamp, *remaining_message_data)
    else:
        raise RuntimeError(f"Unknown message type {repr(message_type)}.")
    return


##########
# Driver #
##########

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        raise RuntimeError(f"Expected one command line arg, got {len(args)}.")
    [target_size_string] = args
    if not target_size_string.isdigit():
        raise TypeError(
            "Target size command line arg must be interpretable as "
            f"an integer, got {repr(target_size_string)}."
        )

    TARGET_SIZE = int(target_size_string)
    for message in sys.stdin:
        process_message(message)
