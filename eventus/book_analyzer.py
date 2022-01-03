###########
# Imports #
###########

import sys
import heapq
from typing import (
    List,
    Dict,
    Optional,
    Iterable,
)


################################
# Application-Specific Classes #
################################


class Order:

    """
    Terminology: The phrase "invalid order" denotes an order with size <= 0 in this script's code.
    """

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
        # assert self.is_max_heap_friendly == False
        self.is_max_heap_friendly = True
        return

    def disable_max_heap_compatibility(self) -> None:
        # assert self.is_max_heap_friendly == True
        self.is_max_heap_friendly = False
        return

    def reduce_size(self, amount: int) -> None:
        # assert self.size >= amount
        self.size -= amount
        return

    def __repr__(self) -> str:
        attributes_string = ", ".join(
            f"{k}={repr(self.__dict__[k])}" for k in sorted(self.__dict__.keys())
        )
        return f"{self.__class__.__name__}({attributes_string})"


class Bid(Order):
    def __lt__(self, other) -> bool:
        # use order_id for deterministic sorting results
        return (
            (self.price, self.order_id) < (other.price, other.order_id)
            if self.is_max_heap_friendly
            else (self.price, self.order_id) > (other.price, other.order_id)
        )


class Ask(Order):
    def __lt__(self, other) -> bool:
        return (
            (self.price, self.order_id) > (other.price, other.order_id)
            if self.is_max_heap_friendly
            else (self.price, self.order_id) < (other.price, other.order_id)
        )


class MinHeapOfOrders:

    """
    "Min" means least costly towards us, i.e. cheapest ask and highest bid.
    """

    def __init__(self, orders: Iterable[Order] = ()):
        """
        O(n) where n = len(orders).
        """
        self.heap: List[Order] = list(orders)
        heapq.heapify(self.heap)  # docs say this is linear time
        self.order_id_to_order: Dict[str, Order] = {
            order.order_id: order for order in self.heap
        }
        return

    def __repr__(self) -> str:
        attributes_string = ", ".join(
            f"{k}={repr(self.__dict__[k])}" for k in sorted(self.__dict__.keys())
        )
        return f"{self.__class__.__name__}({attributes_string})"

    def contains_id(self, order_id: str) -> bool:
        """O(1)."""
        # assert isinstance(order_id, str)
        return order_id in self.order_id_to_order

    def __len__(self) -> int:
        """O(1)."""
        return len(self.order_id_to_order)

    def reduce_order_size(self, order_id: str, reduction_amount: int) -> Order:
        """
        O(1).

        Leaves invalid orders (i.e. orders with size <= 0) in self.heap,
        but removes them from self.order_id_to_order.

        An invalid order will be lazily removed from self.heap elsewhere via
        self._remove_invalid_orders since removing them in a batch is faster.
        """
        order = self.order_id_to_order[order_id]
        # assert order.size >= reduction_amount
        if order.size == reduction_amount:
            del self.order_id_to_order[order_id]
        order.reduce_size(reduction_amount)
        return order

    def _remove_invalid_orders(self) -> None:
        """
        O(n) where n = len(self.heap).

        Utility to explicitly remove all invalid orders.
        """
        self.heap = [order for order in self.heap if order.size > 0]
        heapq.heapify(self.heap)
        return

    def push(self, order: Order) -> None:
        """O(log n) where n = len(self.heap)."""
        # assert order.size > 0
        # assert order.order_id not in self.order_id_to_order
        self.order_id_to_order[order.order_id] = order
        heapq.heappush(self.heap, order)
        return

    def pop(self) -> Order:
        """
        O(log n) where n = len(self.heap) if no invalid orders
        (this is the common case), but can be O(n) since we
        remove the invalid orders.

        Only pops valid orders (i.e. orders with size > 0).
        """
        # assert len(self.heap) > 0
        order = heapq.heappop(self.heap)
        if order.order_id not in self.order_id_to_order:
            self._remove_invalid_orders()
            order = heapq.heappop(self.heap)
        del self.order_id_to_order[order.order_id]
        return order

    def peek(self) -> Order:
        """
        O(1) if no invalid orders (this is the common case),
        but can cost O(n).
        """
        order = self.heap[0]
        if order.order_id in self.order_id_to_order:
            return order
        self._remove_invalid_orders()
        order = self.heap[0]
        return order


class PriceTrackingMaxHeapOfOrders(MinHeapOfOrders):

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
        # assert self.total_size >= 0
        # assert round(self.total_price, 2) >= 0
        # assert round(self.total_price, 2) == round(sum(e.size*e.price for e in self.heap), 2)
        return order

    def push(self, order: Order) -> None:
        order.enable_max_heap_compatibility()
        super().push(order)
        self.total_size += order.size
        self.total_price += order.size * order.price
        # assert self.total_size >= 0
        # assert round(self.total_price, 2) >= 0
        # assert round(self.total_price, 2) == round(sum(e.size*e.price for e in self.heap), 2)
        return

    def pop(self) -> Order:
        order = super().pop()
        order.disable_max_heap_compatibility()
        self.total_size -= order.size
        self.total_price -= order.size * order.price
        # assert self.total_size >= 0
        # assert round(self.total_price, 2) >= 0
        # assert round(self.total_price, 2) == round(sum(e.size*e.price for e in self.heap), 2)
        return order

    def target_price(self) -> float:
        """
        O(1).

        Assumes TARGET_SIZE <= self.total_size.
        """
        # assert TARGET_SIZE <= self.total_size
        excess_num_shares = self.total_size - TARGET_SIZE
        if excess_num_shares == 0:
            ans = self.total_price
        else:
            max_order = self.peek()
            ans = self.total_price - excess_num_shares * max_order.price
        ans = round(ans, 2)
        return ans


###########
# Globals #
###########

TARGET_SIZE: Optional[int] = None

USED_BIDS: PriceTrackingMaxHeapOfOrders = PriceTrackingMaxHeapOfOrders()
USED_ASKS: PriceTrackingMaxHeapOfOrders = PriceTrackingMaxHeapOfOrders()

REMAINING_BIDS: MinHeapOfOrders = MinHeapOfOrders()
REMAINING_ASKS: MinHeapOfOrders = MinHeapOfOrders()

################################
# Message Processing Utilities #
################################


def determine_reduce_print_string(
    timestamp: str,
    order_id: str,
    size: int,
    used_orders: PriceTrackingMaxHeapOfOrders,
    remaining_orders: MinHeapOfOrders,
    action: str,
) -> Optional[str]:
    if used_orders.total_size < TARGET_SIZE:
        used_orders.reduce_order_size(order_id, size)
    else:
        before_target_price = used_orders.target_price()
        used_orders.reduce_order_size(order_id, size)
        while (cannot_sell := used_orders.total_size < TARGET_SIZE) and len(
            remaining_orders
        ) > 0:
            used_orders.push(remaining_orders.pop())
        if cannot_sell:
            return f"{timestamp} {action} NA"
        elif before_target_price != (after_target_price := used_orders.target_price()):
            return f"{timestamp} {action} {after_target_price:.2f}"
    return


def process_reduce_message(timestamp: str, order_id: str, size: str) -> None:
    size = int(size)
    if USED_BIDS.contains_id(order_id):
        print_string = determine_reduce_print_string(
            timestamp, order_id, size, USED_BIDS, REMAINING_BIDS, "S"
        )
        if print_string is not None:
            print(print_string)
    elif USED_ASKS.contains_id(order_id):
        print_string = determine_reduce_print_string(
            timestamp, order_id, size, USED_ASKS, REMAINING_ASKS, "B"
        )
        if print_string is not None:
            print(print_string)
    elif REMAINING_BIDS.contains_id(order_id):
        REMAINING_BIDS.reduce_order_size(order_id, size)
    elif REMAINING_ASKS.contains_id(order_id):
        REMAINING_ASKS.reduce_order_size(order_id, size)
    else:
        raise RuntimeError(f"Unknown order id {order_id}.")
    return


def determine_add_print_string(
    timestamp: str,
    order_id: str,
    price: float,
    size: int,
    used_orders: PriceTrackingMaxHeapOfOrders,
    remaining_orders: MinHeapOfOrders,
    order_class: type,
) -> Optional[str]:
    order = order_class(order_id, price, size)
    if used_orders.total_size < TARGET_SIZE:
        used_orders.push(order)
        changed = used_orders.total_size >= TARGET_SIZE
        if changed:
            while used_orders.total_size - used_orders.peek().size >= TARGET_SIZE:
                remaining_orders.push(used_orders.pop())
            after_target_price = used_orders.target_price()
    else:
        before_target_price = used_orders.target_price()
        used_orders.push(order)
        # assert used_orders.total_size >= TARGET_SIZE
        while used_orders.total_size - used_orders.peek().size >= TARGET_SIZE:
            remaining_orders.push(used_orders.pop())
        # assert used_orders.total_size >= TARGET_SIZE
        after_target_price = used_orders.target_price()
        changed = after_target_price != before_target_price
        # assert (not changed) or (after_target_price > before_target_price if order_class == Bid else after_target_price < before_target_price)
    if changed:
        action = "S" if order_class == Bid else "B"
        return f"{timestamp} {action} {after_target_price:.2f}"
    return


def process_add_message(
    timestamp: str, order_id: str, side: str, price: str, size: str
) -> None:
    price = float(price)
    size = int(size)
    if side == "B":
        print_string = determine_add_print_string(
            timestamp,
            order_id,
            price,
            size,
            USED_BIDS,
            REMAINING_BIDS,
            Bid,
        )
        if print_string is not None:
            print(print_string)
    elif side == "S":
        print_string = determine_add_print_string(
            timestamp,
            order_id,
            price,
            size,
            USED_ASKS,
            REMAINING_ASKS,
            Ask,
        )
        if print_string is not None:
            print(print_string)
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

    # import cProfile
    # cProfile.run("{process_message(message) for message in sys.stdin}")
