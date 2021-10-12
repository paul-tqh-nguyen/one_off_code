from visualize_bytecode import visualize_bytecode


def f(x):
    mod_2 = x % 2
    is_even = mod_2 == 0
    pass
    pass
    if is_even:
        return True
    pass
    pass
    pass


def g(a, b):
    c = a % b
    if c * 10 == 200:
        return b
    b = c - a
    if c * b == a:
        return c
    a *= b / c
    return a


def h(a):
    while a > 10:
        a %= 50
    return a


def gcd(a, b):
    if a < b:
        c = a
        a = b
        b = c

    while b != 0:
        c = a
        a = b
        b = c % b
    return a


if __name__ == "__main__":

    func = g

    # import dis
    # dis.dis(func)

    visualize_bytecode(func, "/Users/pnguyen/Desktop/test/")
