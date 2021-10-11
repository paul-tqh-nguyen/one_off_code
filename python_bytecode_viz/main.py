from visualize_bytecode import visualize_bytecode

if __name__ == "__main__":

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

    visualize_bytecode(f, "/Users/pnguyen/Desktop/test/")
