
from visualize_bytecode import visualize_bytecode

if __name__ == "__main__":

    def f(x):
        is_even = x % 2 == 0
        if is_even:
            return True
        return False
    
    visualize_bytecode(f)
