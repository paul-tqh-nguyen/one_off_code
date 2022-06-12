
import os

import numpy as np

import numba
from numba import njit

@njit(numba.int32(numba.types.Array(numba.types.uint8, 1, 'C', readonly=True)))
def f(byte_array: np.ndarray):
    array = byte_array.view(np.uint8)
    if array.sum() == 3:
        ans = 1234
    else:
        ans = array.sum()
    return ans

def main() -> None:
    input_bytes = b'\x01\x02\x01\x02\x01\x02\x01\x02'
    byte_array = np.frombuffer(input_bytes, dtype=np.uint8)
    ans = f(byte_array)
    print(f"ans {repr(ans)}")
    return
            
if __name__ == '__main__':
    print('\n' * 100)
    main()
