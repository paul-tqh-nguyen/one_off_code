
import decorator
import traceback
import multiprocessing as mp
from typing import Tuple, Optional, Callable

@decorator.decorator
def subprocess_test(func: Callable, *args, **kwargs) -> None:

    output_queue = mp.SimpleQueue()
    
    def subprocess_func() -> None:
        raised_exception = None
        traceback_string = ''
        try:
            result = func(*args, **kwargs)
        except Exception as err:
            traceback_string = traceback.format_exc()
            result = None
            raised_exception = err
        output_queue.put((result, raised_exception, traceback_string))
        return

    process = mp.Process(target=subprocess_func)
    process.start()
    process.join()
    process.close()
    result, raised_exception, traceback_string = output_queue.get()
    assert output_queue.empty()
    assert result is None
    if raised_exception is not None:
        print(traceback_string)
        raise raised_exception
    assert traceback_string == ''
    return
