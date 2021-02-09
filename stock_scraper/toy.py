
import asyncio

def subprocess_code():
     loop = asyncio.get_event_loop()
     @asyncio.coroutine
     def f():
        for i in range(10):
            print(i)
            yield from asyncio.sleep(1)
     loop.run_until_complete(f())
