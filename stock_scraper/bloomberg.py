
'''

'''

# TODO fill in doc string

###########
# Imports #
###########

import sys
import os
import json
import asyncio
import pandas as pd
import pandas_datareader as web

from scrape_utilities import *
from misc_utilities import *

# TODO make sure these imports are used

###########
# Globals #
###########

BLOOMBERG_URL_TEMPLATE = 'https://www.bloomberg.com/markets/api/bulk-time-series/price/{ticker_symbol}%3AUS?timeFrame=1_DAY'

#################
# Functionality #
#################

async def _get_ticker_symbol_data(ticker_symbol: str) -> list:

    with pool_browser() as browser:
        page = only_one(await browser.pages())
        await page.setViewport({'width': 2000, 'height': 2000})
        
        awaitables = []
        lock = mp.Lock()
        @page.on('response')
        def callback(message):
            e = message.text()
            with lock:
                awaitables.append(e)
        
        url = BLOOMBERG_URL_TEMPLATE.format(ticker_symbol=ticker_symbol)
        await page.goto(url)

        with timeout(30):
            while True:
                try:
                    with lock:
                        awaitable = awaitables.pop()
                    json_text = await awaitable
                    json_data = json.loads(json_text)
                    break
                except:
                    pass
    
    answer = 1
    
    return answer

def get_ticker_symbol_data(ticker_symbol: str) -> list:
    return EVENT_LOOP.run_until_complete(_get_ticker_symbol_data(ticker_symbol))

##########
# Driver #
##########
    
if __name__ == '__main__':
    with browser_pool_initialized():
        ticker_symbol = random.choice(list(gather_rh_ticker_symbols()))
        a = get_ticker_symbol_data(ticker_symbol)
        breakpoint()
