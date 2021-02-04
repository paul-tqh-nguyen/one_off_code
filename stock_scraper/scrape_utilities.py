
'''

'''

# TODO fill in doc string

###########
# Imports #
###########

import argparse
import tempfile
import time
import logging
import sys
import os
import json
import asyncio
import pyppeteer
import bs4
import tqdm
import datetime
import sqlite3
import requests
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from typing import Union, List, Set, Tuple, Iterable, Callable, Awaitable, Coroutine

from misc_utilities import *

# TODO make sure these imports are used

###########
# Globals #
###########

MAX_NUMBER_OF_CONCURRENT_BROWSERS = 10

HEADLESS = True

MAX_NUMBER_OF_SCRAPE_ATTEMPTS = 1

DEFFAULT_OUTPUT_DB_FILE = './stock_data.db'

##########################
# Web Scraping Utilities #
##########################

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

async def launch_browser(*args, **kwargs) -> pyppeteer.browser.Browser:
    sentinel = object()
    browser = sentinel
    while browser is sentinel:
        try:
            browser: pyppeteer.browser.Browser = await pyppeteer.launch(kwargs, args=args)
        except Exception as browser_launch_error:
            pass
    return browser
    
@asynccontextmanager
async def new_browser(*args, **kwargs) -> Generator:
    browser = await launch_browser(*args, **kwargs)
    try:
        yield browser
        await browser.close()
    except Exception as error:
        await browser.close()
        raise error
    return

BROWSER_POOL_SIZE = MAX_NUMBER_OF_CONCURRENT_BROWSERS
BROWSER_POOL: List[pyppeteer.browser.Browser] = []
BROWSER_POOL_ID_QUEUE = mp.Queue()

async def initialize_browser_pool() -> None:
    for browser_pool_id in range(BROWSER_POOL_SIZE):
        BROWSER_POOL_ID_QUEUE.put(browser_pool_id)
        browser = await launch_browser(headless=HEADLESS)
        BROWSER_POOL.append(browser)
    return

async def close_browser_pool() -> None:
    for _ in range(len(BROWSER_POOL)):
        browser = BROWSER_POOL.pop()
        await browser.close()
    return

@contextmanager
def browser_pool_initialized() -> Generator:
    assert len(BROWSER_POOL) == 0, f'Browser pool already initialized.'
    EVENT_LOOP.run_until_complete(initialize_browser_pool())
    yield
    EVENT_LOOP.run_until_complete(close_browser_pool())

@contextmanager
def pool_browser() -> Generator:
    assert len(BROWSER_POOL) > 0, f'Browser pool is empty (it is likely uninitialized).'
    browser_pool_id = BROWSER_POOL_ID_QUEUE.get()
    browser = BROWSER_POOL[browser_pool_id]
    yield browser
    BROWSER_POOL_ID_QUEUE.put(browser_pool_id)
    return

def multi_attempt_scrape_function(func: Awaitable) -> Awaitable:
    async def decorating_function(*args, **kwargs):
        unique_bogus_result_identifier = object()
        result = unique_bogus_result_identifier
        failure_message = unique_bogus_result_identifier
        for _ in range(MAX_NUMBER_OF_SCRAPE_ATTEMPTS):
            try:
                result = await func(*args, **kwargs)
                break
            except Exception as e:
                failure_message = f'{func.__qualname__} with the args {args} and kwargs {kwargs} failed due to the following error: {repr(e)}'
                LOGGER.info(failure_message+'\nTrying again.')
                pass
        if result == unique_bogus_result_identifier:
            raise RuntimeError(failure_message)
        return result
    return decorating_function

########################
# Pyppeteer Extensions #
########################

async def _get_elements(self: Union[pyppeteer.page.Page, pyppeteer.element_handle.ElementHandle], selector: str) -> List[pyppeteer.element_handle.ElementHandle]:
    if isinstance(self, pyppeteer.page.Page):
        await self.waitForSelector(selector)
    elements = await self.querySelectorAll(selector)
    return elements
setattr(pyppeteer.page.Page, 'get_elements', _get_elements)
setattr(pyppeteer.element_handle.ElementHandle, 'get_elements', _get_elements)

async def _get_sole_element(self: Union[pyppeteer.page.Page, pyppeteer.element_handle.ElementHandle], selector: str) -> pyppeteer.element_handle.ElementHandle:
    return only_one(await self.get_elements(selector))
setattr(pyppeteer.page.Page, 'get_sole_element', _get_sole_element)
setattr(pyppeteer.element_handle.ElementHandle, 'get_sole_element', _get_sole_element)

async def _safelyWaitForSelector(self: pyppeteer.page.Page, *args, **kwargs) -> bool:
    try:
        await self.waitForSelector(*args, **kwargs)
        success = True
    except pyppeteer.errors.TimeoutError:
        success = False
    return success
setattr(pyppeteer.page.Page, 'safelyWaitForSelector', _safelyWaitForSelector)

async def _safelyWaitForNavigation(self: pyppeteer.page.Page, *args, **kwargs) -> bool:
    try:
        await self.waitForNavigation(*args, **kwargs)
        success = True
    except pyppeteer.errors.TimeoutError:
        success = False
    return success
setattr(pyppeteer.page.Page, 'safelyWaitForNavigation', _safelyWaitForNavigation)

########################
# Threadsafe Utilities #
########################

class ThreadSafeCounter():
    def __init__(self):
        self.val = mp.RawValue('i', 0)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        with self.lock:
            return self.val.value

#########################
# Gather Ticker Symbols #
#########################

RH_LIST_URLS = [
    'https://robinhood.com/collections/100-most-popular',
    'https://robinhood.com/collections/internet',
    'https://robinhood.com/collections/manufacturing',
    'https://robinhood.com/collections/computer-software',
    'https://robinhood.com/collections/software-service',
    'https://robinhood.com/collections/retail',
    'https://robinhood.com/collections/e-commerce',
    'https://robinhood.com/collections/hospitality',
    'https://robinhood.com/collections/food',
    'https://robinhood.com/collections/medical',
    'https://robinhood.com/collections/technology',
    'https://robinhood.com/collections/biotechnology',
    'https://robinhood.com/collections/credit-card',
    'https://robinhood.com/collections/payment',
]

def gather_rh_ticker_symbols() -> Iterable[str]:
    all_ticker_symbols = set()
    for rh_list_url in RH_LIST_URLS:
        response = requests.get(rh_list_url)
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        ticker_symbols = {
            anchor.get('href').replace('/stocks/', '')
            for anchor in soup.select('html body main section table.table tbody tr a')
        }
        all_ticker_symbols |= ticker_symbols
    return all_ticker_symbols

ALL_TICKER_SYMBOLS_URL = 'https://stockanalysis.com/stocks/'

def gather_all_ticker_symbols() -> Iterable[str]:
    response = requests.get(ALL_TICKER_SYMBOLS_URL)
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    ticker_links = soup.select('main.site-main article.page.type-page.status-publish div.inside-article div.entry-content ul.no-spacing li a')
    ticker_symbols = [ticker_link.text.split(' - ')[0] for ticker_link in ticker_links]
    return ticker_symbols

##########
# Driver #
##########

if __name__ == '__main__':
    pass # TODO add something here
