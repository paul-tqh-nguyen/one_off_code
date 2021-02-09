
'''

'''

# TODO fill in doc string

###########
# Imports #
###########

import argparse
import tempfile
import time
import random
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
import threading
import aioprocessing
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict, namedtuple
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from typing import Any, Union, List, Set, Tuple, Iterable, Callable, Awaitable, Coroutine, Dict, Generator

# from scrape_utilities import *
from misc_utilities import *

# TODO make sure these imports are used
# TODO review type hints of all functions below

################
# Date Helpers #
################

def get_local_datetime() -> datetime.datetime:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)

################
# DB Utilities #
################

CURRENT_DATE = get_local_datetime()

DAILY_DB_FILE = f'stock_data_{CURRENT_DATE.month}_{CURRENT_DATE.day}_{CURRENT_DATE.year}.db'

DB_CONNECTION = sqlite3.connect(DAILY_DB_FILE, check_same_thread=False)

DB_CURSOR = DB_CONNECTION.cursor()

DB_CURSOR.execute('CREATE TABLE IF NOT EXISTS stocks(date timestamp, ticker_symbol text, price real)')

###################
# Async Utilities #
###################

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

EVENT_LOOP_THREAD = None

def _run_event_loop_in_thread():
    assert not EVENT_LOOP.is_running()
    EVENT_LOOP.run_forever()

def start_thread_for_event_loop():
    global EVENT_LOOP_THREAD
    assert EVENT_LOOP_THREAD is None or not EVENT_LOOP_THREAD.isAlive()
    EVENT_LOOP_THREAD = threading.Thread(target=_run_event_loop_in_thread, args=())
    EVENT_LOOP_THREAD.start()
    return

def stop_thread_for_event_loop():
    global EVENT_LOOP_THREAD
    assert EVENT_LOOP_THREAD.isAlive()
    assert EVENT_LOOP.is_running()
    EVENT_LOOP.call_soon_threadsafe(EVENT_LOOP.stop)
    EVENT_LOOP_THREAD.join()
    return

def restart_thread_for_event_loop():
    stop_thread_for_event_loop()
    start_thread_for_event_loop()
    return

def enqueue_awaitable(awaitable: Awaitable):
    task = EVENT_LOOP.create_task(awaitable)
    restart_thread_for_event_loop()
    return task

def join_awaitable(awaitable: Awaitable):
    stop_thread_for_event_loop()
    result = EVENT_LOOP.run_until_complete(awaitable)
    start_thread_for_event_loop()
    return result

def run_awaitable(awaitable: Awaitable):
    task = EVENT_LOOP.create_task(awaitable)
    stop_thread_for_event_loop()
    result = EVENT_LOOP.run_until_complete(task)
    start_thread_for_event_loop()
    return result

@contextmanager
def safe_event_loop_thread() -> Generator[None, None, None]:
    try:
        yield
    except Exception as err:
        stop_thread_for_event_loop()
        raise err
    return

def event_loop_func(func: Callable) -> Callable:
    def decorated_function(*args, **kwargs) -> Any:
        with safe_event_loop_thread():
            result = func(*args, **kwargs)
        return result
    return decorated_function

start_thread_for_event_loop()

######################
# Vanguard Utilities #
######################

VANGUARD_LOGIN_URL = 'https://investor.vanguard.com/my-account/log-on'

VANGUARD_BROWSER = None

async def _initialize_vanguard_browser() -> pyppeteer.browser.Browser:
    browser = await launch_browser(headless=False)
    page = only_one(await browser.pages())
    await page.goto(VANGUARD_LOGIN_URL)
    return browser

@event_loop_func
def initialize_vanguard_browser() -> pyppeteer.browser.Browser:
    global VANGUARD_BROWSER
    assert VANGUARD_BROWSER is None, f'Browser already initialized.'
    browser = run_awaitable(_initialize_vanguard_browser())
    VANGUARD_BROWSER = browser
    return browser

################
# RH Utilities #
################

# RH_BROWSER = None

# TRACKED_TICKER_SYMBOL_TO_PAGE: Dict[str, pyppeteer.page.Page] = dict()

# RH_SCRAPER_THREAD = None

# RH_SCRAPER_TASK = None

# async def _initialize_rh_browser() -> pyppeteer.browser.Browser:
#     browser = await launch_browser(headless=False) # TODO make this headless
#     return browser

# def initialize_rh_browser() -> pyppeteer.browser.Browser:
#     global RH_BROWSER
#     assert RH_BROWSER is None, f'Browser already initialized.'
#     browser = EVENT_LOOP.run_until_complete(_initialize_rh_browser())
#     RH_BROWSER = browser
#     return browser

# async def open_pages_for_ticker_symbols(ticker_symbols: List[str]) -> None:
#     ticker_symbols = eager_map(str.upper, ticker_symbols)
#     # Add new pages
#     current_pages = await RH_BROWSER.pages()
#     for _ in range(len(ticker_symbols) - len(current_pages)):
#         page = await RH_BROWSER.newPage()
#     # Close unneccessary pages
#     current_pages = await RH_BROWSER.pages()
#     for index in range(len(current_pages) - len(ticker_symbols)):
#         await current_pages[index].close()
        
#     current_pages = await RH_BROWSER.pages()
#     assert len(current_pages) == len(ticker_symbols)

#     page_opening_tasks = []
#     for page, ticker_symbol in zip(current_pages, ticker_symbols):
#         url = f'https://robinhood.com/stocks/{ticker_symbol}'
#         page_opening_task = EVENT_LOOP.create_task(page.goto(url))
#         page_opening_tasks.append(page_opening_task)
#         TRACKED_TICKER_SYMBOL_TO_PAGE[ticker_symbol] = page
#     for page_opening_task in page_opening_tasks:
#         await page_opening_task
#     return

# async def _scrape_ticker_symbols() -> None:
#     LOGGER.info('0')
#     while True:
#         LOGGER.info('0-1')
#         rows = []
#         LOGGER.info('0-2')
#         for ticker_symbol, page in TRACKED_TICKER_SYMBOL_TO_PAGE.items():
#             LOGGER.info('0-3')
#             assert page.url.split('/')[-1].lower() == ticker_symbol.lower()
#             LOGGER.info(f"ticker_symbol {repr(ticker_symbol)}")
#             LOGGER.info(f"page {repr(page)}")
#             await page.bringToFront()
#             LOGGER.info('1')
            
#             svg = await page.get_sole_element('body main.app main.main-container div.row section[data-testid="ChartSection"] svg:not([role="img"])')
#             LOGGER.info('2')
#             top, left, width, height = await page.evaluate('''
# (element) => {
#     const { top, left, width, height } = element.getBoundingClientRect();
#     return [top, left, width, height];
# }''', svg)

#             LOGGER.info('3')
#             for _ in range(10):
#                 y = top + (top + height) * random.random()
#                 x = left + (left+width) * random.random()
#                 await page.mouse.move(x, y)
#             LOGGER.info('4')
#             await page.mouse.move(1, 1)
#             LOGGER.info('5')
            
#             price_spans = await page.get_elements('body main.app main.main-container div.row section[data-testid="ChartSection"] header')
#             price_span_string = await page.evaluate('(element) => element.innerText', price_spans[0])
#             price_span_string = price_span_string.split('\n')[0]
#             assert price_span_string.startswith('$')
#             price = float(price_span_string.replace('$', '').replace(',', ''))
#             LOGGER.info('6')
            
#             date_time = get_local_datetime()
#             LOGGER.info('7')
#             row = (date_time, ticker_symbol, price)
#             LOGGER.info('8')
#             rows.append(row)
#             LOGGER.info('9')
#         DB_CURSOR.executemany('INSERT INTO stocks VALUES(?,?,?)', rows)
#         LOGGER.info('10')
#         DB_CONNECTION.commit()
#         LOGGER.info('11')
#     return

# def stop_scraping_ticker_symbols() -> None:
#     global RH_SCRAPER_TASK
#     assert bool(RH_SCRAPER_THREAD) == bool(RH_SCRAPER_TASK)
#     if RH_SCRAPER_TASK is not None:
#         RH_SCRAPER_TASK.cancel()
#         EVENT_LOOP.stop()
#     if RH_SCRAPER_THREAD is not None:
#         RH_SCRAPER_THREAD.join()
#     return

# def _start_scraping_ticker_symbols() -> None:
#     LOGGER.info(f"1 _start_scraping_ticker_symbols()")
#     EVENT_LOOP.run_forever()
#     LOGGER.info(f"2 _start_scraping_ticker_symbols()")
#     return

# def restart_scraping_ticker_symbols() -> None:
#     stop_scraping_ticker_symbols()
#     global RH_SCRAPER_TASK
#     print(f"RH_SCRAPER_TASK {repr(RH_SCRAPER_TASK)}")
#     RH_SCRAPER_TASK = EVENT_LOOP.create_task(_scrape_ticker_symbols())
#     print(f"RH_SCRAPER_TASK {repr(RH_SCRAPER_TASK)}")
#     # RH_SCRAPER_PROCESS = mp.Process(target=_start_scraping_ticker_symbols, args=())
#     global RH_SCRAPER_THREAD
#     print(f"RH_SCRAPER_THREAD {repr(RH_SCRAPER_THREAD)}")
#     RH_SCRAPER_THREAD = threading.Thread(target=_start_scraping_ticker_symbols, args=())
#     print(f"RH_SCRAPER_THREAD {repr(RH_SCRAPER_THREAD)}")
#     return

# def track_ticker_symbols(*ticker_symbols: List[str]) -> None:
#     assert RH_BROWSER is not None, f'RH browser not initialized.'
#     EVENT_LOOP.run_until_complete(open_pages_for_ticker_symbols(ticker_symbols))
#     restart_scraping_ticker_symbols()
#     return

##########
# Driver #
##########

# initialize_vanguard_browser()
# initialize_rh_browser()

async def _insert_new_row() -> None:
    #while True:
    for _ in range(10):
        DB_CURSOR.executemany('INSERT INTO stocks VALUES(?,?,?)', [(get_local_datetime(), 'TEST', 1)])
        DB_CONNECTION.commit()
        num_rows = only_one(only_one(DB_CURSOR.execute('SELECT COUNT(*) FROM stocks').fetchall()))
        print(f"num_rows {repr(num_rows)}")
        await asyncio.sleep(1)
    return 444

async def _print_constantly() -> int:
    for _ in range(16):
        print(f"print {repr(print)}")
        await asyncio.sleep(0.25)
    return 123

if __name__ == '__main__':
    with safe_event_loop_thread():
        insert_task = enqueue_awaitable(_insert_new_row())
        time.sleep(2)
        print_result = run_awaitable(_print_constantly())
        assert print_result is 123
        insert_result = join_awaitable(insert_task)
        assert insert_result is 444
        stop_thread_for_event_loop()
        print('done')
    
