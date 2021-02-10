
'''

'''

# TODO fill in doc string

###########
# Imports #
###########

import atexit
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

#################
# Misc. Helpers #
#################

def one_time_execution(func: Callable[[], None]) -> None:
    func()
    return

def get_local_datetime() -> datetime.datetime:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)

################
# DB Utilities #
################

class DB_INFO:

    daily_db_file = (lambda current_date: f'stock_data_{current_date.month}_{current_date.day}_{current_date.year}.db')(get_local_datetime())
    db_lock = mp.Lock()
    db_connection = sqlite3.connect(daily_db_file, check_same_thread=False)
    db_cursor = db_connection.cursor()
    
    @classmethod
    @contextmanager
    def db_access(cls) -> Generator[Tuple[sqlite3.Connection, sqlite3.Cursor], None, None]:
        with cls.db_lock:
            yield cls.db_connection, cls.db_cursor
        return

with DB_INFO.db_access() as (_, db_cursor):
    db_cursor.execute('CREATE TABLE IF NOT EXISTS stocks(date timestamp, ticker_symbol text, price real)')

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
    EVENT_LOOP_THREAD.daemon = True
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

##########################
# Web Scraping Utilities #
##########################

async def launch_browser(*args, **kwargs) -> pyppeteer.browser.Browser:
    sentinel = object()
    browser = sentinel
    while browser is sentinel:
        try:
            browser: pyppeteer.browser.Browser = await pyppeteer.launch(kwargs, args=args)
        except Exception as browser_launch_error:
            pass
    return browser

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

######################
# Vanguard Utilities #
######################

VANGUARD_LOGIN_URL = 'https://investor.vanguard.com/my-account/log-on'

VANGUARD_BROWSER = None

async def _initialize_vanguard_browser() -> pyppeteer.browser.Browser:
    browser = await launch_browser(
        headless=False, 
        handleSIGINT=False,
        handleSIGTERM=False,
        handleSIGHUP=False
    )
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

@event_loop_func
def close_vanguard_browser() -> None:
    global VANGUARD_BROWSER
    assert VANGUARD_BROWSER is not None, f'Browser already closed.'
    run_awaitable(VANGUARD_BROWSER._closeCallback())
    VANGUARD_BROWSER = None
    return 

########################
# RH Browser Utilities #
########################

RH_BROWSER = None

TRACKED_TICKER_SYMBOL_TO_PAGE: Dict[str, pyppeteer.page.Page] = dict()

async def _initialize_rh_browser() -> pyppeteer.browser.Browser:
    browser = await launch_browser(
        headless=True,
        handleSIGINT=False,
        handleSIGTERM=False,
        handleSIGHUP=False
    )
    return browser

@event_loop_func
def initialize_rh_browser() -> pyppeteer.browser.Browser:
    global RH_BROWSER
    assert RH_BROWSER is None, f'Browser already initialized.'
    browser = run_awaitable(_initialize_rh_browser())
    RH_BROWSER = browser
    return browser

@event_loop_func
def close_rh_browser() -> None:
    global RH_BROWSER
    assert RH_BROWSER is not None, f'Browser already closed.'
    run_awaitable(RH_BROWSER._closeCallback())
    RH_BROWSER = None
    return 

########################
# RH Scraper Utilities #
########################

RH_SCRAPER_TASK = None

async def open_pages_for_ticker_symbols(ticker_symbols: List[str]) -> None:
    ticker_symbols = eager_map(str.upper, ticker_symbols)
    
    # Add new pages
    current_pages = await RH_BROWSER.pages()
    for _ in range(len(ticker_symbols) - len(current_pages)):
        page = await RH_BROWSER.newPage()
    
    # Close unneccessary pages
    current_pages = await RH_BROWSER.pages()
    for index in range(len(current_pages) - len(ticker_symbols)):
        await current_pages[index].close()
        
    current_pages = await RH_BROWSER.pages()
    assert len(current_pages) == len(ticker_symbols)

    page_opening_tasks = []
    for page, ticker_symbol in zip(current_pages, ticker_symbols):
        url = f'https://robinhood.com/stocks/{ticker_symbol}'
        page_opening_task = EVENT_LOOP.create_task(page.goto(url))
        page_opening_tasks.append(page_opening_task)
        TRACKED_TICKER_SYMBOL_TO_PAGE[ticker_symbol] = page
    for page_opening_task in page_opening_tasks:
        await page_opening_task
    return

async def scrape_ticker_symbols() -> None:
    try:
        while True:
            rows = []
            for ticker_symbol, page in TRACKED_TICKER_SYMBOL_TO_PAGE.items():
                assert page.url.split('/')[-1].lower() == ticker_symbol.lower()
                await page.bringToFront()
                
                svg = await page.get_sole_element('body main.app main.main-container div.row section[data-testid="ChartSection"] svg:not([role="img"])')
                top, left, width, height = await page.evaluate(
                    '(element) => {'
                    '    const { top, left, width, height } = element.getBoundingClientRect();'
                    '    return [top, left, width, height];'
                    '}', svg)
    
                for _ in range(10):
                    y = top + (top + height) * random.random()
                    x = left + (left+width) * random.random()
                    await page.mouse.move(x, y)
                await page.mouse.move(1, 1)
                await asyncio.sleep(2)
                
                price_spans = await page.get_elements('body main.app main.main-container div.row section[data-testid="ChartSection"] header')
                price_span_string = await page.evaluate('(element) => element.innerText', price_spans[0])
                price_span_string = price_span_string.split('\n')[0]
                assert price_span_string.startswith('$')
                price = float(price_span_string.replace('$', '').replace(',', ''))
                
                date_time = get_local_datetime()
                row = (date_time, ticker_symbol, price)
                print(f"row {repr(row)}")
                rows.append(row)
            with DB_INFO.db_access() as (db_connection, db_cursor):
                db_cursor.executemany('INSERT INTO stocks VALUES(?,?,?)', rows)
                db_connection.commit()
    except asyncio.CancelledError:
        pass
    return

@event_loop_func
def track_ticker_symbols(*ticker_symbols: List[str]) -> None:
    global RH_BROWSER
    global RH_SCRAPER_TASK
    assert RH_BROWSER is not None, f'RH browser not initialized.'
    if RH_SCRAPER_TASK is not None:
        RH_SCRAPER_TASK.cancel()
    run_awaitable(open_pages_for_ticker_symbols(ticker_symbols))
    RH_SCRAPER_TASK = enqueue_awaitable(scrape_ticker_symbols())
    return

##########
# Driver #
##########

@one_time_execution
def initialize_browsers() -> None:
    initialize_vanguard_browser()
    initialize_rh_browser()
    return

@atexit.register
def module_exit_callback() -> None:
    close_vanguard_browser()
    close_rh_browser()
    stop_thread_for_event_loop()
    return

if __name__ == '__main__':
    pass # TODO do something here
