
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
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from typing import Union, List, Tuple, Iterable, Callable, Awaitable, Coroutine

from misc_utilities import *

# TODO make sure these imports are used

###########
# Globals #
###########

MAX_NUMBER_OF_CONCURRENT_BROWSERS = 10

HEADLESS = True

MAX_NUMBER_OF_SCRAPE_ATTEMPTS = 1

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

###########
# Scraper #
###########

@multi_attempt_scrape_function
async def _gather_ticker_symbol_rows(ticker_symbol: str) -> Tuple[List[Tuple[datetime.datetime, str, float]], str]:
    zero_result_explanation = ''
    seen_whole_time_strings = set()
    rows = []
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    with pool_browser() as browser:
        page = only_one(await browser.pages())
        await page.setViewport({'width': 2000, 'height': 2000});
        google_url = f'https://www.google.com/search?q={ticker_symbol}+stock'
        await page.goto(google_url)
        search_div = await page.get_sole_element('div#search')

        chart_found = await page.safelyWaitForSelector('div[jscontroller].knowledge-finance-wholepage-chart__fw-uch', {'timeout': 2_000})
        if not chart_found:
            zero_result_explanation = f'Chart not found for {ticker_symbol}.'
            return rows, zero_result_explanation
        
        chart_div = await search_div.get_sole_element('div[jscontroller].knowledge-finance-wholepage-chart__fw-uch')
        top, left, width, height = await page.evaluate('''
(element) => {
const { top, left, width, height } = element.getBoundingClientRect();
return [top, left, width, height];
}''', chart_div)

        chart_svgs = await chart_div.get_elements('svg.uch-psvg')
        if len(chart_svgs) == 0:
            zero_result_explanation = f'SVG not found for {ticker_symbol}.'
            return rows, zero_result_explanation
        assert len(chart_svgs) == 1, f'{ticker_symbol} has an unexpected number of SVGs ({len(chart_svgs)}) within the chart.'

        whole_time_string = '10:30PM'
        with timeout(30):
            while whole_time_string == '10:30PM':
                info_card = await chart_div.get_sole_element('div.knowledge-finance-wholepage-chart__hover-card')
                time_span = await info_card.get_sole_element('span.knowledge-finance-wholepage-chart__hover-card-time')
                whole_time_string = await page.evaluate('(element) => element.innerHTML', time_span)
        if whole_time_string == '10:30PM':
            zero_result_explanation = f'{ticker_symbol} could not load properly.'
            return rows, zero_result_explanation

        y = (top + top + height) / 2
        for x in range(left, left+width):
            await page.mouse.move(x, y);
            info_card = await chart_div.get_sole_element('div.knowledge-finance-wholepage-chart__hover-card')
            time_span = await info_card.get_sole_element('span.knowledge-finance-wholepage-chart__hover-card-time')
            whole_time_string = await page.evaluate('(element) => element.innerHTML', time_span)
            
            if whole_time_string not in seen_whole_time_strings:

                time_string, period = whole_time_string.split(' ')[-2:]
                hour, minute = eager_map(int, time_string.split(':'))
                assert period in ('AM', 'PM')
                if period == 'PM' and hour != 12:
                    hour += 12
                date_time = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)

                price_span = await info_card.get_sole_element('span.knowledge-finance-wholepage-chart__hover-card-value')
                price_string = await page.evaluate('(element) => element.innerHTML', price_span)

                if not price_string.endswith(' USD'):
                    zero_result_explanation = f'Cannot handle price string {repr(price_string)} for {ticker_symbol}.'
                    return rows, zero_result_explanation
                price = float(price_string.replace(' USD', '').replace(',', ''))

                row = (date_time, ticker_symbol, price)
                rows.append(row)
                seen_whole_time_strings.add(whole_time_string)
        assert len(rows) != 0
        return rows, zero_result_explanation

async def gather_ticker_symbol_rows(ticker_symbol: str) -> Tuple[List[Tuple[datetime.datetime, str, float]], str]:
    rows, zero_result_explanation = await _gather_ticker_symbol_rows(ticker_symbol)
    return rows, zero_result_explanation, ticker_symbol
    
async def update_stock_db(stock_data_db_file: str, ticker_symbols: List[str]) -> None:
    connection = sqlite3.connect(stock_data_db_file)
    cursor = connection.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS stocks(date timestamp, ticker_symbol text, price real)')
        
    total_execution_time = 0
    
    semaphore = asyncio.Semaphore(MAX_NUMBER_OF_CONCURRENT_BROWSERS)
    async def semaphore_task(task: Awaitable):
        async with semaphore:
            return await task
    tasks = asyncio.as_completed(eager_map(semaphore_task, map(gather_ticker_symbol_rows, ticker_symbols)))
    
    for index, task in enumerate(tasks):
        execution_time_container = []
        with timer(exitCallback=lambda time: execution_time_container.append(time)):
            rows, zero_result_explanation, ticker_symbol = await task
        execution_time = only_one(execution_time_container)
        total_execution_time += execution_time
        LOGGER.info(f'[{index+1:4}/{len(ticker_symbols):4}] [{total_execution_time/(index+1):6.2f} s/iter] [{execution_time:6.2f}s] {ticker_symbol:5} yielded {len(rows):4} results. {zero_result_explanation}')
        cursor.executemany('INSERT INTO stocks VALUES(?,?,?);', rows);
        connection.commit();
    
    connection.close()
    return

##########
# Driver #
##########
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-output-file', type=str, default='stock_data.db', help='Output DB file.')
    parser.add_argument('-ticker-symbol-file', type=str, help='Newline-delimited file of ticker symbols.', required=True)
    args = parser.parse_args()

    stock_data_db_file = args.output_file
    with open(args.ticker_symbol_file, 'r') as f:
        ticker_symbols = f.read().split('\n')
    with timer('Data gathering'):
        with browser_pool_initialized():
            EVENT_LOOP.run_until_complete(update_stock_db(stock_data_db_file, ticker_symbols))

