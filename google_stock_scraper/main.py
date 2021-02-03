
'''

'''

# TODO fill in doc string

###########
# Imports #
###########

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
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Union, List, Tuple, Iterable, Callable, Awaitable, Coroutine

from misc_utilities import *

# TODO make sure these imports are used

###########
# Globals #
###########

MAX_NUMBER_OF_SCRAPE_ATTEMPTS = 1
MAX_NUMBER_OF_CONCURRENT_BROWSERS = 10

HEADLESS = True

ALL_TICKER_SYMBOLS_URL = 'https://stockanalysis.com/stocks/'

STOCK_DATA_DB_FILE = './stock_data.db'

###########
# Logging #
###########

LOGGER_NAME = 'scraping_logger'
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER_OUTPUT_FILE = './logs.txt'
LOGGER_STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)

def _initialize_logger() -> None:
    LOGGER.setLevel(logging.INFO)
    logging_formatter = logging.Formatter('{asctime} - pid: {process} - threadid: {thread} - func: {funcName} - {levelname}: {message}', style='{')
    logging_file_handler = logging.FileHandler(LOGGER_OUTPUT_FILE)
    logging_file_handler.setFormatter(logging_formatter)
    LOGGER.addHandler(logging_file_handler)
    LOGGER.addHandler(LOGGER_STREAM_HANDLER)
    return

_initialize_logger()

##########################
# Web Scraping Utilities #
##########################

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

@asynccontextmanager
async def new_browser(*args, **kwargs) -> Generator:
    sentinel = object()
    browser = sentinel
    while browser is sentinel:
        try:
            browser: pyppeteer.browser.Browser = await pyppeteer.launch(kwargs, args=args)
        except Exception as browser_launch_error:
            pass
    try:
        yield browser
        await browser.close()
    except Exception as error:
        await browser.close()
        raise error
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

async def gather_ticker_symbols() -> List[str]:
    async with new_browser(headless=HEADLESS) as browser:
        page = only_one(await browser.pages())
        await page.goto(ALL_TICKER_SYMBOLS_URL)
        main = await page.get_sole_element('main#main.site-main')
        article = await main.get_sole_element('article.page.type-page.status-publish')
        inside_article_div = await article.get_sole_element('div.inside-article')
        entry_content_div = await inside_article_div.get_sole_element('div.entry-content')
        ul = await page.get_sole_element('ul.no-spacing')
        ul_html_string = await page.evaluate('(element) => element.innerHTML', ul)
        soup = bs4.BeautifulSoup(ul_html_string, 'html.parser')
        li_elements = soup.findAll('li')
        ticker_symbols = [only_one(li_element.findAll('a')).getText().split(' - ')[0] for li_element in li_elements]
    return ticker_symbols

@multi_attempt_scrape_function
async def _gather_ticker_symbol_rows(ticker_symbol: str) -> Tuple[List[Tuple[datetime.datetime, str, float]], str]:
    zero_result_explanation = ''
    seen_whole_time_strings = set()
    rows = []
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    async with new_browser(headless=HEADLESS) as browser:
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

        chart_svgs = await chart_div.get_elements('svg')
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

                time_string, period = whole_time_string.split(' ')
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
    
async def update_stock_db(cursor: sqlite3.Cursor) -> None:
    ticker_symbols = await gather_ticker_symbols()
    ticker_symbols = ['GIDYL']
    LOGGER.info(f'{len(ticker_symbols)} ticker symbols gathered.')
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
    return

##########
# Driver #
##########

def main() -> None:
    connection = sqlite3.connect(STOCK_DATA_DB_FILE)
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS stocks(date timestamp, ticker_symbol text, price real)''')
    
    with timer('Data gathering'):
        EVENT_LOOP.run_until_complete(update_stock_db(cursor))
    
    df = pd.read_sql_query('SELECT * from stocks', connection)
    breakpoint()
    return

if __name__ == '__main__':
    main()
