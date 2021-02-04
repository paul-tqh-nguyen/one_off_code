
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

from scrape_utilities import *
from misc_utilities import *

# TODO make sure these imports are used

###########
# Helpers #
###########

COROUTINES_TO_GATHER_TICKER_SYMBOL_ROWS = []

def coroutines_to_gather_ticker_symbol_row(coroutine: Coroutine) -> Coroutine:
    '''Decorator.'''
    assert coroutine not in COROUTINES_TO_GATHER_TICKER_SYMBOL_ROWS, f'{coroutine.__qualname__} already noted.'
    COROUTINES_TO_GATHER_TICKER_SYMBOL_ROWS.append(coroutine)
    return coroutine

COROUTINE_TO_GATHER_TICKER_SYMBOL_ROW_COUNTER = ThreadSafeCounter()

def get_coroutine_to_gather_ticker_symbol_row() -> Coroutine:
    index = COROUTINE_TO_GATHER_TICKER_SYMBOL_ROW_COUNTER.value
    index = index % len(COROUTINES_TO_GATHER_TICKER_SYMBOL_ROWS)
    coroutine = COROUTINES_TO_GATHER_TICKER_SYMBOL_ROWS[index]
    COROUTINE_TO_GATHER_TICKER_SYMBOL_ROW_COUNTER.increment()
    return coroutine

##############
# RH Scraper #
##############

@coroutines_to_gather_ticker_symbol_row
@multi_attempt_scrape_function
async def _gather_ticker_symbol_rows_via_rh(ticker_symbol: str) -> Tuple[List[Tuple[datetime.datetime, str, float]], str]:
    zero_result_explanation = ''
    seen_whole_time_strings = set()
    rows = []
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    with pool_browser() as browser:
        page = only_one(await browser.pages())
        await page.setViewport({'width': 2000, 'height': 2000})
        url = f'https://robinhood.com/stocks/{ticker_symbol.upper()}'
        await page.goto(url)
        
        svg = await page.get_sole_element('body main.app main.main-container div.row section[data-testid="ChartSection"] svg:not([role="img"])')
        
        top, left, width, height = await page.evaluate('''
(element) => {
    const { top, left, width, height } = element.getBoundingClientRect();
    return [top, left, width, height];
}''', svg)
        
        y = (top + top + height) / 2
        for x in range(left, left+width):
            await page.mouse.move(x, y)
            whole_time_string = await page.evaluate('(element) => element.parentElement.parentElement.querySelector("span").innerHTML', svg)

            if whole_time_string not in seen_whole_time_strings:
                time_string, period = whole_time_string.split(' ')[-2:]
                hour, minute = eager_map(int, time_string.split(':'))
                assert period in ('AM', 'PM')
                if period == 'PM' and hour != 12:
                    hour += 12
                date_time = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)
    
                price_spans = await page.get_elements('body main.app main.main-container div.row section[data-testid="ChartSection"] header')
                price_span_string = await page.evaluate('(element) => element.innerText', price_spans[0])
                price_span_string = price_span_string.split('\n')[0]
                assert price_span_string.startswith('$')
                price = float(price_span_string.replace('$', '').replace(',', ''))
    
                row = (date_time, ticker_symbol, price)
                rows.append(row)
                seen_whole_time_strings.add(whole_time_string)
        
        assert len(rows) != 0
        return rows, zero_result_explanation

##################
# Google Scraper #
##################

# @coroutines_to_gather_ticker_symbol_row
# @multi_attempt_scrape_function
# async def _gather_ticker_symbol_rows_via_google(ticker_symbol: str) -> Tuple[List[Tuple[datetime.datetime, str, float]], str]:
#     zero_result_explanation = ''
#     seen_whole_time_strings = set()
#     rows = []
#     now = datetime.datetime.now()
#     year = now.year
#     month = now.month
#     day = now.day
#     with pool_browser() as browser:
#         page = only_one(await browser.pages())
#         await page.setViewport({'width': 2000, 'height': 2000})
#         url = f'https://www.google.com/search?q={ticker_symbol}+stock'
#         await page.goto(url)
#         search_div = await page.get_sole_element('div#search')

#         chart_found = await page.safelyWaitForSelector('div[jscontroller].knowledge-finance-wholepage-chart__fw-uch', {'timeout': 2_000})
#         if not chart_found:
#             zero_result_explanation = f'Chart not found for {ticker_symbol}.'
#             return rows, zero_result_explanation
        
#         chart_div = await search_div.get_sole_element('div[jscontroller].knowledge-finance-wholepage-chart__fw-uch')
#         top, left, width, height = await page.evaluate('''
# (element) => {
#     const { top, left, width, height } = element.getBoundingClientRect();
#     return [top, left, width, height];
# }''', chart_div)

#         chart_svgs = await chart_div.get_elements('svg.uch-psvg')
#         if len(chart_svgs) == 0:
#             zero_result_explanation = f'SVG not found for {ticker_symbol}.'
#             return rows, zero_result_explanation
#         assert len(chart_svgs) == 1, f'{ticker_symbol} has an unexpected number of SVGs ({len(chart_svgs)}) within the chart.'

#         whole_time_string = '10:30PM'
#         with timeout(30):
#             while whole_time_string == '10:30PM':
#                 info_card = await chart_div.get_sole_element('div.knowledge-finance-wholepage-chart__hover-card')
#                 time_span = await info_card.get_sole_element('span.knowledge-finance-wholepage-chart__hover-card-time')
#                 whole_time_string = await page.evaluate('(element) => element.innerHTML', time_span)
#         if whole_time_string == '10:30PM':
#             zero_result_explanation = f'{ticker_symbol} could not load properly.'
#             return rows, zero_result_explanation

#         y = (top + top + height) / 2
#         for x in range(left, left+width):
#             await page.mouse.move(x, y)
#             info_card = await chart_div.get_sole_element('div.knowledge-finance-wholepage-chart__hover-card')
#             time_span = await info_card.get_sole_element('span.knowledge-finance-wholepage-chart__hover-card-time')
#             whole_time_string = await page.evaluate('(element) => element.innerHTML', time_span)
            
#             if whole_time_string not in seen_whole_time_strings:

#                 time_string, period = whole_time_string.split(' ')[-2:]
#                 hour, minute = eager_map(int, time_string.split(':'))
#                 assert period in ('AM', 'PM')
#                 if period == 'PM' and hour != 12:
#                     hour += 12
#                 date_time = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)

#                 price_span = await info_card.get_sole_element('span.knowledge-finance-wholepage-chart__hover-card-value')
#                 price_string = await page.evaluate('(element) => element.innerHTML', price_span)

#                 if not price_string.endswith(' USD'):
#                     zero_result_explanation = f'Cannot handle price string {repr(price_string)} for {ticker_symbol}.'
#                     return rows, zero_result_explanation
#                 price = float(price_string.replace(' USD', '').replace(',', ''))

#                 row = (date_time, ticker_symbol, price)
#                 rows.append(row)
#                 seen_whole_time_strings.add(whole_time_string)
#         assert len(rows) != 0
#         return rows, zero_result_explanation

###########
# Scraper #
###########

async def gather_ticker_symbol_rows(ticker_symbol: str) -> Tuple[List[Tuple[datetime.datetime, str, float]], str]:
    coroutine = get_coroutine_to_gather_ticker_symbol_row()
    rows, zero_result_explanation = await coroutine(ticker_symbol)
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
        cursor.executemany('INSERT INTO stocks VALUES(?,?,?);', rows)
        connection.commit()
    
    connection.close()
    return

##########
# Driver #
##########
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-output-file', type=str, default=DEFFAULT_OUTPUT_DB_FILE, help='Output DB file.')
    parser.add_argument('-ticker-symbol-file', type=str, help='Newline-delimited file of ticker symbols.')
    args = parser.parse_args()

    stock_data_db_file = args.output_file
    ticker_symbol_file = args.ticker_symbol_file
    if ticker_symbol_file:
        with open(args.ticker_symbol_file, 'r') as f:
            ticker_symbols = eager_filter(len, map(str.strip, f.read().split('\n')))
    else:
        # ticker_symbols = gather_all_ticker_symbols()
        ticker_symbols = gather_rh_ticker_symbols()
    with timer('Data gathering'):
        with browser_pool_initialized():
            EVENT_LOOP.run_until_complete(update_stock_db(stock_data_db_file, ticker_symbols))

