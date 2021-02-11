
'''

load_buy_url('cgc', 1, 0.40)
load_sell_url('cgc', 1, 4321.0)

rm *db
conda activate stock_scraper
python3
import vanguard
from vanguard import *

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
import more_itertools
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
import keyring
import aioprocessing
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict, namedtuple
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from typing import Any, Union, List, Set, Tuple, Iterable, Callable, Awaitable, Coroutine, Dict, Generator
from typing_extensions import Literal

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

def execute_query(sql_query: str) -> List[Tuple]:
    with DB_INFO.db_access() as (db_connection, db_cursor):
        result = db_cursor.execute(sql_query).fetchall()
    return result
    
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

##############################
# Vanguard Browser Utilities #
##############################

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
    
    username_input_selector = 'input#username'
    password_input_selector = 'input#password'
    submit_button_selector = 'vui-button button.vui-button[type="submit"]'
    
    await page.waitForSelector(username_input_selector)
    await page.waitForSelector(password_input_selector)
    await page.waitForSelector(submit_button_selector)

    # import keyring, getpass
    # keyring.set_password('vanguard_stock_scraper', 'un', getpass.getpass(prompt='Username: ', stream=None))
    # keyring.set_password('vanguard_stock_scraper', 'pw', getpass.getpass(prompt='Password: ', stream=None))
        
    await page.focus(username_input_selector)
    await page.keyboard.type(keyring.get_password('vanguard_stock_scraper', 'un'))
    await page.focus(password_input_selector)
    await page.keyboard.type(keyring.get_password('vanguard_stock_scraper', 'pw'))
    await page.keyboard.press('Enter')

    return browser

@event_loop_func
def initialize_vanguard_browser() -> pyppeteer.browser.Browser:
    global VANGUARD_BROWSER
    assert VANGUARD_BROWSER is None, f'Vanguard Browser already initialized.'
    browser = run_awaitable(_initialize_vanguard_browser())
    VANGUARD_BROWSER = browser
    return browser

@event_loop_func
def close_vanguard_browser() -> None:
    global VANGUARD_BROWSER
    assert VANGUARD_BROWSER is not None, f'Vanguard browser already closed.'
    run_awaitable(VANGUARD_BROWSER._closeCallback())
    VANGUARD_BROWSER = None
    return

##############################
# Vanguard Scraper Utilities #
##############################

VANGUARD_BUY_SELL_URL = 'https://personal.vanguard.com/us/TradeTicket?investmentType=EQUITY'

async def _load_buy_sell_url(transaction_type: Literal['buy', 'sell'], ticker_symbol: str, number_of_shares: int, limit_price: float) -> None:
    assert transaction_type in ('buy', 'sell')
    page = only_one(await VANGUARD_BROWSER.pages())
    await page.goto(VANGUARD_BUY_SELL_URL)

    need_to_click_ok_button = await page.safelyWaitForSelector('input#okButtonInput', {'timeout': 1_000})
    if need_to_click_ok_button:
        ok_button = await page.get_sole_element('input#okButtonInput')
        await asyncio.sleep(1.0)
        await ok_button.click()

    await select_account(page)    
    await page.waitForSelector(' '.join([
        'div[id="baseForm:accountDetailTabBox:fundsAvailableNavBox"]',
        'table[id="baseForm:accountDetailTabBox:fundsAvailableTable"]',
        'tbody[id="baseForm:accountDetailTabBox:fundsAvailableTabletbody0"]',
        'tr[tbodyid="baseForm:accountDetailTabBox:fundsAvailableTabletbody0"]'
    ]))
    
    await select_transaction_type(page, transaction_type)
    
    await asyncio.sleep(1.0)
    await insert_ticker_symbol(page, ticker_symbol)
    
    await asyncio.sleep(1.0)
    await insert_number_of_shares(page, number_of_shares)
    
    await asyncio.sleep(1.0)
    await select_order_type(page)

    await asyncio.sleep(1.0)
    await insert_limit_price(page, limit_price)
    
    await asyncio.sleep(1.0)
    await select_duration(page)

    await asyncio.sleep(1.0)
    await select_cost_basis_method(page, transaction_type)

    await asyncio.sleep(1.0)
    await click_yes_buttons(page)

    await asyncio.sleep(1.0)
    await click_continue_button(page)

    await asyncio.sleep(1.0)
    await click_yes_buttons(page)

    await asyncio.sleep(1.0)
    await click_submit_button(page)
    
    return

async def select_account(page: pyppeteer.page.Page) -> None:
    account_dropdown_selector = ' '.join([
        'body',
        'table[id="baseForm:accountTable"]',
        'div[id="baseForm:accountSelectOne_label"]',
        'div[id="baseForm:accountSelectOne_cont"]',
        'table[id="baseForm:accountSelectOne-border"]',
        'tbody',
        'tr.vg-SelOneMenuVisRow',
        'td.vg-SelOneMenuIconCell',
    ])
    account_selector = ' '.join([
        'body',
        'div[id="menu-baseForm:accountSelectOne"].vg-SelOneMenuDropDown.vg-SelOneMenuNoWrap',
        'div[id="scroll-baseForm:accountSelectOne"].vg-SelOneMenuDropDownScroll',
        'table',
        'tbody',
        'tr',
        'td[id="baseForm:accountSelectOne:1"]',
    ])
    account_dropdown = await page.get_sole_element(account_dropdown_selector)
    await account_dropdown.click()
    account_option = await page.get_sole_element(account_selector)
    await account_option.click()

    return

async def select_transaction_type(page: pyppeteer.page.Page, transaction_type: Literal['buy', 'sell']) -> None:
    transaction_type_dropdown_selector = ' '.join([
        'body',
        'table[id="baseForm:transactionTypeTable"]',
        'div[id="baseForm:transactionTypeSelectOne_label"]',
        'div[id="baseForm:transactionTypeSelectOne_cont"]',
        'table[id="baseForm:transactionTypeSelectOne-border"]',
        'tbody',
        'tr.vg-SelOneMenuVisRow',
        'td.vg-SelOneMenuIconCell',
    ])

    if transaction_type == 'buy':
        transaction_type_selector = ' '.join([
            'body',
            'div[id="menu-baseForm:transactionTypeSelectOne"]',
            'table',
            'tbody',
            'tr',
            'td[id="baseForm:transactionTypeSelectOne:1"]',
        ])
    elif transaction_type == 'sell':
        transaction_type_selector = ' '.join([
            'body',
            'div[id="menu-baseForm:transactionTypeSelectOne"]',
            'table',
            'tbody',
            'tr',
            'td[id="baseForm:transactionTypeSelectOne:2"]',
        ])
    transaction_type_dropdown = await page.get_sole_element(transaction_type_dropdown_selector)
    await transaction_type_dropdown.click()
    transaction_type_option = await page.get_sole_element(transaction_type_selector)
    await transaction_type_option.click()
    
    if transaction_type == 'buy':
        await page.waitForSelector('li.current a[id="baseForm:accountDetailTabBox_tabBoxItemLink0"] span')
    elif transaction_type == 'sell':
        await page.waitForSelector('li.current a[id="baseForm:accountDetailTabBox_tabBoxItemLink1"] span')
    
    return

async def insert_ticker_symbol(page: pyppeteer.page.Page, ticker_symbol: str) -> None:
    ticker_symbol_input_selector = 'input[id="baseForm:investmentTextField"]'
    await page.focus(ticker_symbol_input_selector)
    await page.keyboard.type(ticker_symbol)
    await page.keyboard.press('Enter')
    return

async def insert_number_of_shares(page: pyppeteer.page.Page, number_of_shares: int) -> None:
    number_of_shares_input_selector = 'input[id="baseForm:shareQuantityTextField"]'
    await page.focus(number_of_shares_input_selector)
    await page.keyboard.type(str(number_of_shares))
    await page.keyboard.press('Enter')
    return

async def select_order_type(page: pyppeteer.page.Page) -> None:
    order_type_dropdown_selector = ' '.join([
        'body',
        'table[id="baseForm:orderTypeTable"]',
        'div[id="baseForm:orderTypeSelectOne_label"]',
        'div[id="baseForm:orderTypeSelectOne_cont"]',
        'table[id="baseForm:orderTypeSelectOne-border"]',
        'tbody',
        'tr.vg-SelOneMenuVisRow',
        'td.vg-SelOneMenuIconCell',
    ])
    order_type_selector = ' '.join([
        'body',
        'div[id="menu-baseForm:orderTypeSelectOne"].vg-SelOneMenuDropDown.vg-SelOneMenuNoWrap',
        'div[id="scroll-baseForm:orderTypeSelectOne"].vg-SelOneMenuDropDownScroll',
        'table',
        'tbody',
        'tr',
        'td[id="baseForm:orderTypeSelectOne:2"]'
    ])
    order_type_dropdown = await page.get_sole_element(order_type_dropdown_selector)
    await order_type_dropdown.click()
    order_type_option = await page.get_sole_element(order_type_selector)
    await order_type_option.click()

    return

async def insert_limit_price(page: pyppeteer.page.Page, limit_price: float) -> None:
    limit_price_input_selector = 'input[id="baseForm:limitPriceTextField"]'
    await page.focus(limit_price_input_selector)
    await page.keyboard.type(str(limit_price))
    await page.keyboard.press('Enter')
    return

async def select_duration(page: pyppeteer.page.Page) -> None:
    duration_dropdown_selector = ' '.join([
        'body',
        'table[id="baseForm:durationTypeTable"]',
        'div[id="baseForm:durationTypeSelectOne_label"]',
        'div[id="baseForm:durationTypeSelectOne_cont"]',
        'table[id="baseForm:durationTypeSelectOne-border"]',
        'tbody',
        'tr.vg-SelOneMenuVisRow',
        'td.vg-SelOneMenuIconCell',
    ])
    duration_selector = ' '.join([
        'body',
        'div[id="menu-baseForm:durationTypeSelectOne"].vg-SelOneMenuDropDown.vg-SelOneMenuNoWrap',
        'div[id="scroll-baseForm:durationTypeSelectOne"].vg-SelOneMenuDropDownScroll',
        'table',
        'tbody',
        'tr',
        'td[id="baseForm:durationTypeSelectOne:1"]'
    ])
    duration_dropdown = await page.get_sole_element(duration_dropdown_selector)
    await duration_dropdown.click()
    duration_option = await page.get_sole_element(duration_selector)
    await duration_option.click()

    return

async def select_cost_basis_method(page: pyppeteer.page.Page, transaction_type: Literal['buy', 'sell']) -> None:
    if transaction_type == 'sell':
        await page.waitForSelector('a[id="baseForm:costBasisMethodLearnMoreLink"]')
        cost_basis_method_dropdown_selector = ' '.join([
            'body',
            'table[id="baseForm:costBasisMethodTable"]',
            'div[id="baseForm:costBasisMethodSelectOne_label"]',
            'div[id="baseForm:costBasisMethodSelectOne_cont"]',
            'table[id="baseForm:costBasisMethodSelectOne-border"]',
            'tbody',
            'tr.vg-SelOneMenuVisRow',
            'td.vg-SelOneMenuIconCell',
        ])
        cost_basis_method_selector = ' '.join([
            'body',
            'div[id="menu-baseForm:costBasisMethodSelectOne"].vg-SelOneMenuDropDown.vg-SelOneMenuNoWrap',
            'div[id="scroll-baseForm:costBasisMethodSelectOne"].vg-SelOneMenuDropDownScroll',
            'table',
            'tbody',
            'tr',
            'td[id="baseForm:costBasisMethodSelectOne:2"]'
        ])
        cost_basis_method_dropdown = await page.get_sole_element(cost_basis_method_dropdown_selector)
        await cost_basis_method_dropdown.click()
        cost_basis_method_option = await page.get_sole_element(cost_basis_method_selector)
        await cost_basis_method_option.click()
    return

async def click_yes_buttons(page: pyppeteer.page.Page) -> None:
    
    need_to_click_yes_button = await page.safelyWaitForSelector('input[id="orderCaptureWarningLayerForm:yesButtonInput"]', {'timeout': 500})
    if need_to_click_yes_button:
        yes_button = await page.get_sole_element('input[id="orderCaptureWarningLayerForm:yesButtonInput"]')
        await yes_button.click()
    
    need_to_click_yes_button = await page.safelyWaitForSelector('span[id="comp-orderCaptureWarningLayerForm:yesButton"]', {'timeout': 500})
    if need_to_click_yes_button:
        await page.evaluate(f'''() => document.querySelector('span[id="comp-orderCaptureWarningLayerForm:yesButton"] input').click()''')
    
    return

async def click_continue_button(page: pyppeteer.page.Page) -> None:
    continue_button_selector = 'input[id="baseForm:reviewButtonInput"]'
    continue_button = await page.get_sole_element(continue_button_selector)
    await continue_button.click()    
    return

async def click_submit_button(page: pyppeteer.page.Page) -> None:
    submit_button_selector = 'input[id="baseForm:submitButtonInput"]'
    submit_button = await page.get_sole_element(submit_button_selector)
    await submit_button.click()    
    return

def load_buy_sell_url(transaction_type: Literal['buy', 'sell'], ticker_symbol: str, number_of_shares: int, limit_price: float) -> None:
    assert VANGUARD_BROWSER is not None, f'Vanguard browser not initialized.'
    run_awaitable(_load_buy_sell_url(transaction_type))
    return

def load_buy_url(ticker_symbol: str, number_of_shares: int, limit_price: float) -> None:
    run_awaitable(_load_buy_sell_url('buy', ticker_symbol, number_of_shares, limit_price))
    return

def load_sell_url(ticker_symbol: str, number_of_shares: int, limit_price: float) -> None:
    run_awaitable(_load_buy_sell_url('sell', ticker_symbol, number_of_shares, limit_price))
    return

########################
# RH Browser Utilities #
########################

# @event_loop_func
# def track_ticker_symbols(*ticker_symbols: List[str]) -> None:
#     global RH_BROWSER
#     global RH_SCRAPER_TASK
#     assert RH_BROWSER is not None, f'RH browser not initialized.'
#     if RH_SCRAPER_TASK is not None:
#         RH_SCRAPER_TASK.cancel()
#     run_awaitable(open_pages_for_ticker_symbols(ticker_symbols))
#     RH_SCRAPER_TASK = enqueue_awaitable(scrape_ticker_symbols())
#     return

##########
# Driver #
##########

DEFAULT_TICKER_SYMBOLS = ('TWTR', 'TSLA', "IT", "T", "F", "COF", "GOOGL", "AAPL", "ZM", "SAVE", "AMC", "BB", "COF", "CMG", "XOM", "TGT", "NKE", "ULTA", "NFLX")

@one_time_execution
def initialize_browsers() -> None:
    initialize_vanguard_browser()
    # track_ticker_symbols(*DEFAULT_TICKER_SYMBOLS)
    return

@atexit.register
def module_exit_callback() -> None:
    close_vanguard_browser()
    stop_thread_for_event_loop()
    return

if __name__ == '__main__':
    pass # TODO do something here
