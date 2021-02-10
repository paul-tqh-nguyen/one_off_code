
'''

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
    await page.keyboard.type(keyring.get_password('vanguard_stock_scraper', 'un'));
    await page.focus(password_input_selector)
    await page.keyboard.type(keyring.get_password('vanguard_stock_scraper', 'pw'));
    await page.keyboard.press('Enter');

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

def get_js_func_string_for_clicking_buy_sell_dropdowns(dropdown_selector: str, choice_selector: str, expected_inner_text: str) -> str:
    program_string = f'''
() => {{

const account_dropdowns = document.querySelectorAll('{dropdown_selector}');

if (account_dropdowns.length != 1) {{
    return false;
}}

account_dropdowns[0].click();

const account_elems = document.querySelectorAll('{choice_selector}');

if ((account_elems.length != 1) || (account_elems[0].innerText != "{expected_inner_text}")) {{
    return false;
}}

account_elems[0].click();

return true;
}}
'''
    return program_string

async def _load_buy_sell_url(transaction_type: Literal['buy', 'sell'], ticker_symbol: str, number_of_shares: int, limit_price: float) -> None:
    assert transaction_type in ('buy', 'sell')
    page = only_one(await VANGUARD_BROWSER.pages())
    await page.goto(VANGUARD_BUY_SELL_URL)

    need_to_click_ok_button = await page.safelyWaitForSelector('input#okButtonInput', {'timeout': 1_000})
    if need_to_click_ok_button:
        await page.evaluate(f'''() => document.querySelector('input#okButtonInput').click()''')
    
    # Select Account
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
    await page.waitForSelector(account_selector)
    account_selection_success = await page.evaluate(
        get_js_func_string_for_clicking_buy_sell_dropdowns(
            account_dropdown_selector,
            account_selector,
            'Paul Nguyen—Brokerage Account—74046904 (Margin)'
        )
    )
    assert account_selection_success
    
    # Select Transaction Type
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
        transaction_selector = ' '.join([
            'body',
            'div[id="menu-baseForm:transactionTypeSelectOne"]',
            'table',
            'tbody',
            'tr',
            'td[id="baseForm:transactionTypeSelectOne:1"]',
        ]).replace(':', '\\\\:')
        expected_inner_text = 'Buy'
    elif transaction_type == 'sell':
        transaction_selector = ' '.join([
            'body',
            'div[id="menu-baseForm:transactionTypeSelectOne"]',
            'table',
            'tbody',
            'tr',
            'td[id="baseForm:transactionTypeSelectOne:2"]',
        ]).replace(':', '\\\\:')
        expected_inner_text = 'Sell'
    await page.waitForSelector(' '.join([
        'div[id="baseForm:accountDetailTabBox:fundsAvailableNavBox"]',
        'table[id="baseForm:accountDetailTabBox:fundsAvailableTable"]',
        'tbody[id="baseForm:accountDetailTabBox:fundsAvailableTabletbody0"]',
        'tr[tbodyid="baseForm:accountDetailTabBox:fundsAvailableTabletbody0"]'
    ]))
    transaction_type_selection_success = await page.evaluate(
        get_js_func_string_for_clicking_buy_sell_dropdowns(
            transaction_type_dropdown_selector,
            transaction_selector,
            expected_inner_text
        )
    )
    assert transaction_type_selection_success
    
    # Select Duration
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
    duration_selection_success = await page.evaluate(
        get_js_func_string_for_clicking_buy_sell_dropdowns(
            duration_dropdown_selector,
            duration_selector,
            'Day'
        )
    )
    assert duration_selection_success

    # Select Order Type
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
    order_type_selection_success = await page.evaluate(
        get_js_func_string_for_clicking_buy_sell_dropdowns(
            order_type_dropdown_selector,
            order_type_selector,
            'Limit'
        )
    )
    assert order_type_selection_success
    
    # Insert Input Box Text
    await page.waitForSelector('input[id="baseForm:limitPriceTextField"]')
    text_insertion_success = await page.evaluate(f'''
() => {{

const ticker_symbol_inputs = document.querySelectorAll('input[id="baseForm:investmentTextField"]');
const number_of_shares_inputs = document.querySelectorAll('input[id="baseForm:shareQuantityTextField"]')
const limit_price_inputs = document.querySelectorAll('input[id="baseForm:limitPriceTextField"]')

if (ticker_symbol_inputs.length != 1 || number_of_shares_inputs.length != 1 || limit_price_inputs.length != 1) {{
    return false;
}}

ticker_symbol_inputs[0].value = '{ticker_symbol}'
number_of_shares_inputs[0].value = '{number_of_shares}'
limit_price_inputs[0].value = '{limit_price}'

return true;
}}
''')
    assert text_insertion_success

    # Select Account Type
    security_account_type_dropdown_selector = ' '.join([
        'body',
        'table[id="baseForm:securityAccountTypeTable"]',
        'div[id="baseForm:securityAccountTypeSelectOne_label"]',
        'div[id="baseForm:securityAccountTypeSelectOne_cont"]',
        'table[id="baseForm:securityAccountTypeSelectOne-border"]',
        'tbody',
        'tr.vg-SelOneMenuVisRow',
        'td.vg-SelOneMenuIconCell',
    ])
    need_to_select_account_type = await page.safelyWaitForSelector(security_account_type_dropdown_selector, {'timeout': 1_000})
    if need_to_select_account_type:
        security_account_type_selector = ' '.join([
            'body',
            'div[id="menu-baseForm:securityAccountTypeSelectOne"].vg-SelOneMenuDropDown.vg-SelOneMenuNoWrap',
            'div[id="scroll-baseForm:securityAccountTypeSelectOne"].vg-SelOneMenuDropDownScroll',
            'table',
            'tbody',
            'tr',
            'td[id="baseForm:securityAccountTypeSelectOne:2"]'
        ])
        security_account_type_selection_success = await page.evaluate(
            get_js_func_string_for_clicking_buy_sell_dropdowns(
                security_account_type_dropdown_selector,
                security_account_type_selector,
                'Margin'
            )
        )
        assert security_account_type_selection_success
    
    # Select Cost Basis Method
    if transaction_type == 'sell':
        if await page.safelyWaitForSelector('a[id="baseForm:costBasisMethodLearnMoreLink"]', {'timeout': 1_000}):
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
            cost_basis_method_selection_success = await page.evaluate(
                get_js_func_string_for_clicking_buy_sell_dropdowns(
                    cost_basis_method_dropdown_selector,
                    cost_basis_method_selector,
                    'First In, First Out (FIFO)'
                )
            )
            assert cost_basis_method_selection_success
    
#     continue_button_selector = 'input[id="baseForm:reviewButtonInput"]'
#     page.waitForSelector(continue_button_selector)
#     continue_button_press_success = await page.evaluate(f'''
# () => {{

# const continue_buttons = document.querySelectorAll('{continue_button_selector}');

# if (continue_buttons.length != 1) {{
#     return false;
# }}

# continue_buttons[0].click();

# return true;
# }}
# ''')
#     assert continue_button_press_success

#     need_to_click_yes_button = await page.safelyWaitForSelector('span[id="comp-orderCaptureWarningLayerForm:yesButton"]', {'timeout': 1_000})
#     if need_to_click_yes_button:
#         await page.evaluate(f'''() => document.querySelector('span[id="comp-orderCaptureWarningLayerForm:yesButton"] input').click()''')

#     await page.safelyWaitForNavigation({'timeout': 1_000})
    
#     submit_button_selector = 'input[id="baseForm:submitButtonInput"]'
#     page.waitForSelector(submit_button_selector)
#     submit_button_press_success = await page.evaluate(f'''
# () => {{

# const submission_buttons = document.querySelectorAll('{submit_button_selector}');

# if (submission_buttons.length != 1) {{
#     return false;
# }}

# submission_buttons[0].click();

# return true;
# }}
# ''')
#     assert submit_button_press_success

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
