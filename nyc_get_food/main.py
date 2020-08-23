'#!/usr/bin/python3 -OO' # @todo enable this

"""
"""

# @todo update doc string

###########
# Imports #
###########

import os
import psutil
import asyncio
import pyppeteer
import itertools
import time
import warnings
import pandas as pd
from statistics import mean
from typing import Union, List, Tuple, Iterable, Callable, Awaitable

from misc_utilities import *

# @todo review these imports to make sure everything is used

###########
# Globals #
###########

MAX_NUMBER_OF_NEW_PAGE_ATTEMPTS = 1000
NUMBER_OF_ATTEMPTS_PER_SLEEP = 1
SLEEPING_RANGE_SLEEP_TIME = 10
BROWSER_IS_HEADLESS = False

TARGET_URL = "https://dsny.maps.arcgis.com/apps/webappviewer/index.html?id=35901167a9d84fb0a2e0672d344f176f"

OUTPUT_CSV_FILE = './raw_data.csv'

##########################
# Web Scraping Utilities #
##########################

def _sleeping_range(upper_bound: int):
    for attempt_index in range(upper_bound):
        if attempt_index and attempt_index % NUMBER_OF_ATTEMPTS_PER_SLEEP == 0:
            time.sleep(SLEEPING_RANGE_SLEEP_TIME*(attempt_index//NUMBER_OF_ATTEMPTS_PER_SLEEP))
        yield attempt_index

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

async def _launch_browser() -> pyppeteer.browser.Browser:
    browser: pyppeteer.browser.Browser = await pyppeteer.launch(
        {
            'headless': BROWSER_IS_HEADLESS,
            'defaultViewport': None,
        },
        args=['--start-fullscreen'])
    return browser

BROWSER = EVENT_LOOP.run_until_complete(_launch_browser())

def scrape_function(func: Awaitable) -> Awaitable:
    async def decorating_function(*args, **kwargs):
        unique_bogus_result_identifier = object()
        result = unique_bogus_result_identifier
        global BROWSER
        for _ in _sleeping_range(MAX_NUMBER_OF_NEW_PAGE_ATTEMPTS):
            try:
                updated_kwargs = kwargs.copy()
                pages = await BROWSER.pages()
                page = pages[-1]
                updated_kwargs['page'] = page
                result = await func(*args, **updated_kwargs)
            except (pyppeteer.errors.BrowserError,
                    pyppeteer.errors.ElementHandleError,
                    pyppeteer.errors.NetworkError,
                    pyppeteer.errors.PageError,
                    pyppeteer.errors.PyppeteerError) as err:
                warnings.warn(f'\n{time.strftime("%m/%d/%Y_%H:%M:%S")} {func.__name__} {err}')
                warnings.warn(f'\n{time.strftime("%m/%d/%Y_%H:%M:%S")} Launching new page.')
                await BROWSER.newPage()
            except pyppeteer.errors.TimeoutError as err:
                warnings.warn(f'\n{time.strftime("%m/%d/%Y_%H:%M:%S")} {func.__name__} {err}')
                warnings.warn(f'\n{time.strftime("%m/%d/%Y_%H:%M:%S")} Launching new browser.')
                browser_process = only_one([process for process in psutil.process_iter() if process.pid==BROWSER.process.pid])
                for child_process in browser_process.children(recursive=True):
                    child_process.kill()
                browser_process.kill() # @hack memory leak ; this line doesn't actually kill the process (or maybe it just doesn't free the PID?)
                BROWSER = await _launch_browser()
            except Exception as err:
                raise
            if result != unique_bogus_result_identifier:
                break
        if result == unique_bogus_result_identifier:
            raise Exception
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

########################
# Get Food NYC Scraper #
########################

async def hide_welcome_div(page: pyppeteer.page.Page) -> None:
    welcome_div = await page.get_sole_element('div#_5_panel')
    await page.evaluate('(element) => {element.style.display = "none"}', welcome_div)
    return

@scrape_function
async def _gather_location(*, page: pyppeteer.page.Page) -> None:
    await page.goto(TARGET_URL)
    await hide_welcome_div(page) # strictly for debugging
    display_div = await page.get_sole_element('.esriPopupWrapper')
    circles = await page.get_elements('g#DOE_active_hub_2222_layer circle')
    for circle in circles:
        await circle.click()
        display_div_html = await page.evaluate('(element) => element.innerHTML', display_div)
    return None

def gather() -> None:
    results = EVENT_LOOP.run_until_complete(_gather())
    time.sleep(10)
    return 

###################
# Sanity Checking #
###################

def sanity_check_output_csv_file() -> None:
    pass # @todo fill this in
    return 

##########
# Driver #
##########

def gather_data() -> None:
    with timer("Data gathering"):
        gather()
        sanity_check_output_csv_file()
    return

if __name__ == '__main__':
    gather_data()
