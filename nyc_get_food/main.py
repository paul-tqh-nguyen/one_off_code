'#!/usr/bin/python3 -OO' # @todo enable this

'''
'''

# @todo update doc string

###########
# Imports #
###########

import os
import psutil
import asyncio
import pyppeteer
import itertools
import queue
import time
import warnings
import bs4
import random
import pandas as pd
import multiprocessing as mp
from statistics import mean
from typing import Union, List, Tuple, Iterable, Callable, Awaitable

from misc_utilities import *

# @todo review these imports to make sure everything is used

###########
# Globals #
###########

BROWSER_IS_HEADLESS = False # @hack clicking has problems when headless

TARGET_URL = 'https://dsny.maps.arcgis.com/apps/webappviewer/index.html?id=35901167a9d84fb0a2e0672d344f176f'

OUTPUT_CSV_FILE = './raw_data.csv'

##########################
# Web Scraping Utilities #
##########################

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
        global BROWSER
        updated_kwargs = kwargs.copy()
        pages = await BROWSER.pages()
        page = pages[-1]
        updated_kwargs['page'] = page
        result = await func(*args, **updated_kwargs)
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

async def _hide_sole_element(self: Union[pyppeteer.page.Page, pyppeteer.element_handle.ElementHandle], selector: str) -> None:
    element = await self.get_sole_element(selector)
    await self.evaluate('(element) => {element.style.display = "none";}', element)
    return 
setattr(pyppeteer.page.Page, 'hide_sole_element', _hide_sole_element)
setattr(pyppeteer.element_handle.ElementHandle, 'hide_sole_element', _hide_sole_element)

async def _remove_display_style_on_sole_element(self: Union[pyppeteer.page.Page, pyppeteer.element_handle.ElementHandle], selector: str) -> None:
    element = await self.get_sole_element(selector)
    await self.evaluate('(element) => {element.style.display = null;}', element)
    return 
setattr(pyppeteer.page.Page, 'remove_display_style_on_sole_element', _remove_display_style_on_sole_element)
setattr(pyppeteer.element_handle.ElementHandle, 'remove_display_style_on_sole_element', _remove_display_style_on_sole_element)

########################
# Get Food NYC Scraper #
########################

@trace
def process_location_html_string(html_string: str) -> dict:
    soup = bs4.BeautifulSoup(html_string,'html.parser')
    result = dict(location_type=only_one(soup.findAll('div', {'dojoattachpoint' : '_title'})).text)
    description_div = only_one(soup.findAll('div', {'dojoattachpoint' : '_description'}))
    for tag_index, tag in enumerate(description_div.findChildren('font' , recursive=False)):
        line = tag.text.replace('\xa0', ' ')
        if ' is located at ' in line:
            location_name, location_address = [child.text for child in tag.findChildren('font')]
            result['location_name'] = location_name
            result['location_address'] = location_address
            # @todo handle other cases of location_address where it might talk about programs included with the location.
        else:
            pass
            # raise ValueError(f'Unhandled html string:\n{html_string}')
    return result

async def zoom_to_circle(page: pyppeteer.page.Page, circle: pyppeteer.element_handle.ElementHandle, display_div: pyppeteer.element_handle.ElementHandle) -> None:
    await circle.click()
    await page.remove_display_style_on_sole_element('div.esriPopupWrapper')
    await page.remove_display_style_on_sole_element('div.outerPointer')
    await circle.click()
    zoom_to_link = await page.get_sole_element('a.action.zoomTo')
    await zoom_to_link.click()
    await page.hide_sole_element('div.esriPopupWrapper')
    await page.hide_sole_element('div.outerPointer')
    return

@scrape_function
async def _gather_location_display_df(*, page: pyppeteer.page.Page) -> pd.DataFrame:
    row_dicts: List[str] = []
    await page.goto(TARGET_URL)
    home_button = await page.get_sole_element('div.home')
    await home_button.click()
    await page.hide_sole_element('div#_5_panel')
    await page.hide_sole_element('div.esriPopupWrapper')
    initial_zooming_circles = await page.get_elements('g#DOE_active_hub_2222_layer circle')
    await initial_zooming_circles[0].click()
    await page.hide_sole_element('div.outerPointer')
    display_div = await page.get_sole_element('.esriPopupWrapper')
    
    assert len(row_dicts) == len(initial_zooming_circles)
    df = pd.DataFrame(row_dicts)
    return df

def gather_location_display_df() -> pd.DataFrame:
    df = EVENT_LOOP.run_until_complete(_gather_location_display_df())
    # @todo add lat longs
    # @todo add boroughs
    return df

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
    with timer('Data gathering'):
        df = gather_location_display_df()
    return

if __name__ == '__main__':
    gather_data()
