'#!/usr/bin/python3 -OO' # @todo enable this

'''
'''

# @todo update doc string

###########
# Imports #
###########

import os
import re
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
from typing_extensions import Literal
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

########################
# Get Food NYC Scraper #
########################

def simplify_newlines(input_string: str) -> str:
    output_string = input_string
    for _ in range(len(input_string)):
        if '\n\n' not in output_string:
            break
        output_string = output_string.replace('\n\n', '\n')
    return output_string

def process_location_html_string(html_string: str) -> dict:
    html_string = html_string.replace('<b>', '').replace('</b>', '')
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    result = dict(location_type=only_one(soup.findAll('div', {'dojoattachpoint' : '_title'})).text)
    description_div = only_one(soup.findAll('div', {'dojoattachpoint' : '_description'}))
    lines = simplify_newlines(description_div.get_text(separator='\n')).split('\n')
    current_tag_into_type: Literal[None, 'name', 'address', 'available_programs', 'operating_hours', 'halal_meal_availability', 'accessibility'] = 'name'
    print()
    print(f"html_string {repr(html_string)}")
    print()
    p1(lines)
    print()
    for line in lines:
        line = line.replace('\xa0', ' ').strip()
        print(f"line {repr(line)}")
        if line == 'is located at':
            current_tag_into_type = 'address'
        elif line == 'Operating hours are':
            current_tag_into_type = 'operating_hours'
        elif line == 'Halal':
            current_tag_into_type = 'halal_meal_availability'
        elif line == 'This site is':
            current_tag_into_type = 'accessibility'
        elif line in ('.', '',) \
             or re.fullmatch(r'^(New York|Queens|Bronx|Manhattan|Staten Island|Brooklyn),? NY [0-9][0-9][0-9][0-9][0-9]$', line) is not None \
             or (any(line.startswith(area) for area in {'New York', 'Queens', 'Bronx', 'Manhattan', 'Staten Island', 'Arverne', 'Brooklyn'}) and list(result.keys())[-1] == 'location_address') \
             or line == 'To ensure every New York City resident can access nutritious meals, the Department of Education meal hub sites provide three meals a day, Monday through Friday, to both youth and adults in need. There is no registration or identification required.':
            pass
        elif line.startswith('which has '):
            assert line.endswith(' programs.')
            progams_string = line.replace('which has ', '').replace(' programs.', '')
            result['location_available_programs'] = progams_string
            current_tag_into_type = None
        elif current_tag_into_type == 'name':
            result[f'location_name'] = line
            current_tag_into_type = None
        elif current_tag_into_type == 'address':
            result[f'location_address'] = line
            current_tag_into_type = None
        elif current_tag_into_type == 'operating_hours':
            operating_hours_string = result.get('location_operating_hours', [])
            result['location_operating_hours'] = operating_hours_string
            result['location_operating_hours'].append(line)
        elif current_tag_into_type == 'halal_meal_availability':
            assert line == 'meals available at this location.'
            result[f'location_halal_meal_availability'] = True
            current_tag_into_type = None
        elif current_tag_into_type == 'accessibility':
            result[f'location_accessibility'] = line
            current_tag_into_type = None
        elif line in {'121-12 Liberty Ave (Office)', }:
            pass # @todo get rid of this section
        else:
            print(f"current_tag_into_type {repr(current_tag_into_type)}")
            time.sleep(7200)
            raise ValueError(f'Unhandled html string (at {repr(line)}):\n{html_string}')
    print(f"result  {repr(result )}")
    return result

@scrape_function
async def _gather_location_display_df(*, page: pyppeteer.page.Page) -> pd.DataFrame:
    row_dicts: List[str] = []
    await page.goto(TARGET_URL)
    home_button = await page.get_sole_element('div.home')
    await home_button.click()
    circles = await page.get_elements('div#map_gc circle')
    random.seed(2) # @todo remove this
    random.shuffle(circles)
    await circles[-1].click()
    window_width = await page.evaluate('() => window.innerWidth')
    window_height = await page.evaluate('() => window.innerHeight')
    big_radius = window_width + window_height
    display_div = await page.get_sole_element('.esriPopupWrapper')
    for circle_index, circle in enumerate(circles):
        await page.evaluate(f'(element) => {{element.setAttribute("r", "{big_radius}px")}}', circle)
        await home_button.click()
        await page.mouse.click(2 if bool(circle_index%2) else window_width - 5, window_height//2)
        display_div_html_string = await page.evaluate('(element) => element.innerHTML', display_div)
        row_dicts.append(process_location_html_string(display_div_html_string))
        await page.evaluate('(element) => {{element.setAttribute("r", "0px")}}',  circle)
    assert len(row_dicts) == len(circles)
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
