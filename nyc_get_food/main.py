'#!/usr/bin/python3 -OO' # @todo enable this

'''
'''

# @todo update doc string

###########
# Imports #
###########

import os
import re
import json
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

BROWSER_IS_HEADLESS = False # location-based clicking in pyppeteer currently buggy

GET_FOOD_URL = 'https://dsny.maps.arcgis.com/apps/webappviewer/index.html?id=35901167a9d84fb0a2e0672d344f176f'
LAT_LONG_URL = 'https://www.latlong.net/convert-address-to-lat-long.html'

RAW_SCRAPED_DATA_JSON_FILE = './raw_scraped_data.json'
OUTPUT_JSON_FILE = './scraped_data.json'

##########################
# Web Scraping Utilities #
##########################

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

async def launch_browser() -> pyppeteer.browser.Browser:
    browser: pyppeteer.browser.Browser = await pyppeteer.launch(
        {
            'headless': BROWSER_IS_HEADLESS,
            'defaultViewport': None,
        },
        args=['--start-fullscreen'])
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

###################
# Sanity Checking #
###################

def sanity_check_raw_scraped_location_dict(location_dict: dict) -> None:
    assert len(location_dict) >= 4
    assert isinstance(location_dict['raw_html_string'], str) and len(location_dict['raw_html_string']) > 0
    assert isinstance(location_dict['location_type'], str) and len(location_dict['location_type']) > 0
    assert isinstance(location_dict['location_name'], str) and len(location_dict['location_name']) > 0
    assert isinstance(location_dict['location_address'], list) and len(location_dict['location_address']) > 0 and all(isinstance(address_line, str) for address_line in location_dict['location_address'])
    assert implies('location_operating_hours' in location_dict, isinstance(location_dict.get('location_operating_hours'), list) and len(location_dict.get('location_operating_hours')) > 0)
    assert implies('location_accepts_ebt' in location_dict, isinstance(location_dict.get('location_accepts_ebt'), bool))
    assert implies('location_halal_meal_availability' in location_dict, isinstance(location_dict.get('location_halal_meal_availability'), bool))
    assert implies('location_available_programs' in location_dict, isinstance(location_dict.get('location_available_programs'), list) and len(location_dict.get('location_available_programs')) > 0 and \
                   all(isinstance(program, str) and len(program) > 0 for program in location_dict.get('location_available_programs')))
    assert implies('location_accessibility' in location_dict, isinstance(location_dict.get('location_accessibility'), str) and len(location_dict.get('location_accessibility')) > 0)
    return 

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

def process_location_html_string(raw_html_string: str) -> dict:
    html_string = raw_html_string.replace('<b>', '').replace('</b>', '')
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    location_dict = dict(location_type=only_one(soup.findAll('div', {'dojoattachpoint' : '_title'})).text, raw_html_string=raw_html_string)
    description_div = only_one(soup.findAll('div', {'dojoattachpoint' : '_description'}))
    lines = simplify_newlines(description_div.get_text(separator='\n')).split('\n')
    current_tag_into_type: Literal[None, 'name', 'address', 'available_programs', 'operating_hours', 'halal_meal_availability', 'accessibility'] = 'name'
    for line in lines:
        line = line.replace('\xa0', ' ').strip()
        if line == 'is located at':
            current_tag_into_type = 'address'
        elif line == 'Operating hours are':
            current_tag_into_type = 'operating_hours'
        elif line == 'Halal':
            current_tag_into_type = 'halal_meal_availability'
        elif line == 'This site is':
            current_tag_into_type = 'accessibility'
        elif line in ('.', '', 'which has  programs.', 'To ensure every New York City resident can access nutritious meals, the Department of Education meal hub sites provide three meals a day, Monday through Friday, to both youth and adults in need. There is no registration or identification required.'):
            pass
        elif line.startswith('which has '):
            assert line.endswith(' programs.')
            programs_string = line.replace('which has ', '').replace(' programs.', '').replace(' and ', ', ')
            assert len(programs_string) > 0, f'{repr(line)} not correctly parsed.'
            programs = programs_string.split(', ')
            location_dict['location_available_programs'] = programs
            current_tag_into_type = None
        elif current_tag_into_type == 'name':
            location_dict['location_name'] = line
            current_tag_into_type = None
        elif current_tag_into_type == 'address':
            address = location_dict.get('location_address', [])
            address.append(line)
            location_dict['location_address'] = address
        elif current_tag_into_type == 'operating_hours':
            operating_hours_string = location_dict.get('location_operating_hours', [])
            location_dict['location_operating_hours'] = operating_hours_string
            location_dict['location_operating_hours'].append(line)
        elif current_tag_into_type == 'halal_meal_availability':
            assert line == 'meals available at this location.'
            location_dict['location_halal_meal_availability'] = True
            current_tag_into_type = None
        elif current_tag_into_type == 'accessibility':
            location_dict['location_accessibility'] = line
            current_tag_into_type = None
        else:
            raise ValueError(f'Unhandled html string (at {repr(line)}):\n{html_string}')
    ebt_string = ''.join(location_dict.get('location_operating_hours', []))
    if 'accepts EBT' in ebt_string:
        location_dict['location_accepts_ebt'] = True
    elif 'does not accept EBT' in ebt_string:
        location_dict['location_accepts_ebt'] = False
    else:
        assert 'ebt' not in ebt_string.lower(), ebt_string
    return location_dict

async def _scrape_location_dicts(page: pyppeteer.page.Page) -> List[dict]:
    location_dicts: List[str] = []
    await page.goto(GET_FOOD_URL)
    home_button = await page.get_sole_element('div.home')
    await home_button.click() # ensure consistent zoom level
    circles = await page.get_elements('div#map_gc circle')
    circles = circles[:10] # @todo remove this
    await circles[-1].click() # activate ERSI popup
    window_width, window_height = await page.evaluate('() => [window.innerWidth, window.innerHeight]')
    big_radius = window_width + window_height
    display_div = await page.get_sole_element('.esriPopupWrapper')
    for circle_index, circle in enumerate(circles):
        await page.evaluate(f'(element) => {{element.setAttribute("r", "{big_radius}px")}}', circle) # avoid obfuscated and overlapping circles by expanding the desired circle
        await home_button.click()
        await page.mouse.click(2 if bool(circle_index%2) else window_width - 5, window_height//2) # avoid double clicking while pinging at sub-double-click speed
        display_div_html_string = await page.evaluate('(element) => element.innerHTML', display_div)
        location_dict = process_location_html_string(display_div_html_string)
        sanity_check_raw_scraped_location_dict(location_dict)
        location_dicts.append(location_dict)
        await page.evaluate('(element) => {{element.setAttribute("r", "0px")}}',  circle) # hide processed circle to avoid obfuscation and overlap
    assert len(location_dicts) == len(circles)
    return location_dicts

async def _scrape_lat_longs(location_dicts: List[dict], page: pyppeteer.page.Page) -> List[dict]:
    await page.goto(LAT_LONG_URL)
    await page.waitForSelector('div.col-7.graybox')
    input_element_id = await page.evaluate('() => document.querySelector("div.col-7.graybox").querySelector("input").id')
    button_element = await page.get_sole_element('#btnfind')
    for location_dict in location_dicts:
        await page.type(f'input[id={input_element_id}]', location_dict['location_address'])
        await button_element.click()
        await page.waitForSelector('div.leaflet-popup-content')
        await page.waitForFunction('document.querySelector("div.leaflet-popup-content").innerText.length > 0')
        coordinate_string = await page.evaluate('document.querySelector("div.leaflet-popup-content").innerText')
        latitude, longitude = coordinate_string.split(', ')
        await page.evaluate('() => {document.querySelector("div.leaflet-popup-content").innerText = ''}')
        location_dict['latitude'] = latitude
        location_dict['longitude'] = longitude
        print()
        print(f"location_dict['location_address'] {repr(location_dict['location_address'])}")
        print(f"latitude {repr(latitude)}")
        print(f"longitude {repr(longitude)}")
    return location_dicts

async def _gather_location_dicts() -> List[dict]:
    browser = await launch_browser()
    page = only_one(await browser.pages())
    if os.path.isfile(RAW_SCRAPED_DATA_JSON_FILE):
        with open(RAW_SCRAPED_DATA_JSON_FILE, 'r') as json_file_handle:
            location_dicts = json.load(json_file_handle)
    else:
        location_dicts = await _scrape_location_dicts(page)
        with open(RAW_SCRAPED_DATA_JSON_FILE, 'w') as json_file_handle:
            json.dump(location_dicts, json_file_handle, indent=4)
    location_dicts = await _scrape_lat_longs(location_dicts, page)
    # @todo add boroughs
    browser_process = only_one([process for process in psutil.process_iter() if process.pid==browser.process.pid])
    for child_process in browser_process.children(recursive=True):
        child_process.kill()
    browser_process.kill() # @hack this line doesn't actually kill the process (or maybe it just doesn't free the PID?)
    return location_dicts

##########
# Driver #
##########

def gather_data() -> None:
    with timer('Data gathering'):
        location_dicts = EVENT_LOOP.run_until_complete(_gather_location_dicts())
    with open(OUTPUT_JSON_FILE, 'w') as json_file_handle:
        json.dump(location_dicts, json_file_handle, indent=4)
    breakpoint()
    return

if __name__ == '__main__':
    gather_data()
