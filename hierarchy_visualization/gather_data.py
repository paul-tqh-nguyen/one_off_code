#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

'''
'''

# @todo update doc string

###########
# Imports #
###########

import re
import asyncio
import pyppeteer
import warnings
import urllib
import time
import psutil
from typing import Awaitable, List

from misc_utilities import *

###########
# Globals #
###########

OUTPUT_CSV_FILE = './data.csv'

START_NODE_LINK = 'https://www.wikidata.org/wiki/Q10884' # tree

BROWSER_IS_HEADLESS = False
MAX_NUMBER_OF_NEW_PAGE_ATTEMPTS = 50
NUMBER_OF_ATTEMPTS_PER_SLEEP = 3
SLEEPING_RANGE_SLEEP_TIME= 10

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
    browser: pyppeteer.browser.Browser = await pyppeteer.launch({'headless': BROWSER_IS_HEADLESS,})
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
                browser_process.kill() # @hack memory leak ; this line doesn't actually kill the process (or maybe it just doesn't free the PID?) until the encompassing python process closes
                BROWSER = await _launch_browser()
            except Exception as err:
                raise
            if result != unique_bogus_result_identifier:
                break
        if result == unique_bogus_result_identifier:
            raise Exception
        return result
    return decorating_function

######################
# Wikidata Utilities #
######################

WIKI_DATA_QUERY_SERVICE_URI = 'https://query.wikidata.org'

@scrape_function
async def _query_wikidata_via_web_scraper(sparql_query:str, *, page: pyppeteer.page.Page) -> List[dict]:
    results = []
    sparql_query_encoded = urllib.parse.quote(sparql_query)
    uri = WIKI_DATA_QUERY_SERVICE_URI+'/#'+sparql_query_encoded
    await page.goto(uri)
    selector_query_for_arbitrary_text_inside_query_box = 'span.cm-variable-2'
    await page.waitForSelector(selector_query_for_arbitrary_text_inside_query_box)
    button = await page.querySelector('button#execute-button')
    await page.evaluate('(button) => button.click()', button)
    await page.waitForSelector('div.th-inner.sortable.both.asc')
    column_header_divs = await page.querySelectorAll('div.th-inner.sortable.both.asc')
    number_of_variables_queried = len(column_header_divs)
    variable_names = []
    for column_header_div in column_header_divs:
        variable_name = await page.evaluate('(column_header_div) => column_header_div.textContent', column_header_div)
        variable_names.append(variable_name)
    anchors = await page.querySelectorAll('a.item-link')
    result = dict()
    for anchor_index, anchor in enumerate(anchors):
        anchor_variable = variable_names[anchor_index%number_of_variables_queried]
        anchor_link = await page.evaluate('(anchor) => anchor.href', anchor)
        assert len(re.findall(r'^http://www.wikidata.org/entity/\w+$', anchor_link))==1
        entity_id = anchor_link.replace('http://www.wikidata.org/entity/','')
        anchor_variable_with_question_mark_prefix = '?'+anchor_variable
        result[anchor_variable_with_question_mark_prefix] = entity_id
        if (1+anchor_index)%number_of_variables_queried==0:
            assert len(result) == number_of_variables_queried
            results.append(result)
            result = dict()
    return results

def execute_sparql_query_via_wikidata(sparql_query:str) -> List[dict]:
    return EVENT_LOOP.run_until_complete(_query_wikidata_via_web_scraper(sparql_query))

######################################
# Domain Specific Wikidata Utilities #
######################################

INSTANCE_OF = 'wdt:P31'
SUBCLASS_OF = 'wdt:P279'

def get_number_of_instances(class_term: str) -> int:
    query = f'''
SELECT (count(?INSTANCE) as ?NUM_INSTANCES) WHERE {{
  ?INSTANCE {INSTANCE_OF} ?INSTANCE.
}}
'''
    query_results = execute_sparql_query_via_wikidata(query)
    number_of_instances = only_one(query_results)['?NUM_INSTANCES']
    return number_of_instances

##########
# Driver #
##########

@debug_on_error
def gather_data() -> None:
    get_number_of_instances('Q146')
    return

if __name__ == '__main__':
    gather_data()
