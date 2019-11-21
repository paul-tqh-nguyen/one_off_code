#!/usr/bin/python3 -O

"""

Utilities for named entity recognition via Wikidata scraping (to avoid banning).

Owner : paul-tqh-nguyen

Created : 11/19/2019

File Name : named_entity_recognition_via_wikidata.py

File Organization:
* Imports
* Utilities

"""

###########
# Imports #
###########

import requests
import urllib.parse
import asyncio
import pyppeteer
import unicodedata
import re
from typing import List, Tuple

#############
# Utilities #
#############

def _execute_async_task(task):
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    results = event_loop.run_until_complete(asyncio.gather(task))
    event_loop.close()
    assert len(results) == 1
    result = results[0]
    return result

WIKIDATA_SEARCH_URI_TEMPLATE = 'https://www.wikidata.org/w/index.php?sort=relevance&search={encoded_string}'

def _normalize_string_wrt_unicode(input_string: str) -> str:
    normalized_string = unicodedata.normalize('NFKD', input_string).encode('ascii', 'ignore').decode('utf-8')
    return normalized_string

async def _most_relevant_wikidata_entities_corresponding_to_string(input_string: str) -> str:
    wikidata_entities_corresponding_to_string = []
    browser = await pyppeteer.launch({'headless': True})
    page = await browser.newPage()
    input_string_encoded = urllib.parse.quote(input_string)
    uri = WIKIDATA_SEARCH_URI_TEMPLATE.format(encoded_string=input_string_encoded)
    try:
        await page.goto(uri)
        await page.waitForSelector('div#mw-content-text')
        await page.waitForSelector('ul.mw-search-results')
        search_results_divs = await page.querySelectorAll('div.mw-search-result-heading')
        for search_results_div in search_results_divs:
            search_results_div_text_content = await page.evaluate('(search_results_div) => search_results_div.textContent', search_results_div)
            parsable_text_match = re.match(r'^.+\(Q[0-9]+\) +$', search_results_div_text_content)
            if parsable_text_match:
                parsable_text = parsable_text_match.group()
                parsable_text = parsable_text.replace(')','')
                (label, term_id) = parsable_text.split('(')
                label = label.strip()
                term_id = term_id.strip()
                if _normalize_string_wrt_unicode(label).lower() == _normalize_string_wrt_unicode(input_string).lower():
                    wikidata_entities_corresponding_to_string.append(term_id)
                    if len(wikidata_entities_corresponding_to_string)>5:
                        break
    except pyppeteer.errors.NetworkError:
        pass
    finally:
        await browser.close()
    return wikidata_entities_corresponding_to_string

def string_corresponding_commonly_known_entities(input_string: str) -> List[str]:
    task = _most_relevant_wikidata_entities_corresponding_to_string(input_string)
    result = _execute_async_task(task)
    return result

ORGANIZATION_ID = 'Q43229'
QUERY_TEMPLATE_FOR_ENTITY_COMMONLY_KNOWN_ISAS = '''
SELECT ?VALID_GENLS ?TERM
WHERE 
{{
  VALUES ?TERM {{ {space_separated_term_ids} }}.
  ?TERM wdt:P31 ?IMMEDIATE_GENLS.
  ?IMMEDIATE_GENLS 	wdt:P279* ?VALID_GENLS.
  VALUES ?VALID_GENLS {{ '''+'wd:'+ORGANIZATION_ID+''' }}.
}}
'''
WIKI_DATA_QUERY_SERVICE_URI = 'https://query.wikidata.org'

async def _find_commonly_known_isas_via_web_scraper(term_ids_without_item_prefix:str) -> List[str]:
    term_type_pairs = []
    if len(term_ids_without_item_prefix) != 0:
        term_ids = map(lambda raw_term_id: 'wd:'+raw_term_id, term_ids_without_item_prefix)
        space_separated_term_ids = ' '.join(term_ids)
        sparql_query = QUERY_TEMPLATE_FOR_ENTITY_COMMONLY_KNOWN_ISAS.format(space_separated_term_ids=space_separated_term_ids)
        sparql_query_encoded = urllib.parse.quote(sparql_query)
        number_of_variables_queried = 2
        uri = WIKI_DATA_QUERY_SERVICE_URI+'/#'+sparql_query_encoded
        browser = await pyppeteer.launch({'headless': False})
        page = await browser.newPage()
        try:
            await page.goto(uri)
            selector_query_for_arbitrary_text_inside_query_box = 'span.cm-variable-2'
            await page.waitForSelector(selector_query_for_arbitrary_text_inside_query_box)
            button = await page.querySelector('button#execute-button')
            await page.evaluate('(button) => button.click()', button)
            await page.waitForSelector('a.item-link')
            column_header_divs = await page.querySelectorAll('div.th-inner.sortable.both')
            assert len(column_header_divs) == number_of_variables_queried
            variable_names = []
            for column_header_div in column_header_divs:
                variable_name = await page.evaluate('(column_header_div) => column_header_div.textContent', column_header_div)
                variable_names.append(variable_name)
            anchors = await page.querySelectorAll('a.item-link')
            current_term = None
            current_type = None
            for anchor_index, anchor in enumerate(anchors):
                anchor_variable = variable_names[anchor_index%number_of_variables_queried]
                anchor_link = await page.evaluate('(anchor) => anchor.href', anchor)
                assert len(re.findall(r"^http://www.wikidata.org/entity/\w+$", anchor_link))==1
                entity_id = anchor_link.replace('http://www.wikidata.org/entity/','')
                if anchor_variable=='TERM':
                    current_term = entity_id
                elif anchor_variable=='VALID_GENLS':
                    current_type = entity_id
                else:
                    raise Exception("Unexpected variable name {unexpected_variable_name}".format(unexpected_variable_name=anchor_variable))
                if bool(current_term) and bool(current_type):
                    term_type_pairs.append((current_term, current_type))
        except pyppeteer.errors.NetworkError:
            pass
        finally:
            await browser.close()
    return term_type_pairs

def find_commonly_known_isas(term_ids: List[str]) -> List[Tuple[str, str]]:
    task = _find_commonly_known_isas_via_web_scraper(term_ids)
    result = _execute_async_task(task)
    return result

def string_corresponding_wikidata_term_type_pairs(input_string: str) -> List[Tuple[str, str]]:
    term_ids = string_corresponding_commonly_known_entities(input_string)
    term_type_pairs = find_commonly_known_isas(term_ids)
    return term_type_pairs

def main():
    print("This module contains utilities for named entity recognition via Wikidata scraping.")

if __name__ == '__main__':
    main()
