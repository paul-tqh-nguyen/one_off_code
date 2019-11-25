#!/usr/bin/python3 -O

"""

Utilities for named entity recognition via Wikidata scraping (to avoid banning).

Owner : paul-tqh-nguyen

Created : 11/19/2019

File Name : named_entity_recognition_via_wikidata.py

File Organization:
* Imports
* Async IO Utilities
* Wikidata Search Utilities
* Wikidata Query Service Utilities
* Most Abstract Interface

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
import bidict
import string
from functools import lru_cache
from typing import List, Tuple, Set

######################
# Async IO Utilities #
######################

def _execute_async_task(task):
    event_loop = asyncio.new_event_loop()
    results = None
    try:
        asyncio.set_event_loop(event_loop)
        results = event_loop.run_until_complete(asyncio.gather(task))
    except Exception as err:
        pass
    finally:
        event_loop.close()
    import time; time.sleep(4); print("done sleeping")
    assert len(results) == 1
    result = results[0]
    return result

#############################
# Wikidata Search Utilities #
#############################

WIKIDATA_SEARCH_URI_TEMPLATE = 'https://www.wikidata.org/w/index.php?sort=relevance&search={encoded_string}'

def _normalize_string_wrt_unicode(input_string: str) -> str:
    normalized_string = unicodedata.normalize('NFKD', input_string).encode('ascii', 'ignore').decode('utf-8')
    return normalized_string

PUNUCTION_REMOVING_TRANSLATION_TABLE = str.maketrans('', '', string.punctuation)

def _normalize_string_for_wikidata_entity_label_comparison(input_string: str) -> str:
    normalized_string = input_string
    normalized_string = _normalize_string_wrt_unicode(normalized_string)
    normalized_string = normalized_string.lower()
    normalized_string = normalized_string.translate(PUNUCTION_REMOVING_TRANSLATION_TABLE)
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
        search_results_div = await page.waitForSelector('div.searchresults')
        search_results_paragraph_elements = await search_results_div.querySelectorAll('p')
        search_results_have_shown_up = None
        for paragraph_element in search_results_paragraph_elements:
            paragraph_element_classname_string = await page.evaluate('(p) => p.className', paragraph_element)
            paragraph_element_classnames = paragraph_element_classname_string.split(' ')
            for paragraph_element_classname in paragraph_element_classnames:
                if paragraph_element_classname == 'mw-search-nonefound':
                    search_results_have_shown_up = False
                elif paragraph_element_classname == 'mw-search-pager-bottom':
                    search_results_have_shown_up = True
                if search_results_have_shown_up is not None:
                    break
            if search_results_have_shown_up is not None:
                break
        if search_results_have_shown_up:
            search_results_divs = await page.querySelectorAll('div.mw-search-result-heading')
            for search_results_div in search_results_divs:
                search_results_div_text_content = await page.evaluate('(search_results_div) => search_results_div.textContent', search_results_div)
                parsable_text_match = re.match(r'^.+\(Q[0-9]+\) +$', search_results_div_text_content)
                if parsable_text_match:
                    parsable_text = parsable_text_match.group()
                    parsable_text = parsable_text.replace(')','')
                    parsable_text_parts = parsable_text.split('(')
                    if len(parsable_text_parts)==2:
                        (label, term_id) = parsable_text_parts
                        label = label.strip()
                        term_id = term_id.strip()
                        if _normalize_string_for_wikidata_entity_label_comparison(label) == _normalize_string_for_wikidata_entity_label_comparison(input_string):
                            wikidata_entities_corresponding_to_string.append(term_id)
                            if len(wikidata_entities_corresponding_to_string)>5:
                                break
    except pyppeteer.errors.NetworkError:
        pass
    finally:
        await browser.close()
    return wikidata_entities_corresponding_to_string

def string_corresponding_commonly_known_entities(input_string: str) -> List[str]:
    print()
    print("string_corresponding_commonly_known_entities")
    print("input_string {}".format(input_string))
    task = _most_relevant_wikidata_entities_corresponding_to_string(input_string)
    result = _execute_async_task(task)
    return result

####################################
# Wikidata Query Service Utilities #
####################################

TYPE_TO_ID_MAPPING = bidict.bidict({
    'Organization': 'Q43229',
    'Anthroponym': 'Q10856962',
    'Work': 'Q386724',
    'Natural Geographic Entity': 'Q27096220',
})

QUERY_TEMPLATE_FOR_ENTITY_COMMONLY_KNOWN_ISAS = '''
SELECT ?VALID_GENLS ?TERM
WHERE 
{{
  VALUES ?TERM {{ {space_separated_term_ids} }}.
  ?TERM wdt:P31 ?IMMEDIATE_GENLS.
  ?IMMEDIATE_GENLS 	wdt:P279* ?VALID_GENLS.
  VALUES ?VALID_GENLS {{ '''+' '.join(map(lambda type_string: 'wd:'+type_string, TYPE_TO_ID_MAPPING.values()))+''' }}.
  MINUS {{
    ?TERM wdt:P31 wd:Q4167410 .
  }}
}}
'''

WIKI_DATA_QUERY_SERVICE_URI = 'https://query.wikidata.org'

async def _find_commonly_known_isas_via_web_scraper(term_ids_without_item_prefix:str) -> Set[Tuple[str, str]]:
    term_type_id_pairs = set()
    if len(term_ids_without_item_prefix) != 0:
        term_ids = map(lambda raw_term_id: 'wd:'+raw_term_id, term_ids_without_item_prefix)
        space_separated_term_ids = ' '.join(term_ids)
        sparql_query = QUERY_TEMPLATE_FOR_ENTITY_COMMONLY_KNOWN_ISAS.format(space_separated_term_ids=space_separated_term_ids)
        sparql_query_encoded = urllib.parse.quote(sparql_query)
        number_of_variables_queried = 2
        uri = WIKI_DATA_QUERY_SERVICE_URI+'/#'+sparql_query_encoded
        browser = await pyppeteer.launch({'headless': True})
        page = await browser.newPage()
        try:
            await page.goto(uri)
            selector_query_for_arbitrary_text_inside_query_box = 'span.cm-variable-2'
            await page.waitForSelector(selector_query_for_arbitrary_text_inside_query_box)
            button = await page.querySelector('button#execute-button')
            await page.evaluate('(button) => button.click()', button)
            await page.waitForSelector('table.table.table-bordered.table-hover')
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
                    term_type_id_pairs.add((current_term, current_type))
                    current_term = None
                    current_type = None
        except pyppeteer.errors.NetworkError:
            pass
        finally:
            await browser.close()
    return term_type_id_pairs

def find_commonly_known_isas(term_ids: List[str]) -> Set[Tuple[str, str]]:
    print()
    print("find_commonly_known_isas")
    print("term_ids {}".format(term_ids))
    task = _find_commonly_known_isas_via_web_scraper(term_ids)
    result = _execute_async_task(task)
    return result

###########################
# Most Abstract Interface #
###########################

@lru_cache(maxsize=32768)
def string_corresponding_wikidata_term_type_pairs(input_string: str) -> Set[Tuple[str, str]]:
    term_ids = string_corresponding_commonly_known_entities(input_string)
    term_type_id_pairs = find_commonly_known_isas(term_ids)
    term_type_pairs = [(term, TYPE_TO_ID_MAPPING.inverse[type_id]) for term, type_id in term_type_id_pairs]
    return term_type_pairs

def main():
    print("This module contains utilities for named entity recognition via Wikidata scraping.")

if __name__ == '__main__':
    main()
