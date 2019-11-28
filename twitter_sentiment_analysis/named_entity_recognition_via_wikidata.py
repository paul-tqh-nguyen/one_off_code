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

def _string_corresponding_commonly_known_entities(input_string: str) -> List[str]:
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

def _sparql_query_queried_variables(sparql_query:str) -> List[str]:
    queried_variables = []
    sparql_tokens = sparql_query.split()
    assert sparql_tokens[0].lower()=='select'
    for sparql_token in sparql_tokens[1:]:
        if sparql_token[0]=='?':
            queried_variables.append(sparql_token)
        else:
            break
    return queried_variables

async def _query_wikidata_via_web_scraper(sparql_query:str) -> List[dict]:
    results = []
    sparql_query_encoded = urllib.parse.quote(sparql_query)
    uri = WIKI_DATA_QUERY_SERVICE_URI+'/#'+sparql_query_encoded
    browser = await pyppeteer.launch({'headless': True})
    page = await browser.newPage()
    sparql_query_queried_variables = _sparql_query_queried_variables(sparql_query)
    number_of_variables_queried = len(sparql_query_queried_variables)
    try:
        await page.goto(uri)
        selector_query_for_arbitrary_text_inside_query_box = 'span.cm-variable-2'
        await page.waitForSelector(selector_query_for_arbitrary_text_inside_query_box)
        button = await page.querySelector('button#execute-button')
        await page.evaluate('(button) => button.click()', button)
        await page.waitForSelector('div.th-inner.sortable.both')
        column_header_divs = await page.querySelectorAll('div.th-inner.sortable.both')
        assert len(column_header_divs) == number_of_variables_queried
        variable_names = []
        for column_header_div in column_header_divs:
            variable_name = await page.evaluate('(column_header_div) => column_header_div.textContent', column_header_div)
            variable_names.append(variable_name)
        assert sparql_query_queried_variables == list(map(lambda variable_name: '?'+variable_name, variable_names))
        anchors = await page.querySelectorAll('a.item-link')
        result = dict()
        for anchor_index, anchor in enumerate(anchors):
            anchor_variable = variable_names[anchor_index%number_of_variables_queried]
            anchor_link = await page.evaluate('(anchor) => anchor.href', anchor)
            assert len(re.findall(r"^http://www.wikidata.org/entity/\w+$", anchor_link))==1
            entity_id = anchor_link.replace('http://www.wikidata.org/entity/','')
            anchor_variable_with_question_mark_prefix = '?'+anchor_variable
            result[anchor_variable_with_question_mark_prefix] = entity_id
            if (1+anchor_index)%number_of_variables_queried==0:
                assert len(result) == number_of_variables_queried
                results.append(result)
                result = dict()
    except pyppeteer.errors.NetworkError:
        pass
    finally:
        await browser.close()
    return results

###########################
# Most Abstract Interface #
###########################

def execute_sparql_query_via_wikidata(sparql_query:str) -> List[dict]:
    task = _query_wikidata_via_web_scraper(sparql_query)
    result = _execute_async_task(task)
    return result

def _find_commonly_known_isas(term_ids_without_item_prefix: List[str]) -> Set[Tuple[str, str]]:
    term_type_pairs = set()
    if len(term_ids_without_item_prefix) != 0:
        term_ids = map(lambda raw_term_id: 'wd:'+raw_term_id, term_ids_without_item_prefix)
        space_separated_term_ids = ' '.join(term_ids)
        sparql_query = QUERY_TEMPLATE_FOR_ENTITY_COMMONLY_KNOWN_ISAS.format(space_separated_term_ids=space_separated_term_ids)
        results = execute_sparql_query_via_wikidata(sparql_query)
        for result in results:
            term = result['?TERM']
            term_type = result['?VALID_GENLS']
            term_type_pair = (term, term_type)
            term_type_pairs.add(term_type_pair)
    return term_type_pairs

def string_corresponding_wikidata_term_type_pairs(input_string: str) -> Set[Tuple[str, str]]:
    term_ids = _string_corresponding_commonly_known_entities(input_string)
    print("input_string {}".format(input_string)) if term_ids else None
    term_type_pairs = _find_commonly_known_isas(term_ids)
    term_type_pairs = [(term, TYPE_TO_ID_MAPPING.inverse[type_id]) for term, type_id in term_type_pairs]
    return term_type_pairs

def main():
    print("This module contains utilities for named entity recognition via Wikidata scraping.")

if __name__ == '__main__':
    main()
