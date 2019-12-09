#!/usr/bin/python3 -O

"""

Utilities for named entity recognition via Wikidata scraping (to avoid banning).

Owner : paul-tqh-nguyen

Created : 11/19/2019

File Name : named_entity_recognition_via_wikidata.py

File Organization:
* Imports
* Async IO Utilities
* Web Scraping Utilities
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
import time
import warnings
from typing import List, Tuple, Set
import subprocess # @todo Remove after done debugging

LOGGING_FILE = "/home/pnguyen/Desktop/log.txt"

#@profile
def _logging_print(input_string: str) -> None:
    with open(LOGGING_FILE, 'a') as f:
        f.write(input_string+'\n')
    print(input_string)
    return None

#@profile
def _print_all_gnome_shell_processes() -> None:
    ps_e_process = subprocess.Popen("top -b -n 1", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ps_e_stdout_string, _ = ps_e_process.communicate()
    _logging_print("number of chrome processes {}".format(len(list(filter(lambda line: 'chrom' in line, ps_e_stdout_string.decode("utf-8").split('\n'))))))
    list(map(_logging_print, filter(lambda line: 'gnome-shell' in line, ps_e_stdout_string.decode("utf-8").split('\n'))))
    return None

######################
# Async IO Utilities #
######################

UNIQUE_BOGUS_RESULT_IDENTIFIER = (lambda x: x)

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

#@profile
def _indefinitely_attempt_task_until_success(coroutine, coroutine_args):
    result = UNIQUE_BOGUS_RESULT_IDENTIFIER
    while result == UNIQUE_BOGUS_RESULT_IDENTIFIER:
        task = coroutine(*coroutine_args)
        from datetime import datetime; _logging_print("_indefinitely_attempt_task_until_success attempt start time {}".format(datetime.now()))
        _logging_print("All shell processes 1")
        _print_all_gnome_shell_processes()
        try:
            results = EVENT_LOOP.run_until_complete(asyncio.gather(task))
            if isinstance(results, list) and len(results) == 1:
                result = results[0]
        except Exception as err:
            _logging_print("err :: {}".format(err))
            pass
        finally:
            _logging_print("All shell processes 2")
            _print_all_gnome_shell_processes()
            pending_tasks = asyncio.Task.all_tasks()
            _logging_print("len(pending_tasks) {}".format(len(pending_tasks)))
            for pending_task in pending_tasks:
                _logging_print("pending_task {}".format(pending_task))
            _logging_print("All shell processes 3")
            _print_all_gnome_shell_processes()
        if result == UNIQUE_BOGUS_RESULT_IDENTIFIER:
            warnings.warn("Attempting to execute {coroutine} on {coroutine_args} failed.".format(coroutine=coroutine, coroutine_args=coroutine_args))
            time.sleep(1)
    return result

##########################
# Web Scraping Utilities #
##########################

async def _launch_browser_page():
    browser = await pyppeteer.launch({'headless': True})
    page = await browser.newPage()
    return page

BROWSER_PAGE = _indefinitely_attempt_task_until_success(_launch_browser_page, [])

#############################
# Wikidata Search Utilities #
#############################

WIKIDATA_SEARCH_URI_TEMPLATE = 'https://www.wikidata.org/w/index.php?sort=relevance&search={encoded_string}'

#@profile
def _normalize_string_wrt_unicode(input_string: str) -> str:
    normalized_string = unicodedata.normalize('NFKD', input_string).encode('ascii', 'ignore').decode('utf-8')
    return normalized_string

PUNUCTION_REMOVING_TRANSLATION_TABLE = str.maketrans('', '', string.punctuation)

#@profile
def _normalize_string_for_wikidata_entity_label_comparison(input_string: str) -> str:
    normalized_string = input_string
    normalized_string = _normalize_string_wrt_unicode(normalized_string)
    normalized_string = normalized_string.lower()
    normalized_string = normalized_string.translate(PUNUCTION_REMOVING_TRANSLATION_TABLE)
    return normalized_string

#@profile
async def _most_relevant_wikidata_entities_corresponding_to_string(input_string: str) -> str:
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 1")
    _print_all_gnome_shell_processes()
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 1")
    wikidata_entities_corresponding_to_string = []
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 1.1")
    page = BROWSER_PAGE
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 2")
    input_string_encoded = urllib.parse.quote(input_string)
    uri = WIKIDATA_SEARCH_URI_TEMPLATE.format(encoded_string=input_string_encoded)
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 2")
    _print_all_gnome_shell_processes()
    try:
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 3")
        _logging_print("uri {}".format(uri))
        await page.goto(uri)
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 3.5")
        await page.waitForSelector('div#mw-content-text')
        search_results_div = await page.waitForSelector('div.searchresults')
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 4")
        search_results_paragraph_elements = await search_results_div.querySelectorAll('p')
        search_results_have_shown_up = None
        for paragraph_element in search_results_paragraph_elements:
            _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 5")
            paragraph_element_classname_string = await page.evaluate('(p) => p.className', paragraph_element)
            paragraph_element_classnames = paragraph_element_classname_string.split(' ')
            _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 6")
            for paragraph_element_classname in paragraph_element_classnames:
                if paragraph_element_classname == 'mw-search-nonefound':
                    search_results_have_shown_up = False
                elif paragraph_element_classname == 'mw-search-pager-bottom':
                    search_results_have_shown_up = True
                if search_results_have_shown_up is not None:
                    break
            _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 7")
            if search_results_have_shown_up is not None:
                break
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 8")
        if search_results_have_shown_up:
            search_results_divs = await page.querySelectorAll('div.mw-search-result-heading')
            # _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 9")
            for search_results_div in search_results_divs:
                search_results_div_text_content = await page.evaluate('(search_results_div) => search_results_div.textContent', search_results_div)
                parsable_text_match = re.match(r'^.+\(Q[0-9]+\) +$', search_results_div_text_content)
                # _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 10")
                if parsable_text_match:
                    parsable_text = parsable_text_match.group()
                    parsable_text = parsable_text.replace(')','')
                    parsable_text_parts = parsable_text.split('(')
                    # _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 11")
                    if len(parsable_text_parts)==2:
                        (label, term_id) = parsable_text_parts
                        label = label.strip()
                        term_id = term_id.strip()
                        # _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 12")
                        if _normalize_string_for_wikidata_entity_label_comparison(label) == _normalize_string_for_wikidata_entity_label_comparison(input_string):
                            wikidata_entities_corresponding_to_string.append(term_id)
                            if len(wikidata_entities_corresponding_to_string)>5:
                                break
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 13")
    except pyppeteer.errors.NetworkError:
        pass
    finally:
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 3")
        _print_all_gnome_shell_processes()
        # await page.close()
        # await browser.close()
        # _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 4")
        # _print_all_gnome_shell_processes()
        # _logging_print("before communicate browser.process {}".format(browser.process))
        # _, errs = browser.process.communicate()
        # assert errs is None
        # _logging_print("after communicate browser.process {}".format(browser.process))
        # _logging_print("errs {}".format(errs))
        # process_is_still_running = browser.process.poll() is None
        # _logging_print("process_is_still_running {}".format(process_is_still_running))
        # assert not process_is_still_running
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 5")
        _print_all_gnome_shell_processes()
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 14")
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 6")
    _print_all_gnome_shell_processes()
    return wikidata_entities_corresponding_to_string

#@profile
def _string_corresponding_commonly_known_entities(input_string: str) -> List[str]:
    _logging_print("")
    _logging_print("_string_corresponding_commonly_known_entities")
    _logging_print("input_string {}".format(input_string))
    result = _indefinitely_attempt_task_until_success(_most_relevant_wikidata_entities_corresponding_to_string, [input_string])
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

#@profile
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

#@profile
async def _query_wikidata_via_web_scraper(sparql_query:str) -> List[dict]:
    _logging_print("_query_wikidata_via_web_scraper All shell processes 1")
    _print_all_gnome_shell_processes()
    results = []
    sparql_query_encoded = urllib.parse.quote(sparql_query)
    uri = WIKI_DATA_QUERY_SERVICE_URI+'/#'+sparql_query_encoded
    page = BROWSER_PAGE
    sparql_query_queried_variables = _sparql_query_queried_variables(sparql_query)
    number_of_variables_queried = len(sparql_query_queried_variables)
    _logging_print("_query_wikidata_via_web_scraper All shell processes 2")
    _print_all_gnome_shell_processes()
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
        _logging_print("_query_wikidata_via_web_scraper All shell processes 3")
        _print_all_gnome_shell_processes()
        # await page.close()
        # await browser.close()
        # _logging_print("_query_wikidata_via_web_scraper All shell processes 4")
        # _print_all_gnome_shell_processes()
        # _logging_print("before communicate browser.process {}".format(browser.process))
        # _, errs = browser.process.communicate()
        # assert errs is None
        # _logging_print("after communicate browser.process {}".format(browser.process))
        # _logging_print("errs {}".format(errs))
        # process_is_still_running = browser.process.poll() is None
        # _logging_print("process_is_still_running {}".format(process_is_still_running))
        # assert not process_is_still_running
        _logging_print("_query_wikidata_via_web_scraper All shell processes 5")
        _print_all_gnome_shell_processes()
    return results

###########################
# Most Abstract Interface #
###########################

#@profile
def execute_sparql_query_via_wikidata(sparql_query:str) -> List[dict]:
    _logging_print("")
    _logging_print("execute_sparql_query_via_wikidata")
    _logging_print("sparql_query {}".format(sparql_query))
    result = _indefinitely_attempt_task_until_success(_query_wikidata_via_web_scraper, [sparql_query])
    return result

#@profile
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

#@profile
def string_corresponding_wikidata_term_type_pairs(input_string: str) -> Set[Tuple[str, str]]:
    _logging_print("string_corresponding_wikidata_term_type_pairs All shell processes 1")
    _print_all_gnome_shell_processes()
    term_ids = _string_corresponding_commonly_known_entities(input_string)
    _logging_print("")
    _logging_print("string_corresponding_wikidata_term_type_pairs")
    _logging_print("input_string {}".format(input_string))
    _logging_print("term_ids {}".format(term_ids))
    term_type_pairs = _find_commonly_known_isas(term_ids)
    term_type_pairs = [(term, TYPE_TO_ID_MAPPING.inverse[type_id]) for term, type_id in term_type_pairs]
    _logging_print("string_corresponding_wikidata_term_type_pairs All shell processes 2")
    _print_all_gnome_shell_processes()
    return term_type_pairs

#@profile
def main():
    _logging_print("This module contains utilities for named entity recognition via Wikidata scraping.")
    _logging_print("BROWSER {}".format(BROWSER))
    for _ in range(10):
        answer = string_corresponding_wikidata_term_type_pairs("friar")
    _logging_print("answer {}".format(answer))
    _logging_print("success")

if __name__ == '__main__':
    main()
