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
import re
from typing import List

#############
# Utilities #
#############

# use this url instead of the API https://www.wikidata.org/w/index.php?sort=relevance&search=edeka

ORGANIZATION_ID = 'Q43229'
QUERY_TEMPLATE_FOR_ENTITY_RELEVANT_ISAS = '''
SELECT ?VALID_GENLS
WHERE 
{{
  {term_id} wdt:P31 ?IMMEDIATE_GENLS.
  ?IMMEDIATE_GENLS 	wdt:P279* ?VALID_GENLS.
  VALUES ?VALID_GENLS {{ '''+'wd:'+ORGANIZATION_ID+''' }}.
}}
'''

async def _find_relevant_isas_via_web_scraper(term_id_without_item_prefix:str) -> List[str]:
    term_id = 'wd:'+term_id_without_item_prefix
    sparql_query = QUERY_TEMPLATE_FOR_ENTITY_RELEVANT_ISAS.format(term_id=term_id)
    sparql_query_encoded = urllib.parse.quote(sparql_query)
    uri = 'https://query.wikidata.org/#'+sparql_query_encoded
    entity_ids = []
    browser = await pyppeteer.launch({'headless': True})
    page = await browser.newPage()
    try:
        await page.goto(uri)
        selector_query_for_arbitrary_text_inside_query_box = 'span.cm-variable-2'
        await page.waitForSelector(selector_query_for_arbitrary_text_inside_query_box)
        button = await page.querySelector('button#execute-button')
        await page.evaluate('(button) => button.click()', button)
        await page.waitForSelector('a.item-link') # @todo recursively go down the dom instead of doing a broad search from the top
        anchors = await page.querySelectorAll('a.item-link')
        for anchor in anchors:
            anchor_link = await page.evaluate('(anchor) => anchor.href', anchor)
            assert len(re.findall(r"^http://www.wikidata.org/entity/\w+$", anchor_link))==1
            entity_id = anchor_link.replace('http://www.wikidata.org/entity/','')
            entity_ids.append(entity_id)
    except pyppeteer.errors.NetworkError:
        pass
    finally:
        await browser.close()
    return entity_ids

def find_relevant_isas(term_id: str) -> List[str]:
    task = _find_relevant_isas_via_web_scraper(term_id)
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    results = event_loop.run_until_complete(asyncio.gather(task))
    event_loop.close()
    assert len(results) == 1
    result = results[0]
    return result

def main():
    print("This module contains utilities for named entity recognition via Wikidata scraping.")

if __name__ == '__main__':
    main()
