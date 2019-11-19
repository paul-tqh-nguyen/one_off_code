#!/usr/bin/python3 -O

"""

Python wrappers for named entity recognition via Wikidata API.

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

#############
# Utilities #
#############

CHROME_DRIVER = "./selenium_drivers/geckodriver"

'''
SELECT ?VALID_GENLS
WHERE 
{
  wd:Q701755 wdt:P31 ?IMMEDIATE_GENLS.
  ?IMMEDIATE_GENLS 	wdt:P279* ?VALID_GENLS.
  VALUES ?VALID_GENLS { wd:Q43229 }.
}
'''

QUERY_SERVICE_URL_TEMPLATE = "https://query.wikidata.org/bigdata/namespace/wdq/sparql?query={sparql}"

def query_sparql(sparql_string: str):
    encoded_sparql_query = urllib.parse.quote(sparql_string)
    url_with_params = QUERY_SERVICE_URL_TEMPLATE.format(sparql=encoded_sparql_query)
    url_with_params
    return 

SEARCH_API_ENDPOINT = "https://www.wikidata.org/w/api.php"

def most_popular_commonly_known_wikidata_entry_corresponding_to_string(query_string: str):
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': query_string
    }
    request = requests.get(SEARCH_API_ENDPOINT, params=params)
    results = request.json()['search']
    results = filter(lambda result: result['label'].lower()==query_string.lower(), results)
    results = list(results)
    return results

async def paul():
    entity_ids = []
    browser = await pyppeteer.launch({'headless': True})
    page = await browser.newPage()
    uri = 'https://query.wikidata.org/#'+'SELECT%20%3FVALID_GENLS%0AWHERE%20%0A%7B%0A%20%20wd%3AQ701755%20wdt%3AP31%20%3FIMMEDIATE_GENLS.%0A%20%20%3FIMMEDIATE_GENLS%20%09wdt%3AP279%2a%20%3FVALID_GENLS.%0A%20%20VALUES%20%3FVALID_GENLS%20%7B%20wd%3AQ43229%20%7D.%0A%7D%0A'
    await page.goto(uri)
    selector_query_for_arbitrary_text_inside_query_box = 'span.cm-variable-2'
    await page.waitForSelector(selector_query_for_arbitrary_text_inside_query_box)
    button = await page.querySelector('button#execute-button')
    await page.evaluate('(button) => button.click()', button)
    await page.waitForSelector('a.item-link')
    anchors = await page.querySelectorAll('a.item-link')
    for anchor in anchors:
        anchor_link = await page.evaluate('(anchor) => anchor.href', anchor)
        assert len(re.findall(r"^http://www.wikidata.org/entity/\w+$", anchor_link))==1
        entity_id = anchor_link.replace('http://www.wikidata.org/entity/','')
        entity_ids.append(entity_id)
    await browser.close()
    return entity_ids

event_loop = asyncio.get_event_loop()
results = event_loop.run_until_complete(asyncio.gather(paul()))
print(results)
