#!/usr/bin/python3

"""
"""

# @todo fill in doc string
# @todo get rid of @trace decorators

###########
# Imports #
###########

import asyncio
import pyppeteer
import itertools
import traceback
import time
import warnings
import pandas as pd
from typing import List, Iterable, Callable

from misc_utilities import *

###########
# Globals #
###########

UNIQUE_BOGUS_RESULT_IDENTIFIER = object()

MAX_NUMBER_OF_ASYNCHRONOUS_ATTEMPTS = 1000
BROWSER_IS_HEADLESS = True

BLOG_ARCHIVE_URL = "https://www.joelonsoftware.com/archives/"

OUTPUT_CSV_FILE = './output.csv'

##########################
# Web Scraping Utilities #
##########################

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

async def _launch_browser():
    browser = await pyppeteer.launch({'headless': BROWSER_IS_HEADLESS})
    return browser

BROWSER = EVENT_LOOP.run_until_complete(_launch_browser, [])

def _attempt_task_until_success(coroutine, coroutine_args: List = [], max_number_of_attempts = MAX_NUMBER_OF_ASYNCHRONOUS_ATTEMPTS):
    result = UNIQUE_BOGUS_RESULT_IDENTIFIER
    for _ in range(max_number_of_attempts):
        task = coroutine(*coroutine_args)
        try:
            future = asyncio.gather(task)
            results = EVENT_LOOP.run_until_complete(future)
            if isinstance(results, list) and len(results) == 1:
                result = results[0]
                break
        except Exception as err:
            warnings.warn(f'{time.strftime("%Y/%m/%d_%H:%M:%S")} {err}')
            traceback.print_exc()
            pass
    if result == UNIQUE_BOGUS_RESULT_IDENTIFIER:
        warnings.warn(f'{time.strftime("%Y/%m/%d_%H:%M:%S")} Attempting to execute {coroutine} on {coroutine_args} failed.')
    return result

################
# Blog Scraper #
################

# Month Links

async def _gather_month_links() -> List[str]:
    month_links: List[str] = []
    await BROWSER.goto(BLOG_ARCHIVE_URL)
    await BROWSER.waitForSelector("div.site-info")
    month_lis = await BROWSER.querySelectorAll('li.month')
    for month_li in month_lis:
        anchors = await month_li.querySelectorAll('a')
        anchor = only_one(anchors)
        link = await BROWSER.evaluate('(anchor) => anchor.href', anchor)
        month_links.append(link)
    return month_links

@trace
def gather_month_links() -> List[str]:
    month_links = _attempt_task_until_success(_gather_month_links, [])
    return month_links

# Blog Links from Month Links

async def _blog_links_from_month_link(month_link: str) -> List[str]:
    blog_links: List[str] = []
    await BROWSER.goto(month_link)
    await BROWSER.waitForSelector("div.site-info")
    blog_h1s = await BROWSER.querySelectorAll('h1.entry-title')
    for blog_h1 in blog_h1s:
        anchors = await blog_h1.querySelectorAll('a')
        anchor = only_one(anchors)
        link = await BROWSER.evaluate('(anchor) => anchor.href', anchor)
        blog_links.append(link)
    return blog_links

@trace
def blog_links_from_month_link(month_link: str) -> Iterable[str]:
    return _attempt_task_until_success(_blog_links_from_month_link, [month_link])

@trace
def blog_links_from_month_links(month_links: Iterable[str]) -> Iterable[str]:
    return itertools.chain(*eager_map(blog_links_from_month_link, month_links))

# Data from Blog Links

async def _data_dict_from_blog_link(blog_link: str) -> dict:
    print("_data_dict_from_blog_link 0.1")
    data_dict = {'blog_link': blog_link}
    print("_data_dict_from_blog_link 0.2")
    await BROWSER.goto(blog_link)
    print("_data_dict_from_blog_link 0.3")
    await BROWSER.waitForSelector("div.site-info")
    print("_data_dict_from_blog_link 0.4")
    articles = await BROWSER.querySelectorAll('article.post')
    print("_data_dict_from_blog_link 0.5")
    article = only_one(articles)
    
    print("_data_dict_from_blog_link 1")
    
    entry_title_h1s = await article.querySelectorAll('h1.entry-title')
    entry_title_h1 = only_one(entry_title_h1s)
    title = await BROWSER.evaluate('(element) => element.textContent', entry_title_h1)
    data_dict['title'] = title
    
    print("_data_dict_from_blog_link 2")
    
    entry_date_divs = await article.querySelectorAll('div.entry-date')
    entry_date_div = only_one(entry_date_divs)
    
    print("_data_dict_from_blog_link 3")
    
    posted_on_spans = await entry_date_div.querySelectorAll('span.posted-on')
    posted_on_span = only_one(posted_on_spans)
    date = await BROWSER.evaluate('(element) => element.textContent', posted_on_span)
    data_dict['date'] = date
    
    print("_data_dict_from_blog_link 4")
    
    author_spans = await entry_date_div.querySelectorAll('span.author')
    author_span = only_one(author_spans)
    author = await BROWSER.evaluate('(element) => element.textContent', author_span)
    data_dict['author'] = author
    
    print("_data_dict_from_blog_link 5")
    
    entry_meta_divs = await article.querySelectorAll('div.entry-meta')
    entry_meta_div = only_one(entry_meta_divs)
    entry_meta_div_uls = await entry_meta_div.querySelectorAll('ul.meta-list')
    entry_meta_div_ul = only_one(entry_meta_div_uls)
    entry_meta_div_ul_lis = await entry_meta_div_ul.querySelectorAll('li.meta-cat')
    entry_meta_div_ul_li = only_one(entry_meta_div_ul_lis)
    blog_tags_text = await BROWSER.evaluate('(element) => element.textContent', entry_meta_div_ul_li)
    data_dict['blog_tags'] = blog_tags_text
    
    print("_data_dict_from_blog_link 6")
    
    entry_content_divs = await article.querySelectorAll('div.entry-content')
    entry_content_div = only_one(entry_content_divs)
    blog_text = await BROWSER.evaluate('(element) => element.textContent', entry_content_div)
    data_dict['blog_text'] = blog_text
    
    print("_data_dict_from_blog_link 7")
    
    return data_dict

@trace
def data_dict_from_blog_link(blog_link: str) -> Iterable[dict]:
    return _attempt_task_until_success(_data_dict_from_blog_link, [blog_link])

@trace
def data_dicts_from_blog_links(blog_links: Iterable[str]) -> Iterable[dict]:
    return eager_map(data_dict_from_blog_link, blog_links)

##########
# Driver #
##########

@trace
def gather_data():
    with timer("Data gathering"):
        month_links = gather_month_links()
        blog_links = blog_links_from_month_links(month_links)
        rows = data_dicts_from_blog_links(blog_links)
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV_FILE, index=False)
    return

if __name__ == '__main__':
    gather_data()
