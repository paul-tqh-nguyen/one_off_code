#!/usr/bin/python3

"""
"""

###########
# Imports #
###########

import asyncio
import pyppeteer
from typing import List, Iterable

from misc_utilities import *

###########
# Globals #
###########

UNIQUE_BOGUS_RESULT_IDENTIFIER = object()

MAX_NUMBER_OF_ASYNCHRONOUS_ATTEMPTS = 10
BROWSER_IS_HEADLESS = True

BLOG_ARCHIVE_URL = "https://www.joelonsoftware.com/archives/"

##########################
# Web Scraping Utilities #
##########################

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

def _attempt_task_until_success(coroutine, coroutine_args: List = [], max_number_of_attempts = MAX_NUMBER_OF_ASYNCHRONOUS_ATTEMPTS):
    result = UNIQUE_BOGUS_RESULT_IDENTIFIER
    for _ in range(max_number_of_attempts):
        task = coroutine(*coroutine_args)
        try:
            results = EVENT_LOOP.run_until_complete(asyncio.gather(task))
            if isinstance(results, list) and len(results) == 1:
                result = results[0]
        except Exception as err:
            print(f'Error: {err}') # @todo change this message
            pass
        if result == UNIQUE_BOGUS_RESULT_IDENTIFIER:
            print("Attempting to execute {coroutine} on {coroutine_args} failed.".format(coroutine=coroutine, coroutine_args=coroutine_args))
        if result != UNIQUE_BOGUS_RESULT_IDENTIFIER:
            break
    return result

async def _launch_browser_page():
    browser = await pyppeteer.launch({'headless': BROWSER_IS_HEADLESS})
    page = await browser.newPage()
    return page

BROWSER_PAGE = _attempt_task_until_success(_launch_browser_page, [])

################
# Blog Scraper #
################

# Month Links

async def _gather_month_links() -> List[str]:
    month_links: List[str] = []
    await BROWSER_PAGE.goto(BLOG_ARCHIVE_URL)
    await BROWSER_PAGE.waitForSelector("div.site-info")
    month_lis = await BROWSER_PAGE.querySelectorAll('li.month')
    for month_li in month_lis:
        anchors = await month_li.querySelectorAll('a')
        anchor = only_one(anchors)
        link = await BROWSER_PAGE.evaluate('(anchor) => anchor.href', anchor)
        month_links.append(link)
    return month_links

def gather_month_links() -> List[str]:
    month_links = _attempt_task_until_success(_gather_month_links, [])
    return month_links

# Blog Links from Month Links

async def _blog_links_from_month_link(month_link: str) -> List[str]:
    blog_links: List[str] = []
    await BROWSER_PAGE.goto(month_link)
    await BROWSER_PAGE.waitForSelector("div.site-info")
    blog_h1s = await BROWSER_PAGE.querySelectorAll('h1.entry-title')
    for blog_h1 in blog_h1s:
        anchors = await blog_h1.querySelectorAll('a')
        anchor = only_one(anchors)
        link = await BROWSER_PAGE.evaluate('(anchor) => anchor.href', anchor)
        blog_links.append(link)
    return blog_links

def blog_links_from_month_link(month_link: str) -> Iterable[str]:
    return _attempt_task_until_success(_blog_links_from_month_link, [month_link])

def blog_links_from_month_links(month_links: Iterable[str]) -> Iterable[str]:
    return itertools.chain(map(blog_links_from_month_link, month_links))

# Data from Blog Links

async def _data_dict_from_blog_link(blog_link: str) -> dict:
    data_dict = {'blog_link': blog_link}
    await BROWSER_PAGE.goto(blog_link)
    await BROWSER_PAGE.waitForSelector("div.site-info")
    articles = await BROWSER_PAGE.querySelectorAll('article.post')
    article = only_one(articles)
    
    entry_title_h1s = await article.querySelectorAll('h1.entry-title')
    entry_title_h1 = only_one(entry_title_h1s)
    title = await BROWSER_PAGE.evaluate('(element) => element.textContent', entry_title_h1)
    data_dict['title'] = title
    
    entry_date_div = await article.querySelectorAll('div.entry-date')
    
    posted_on_spans = await entry_date_div.querySelectorAll('span.posted-on')
    posted_on_span = only_one(posted_on_spans)
    date = await BROWSER_PAGE.evaluate('(element) => element.textContent', posted_on_span)
    data_dict['date'] = date
    
    author_spans = await entry_date_div.querySelectorAll('span.author')
    author_span = only_one(author_spans)
    author = await BROWSER_PAGE.evaluate('(element) => element.textContent', author_span)
    data_dict['author'] = author
    
    entry_meta_divs = await article.querySelectorAll('div.entry-meta')
    entry_meta_div = only_one(entry_meta_divs)
    entry_meta_div_uls = await entry_meta_div.querySelectorAll('ul.meta-list')
    entry_meta_div_ul = only_one(entry_meta_div_uls)
    entry_meta_div_ul_lis = await entry_meta_div_ul.querySelectorAll('li.meta-cat')
    blog_tags: List[str] = []
    for li in entry_meta_div_ul_lis:
        blog_tag = await BROWSER_PAGE.evaluate('(element) => element.textContent', li)
        blog_tags.append(blog_tag)
    data_dict['blog_tags'] = blog_tags
    
    entry_content_divs = await article.querySelectorAll('div.entry-content')
    entry_content_div = only_one(entry_content_divs)
    blog_text = await BROWSER_PAGE.evaluate('(element) => element.textContent', entry_content_div)
    data_dict['blog_text'] = blog_text
    
    return data_dict

def data_dict_from_blog_link(blog_link: str) -> Iterable[dict]:
    return _attempt_task_until_success(_data_dict_from_blog_link, [month_link])

def data_dicts_from_blog_links(blog_links: Iterable[str]) -> Iterable[dict]:
    return itertools.chain(map(data_dict_from_blog_link, blog_links))

##########
# Driver #
##########

def main():
    month_links = gather_month_links()
    blog_links = blog_links_from_month_links(month_links)
    rows = data_dicts_from_blog_links(blog_links)
    results = rows
    return results

if __name__ == '__main__':
    main()
