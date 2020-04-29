#!/usr/bin/python3

"""
"""

# @todo fill in doc string
# @todo get rid of @trace decorators

###########
# Imports #
###########

import os
import asyncio
import pyppeteer
import itertools
import traceback
import time
import warnings
import pandas as pd
from typing import List, Tuple, Iterable, Callable, Awaitable

from misc_utilities import *

###########
# Globals #
###########

UNIQUE_BOGUS_RESULT_IDENTIFIER = object()

MAX_NUMBER_OF_BROWSER_CLOSE_ATTEMPTS = 1000
MAX_NUMBER_OF_BROWSER_LAUNCH_ATTEMPTS = 1000
NUMBER_OF_ATTEMPTS_PER_SLEEP = 10
BROWSER_IS_HEADLESS = False

BLOG_ARCHIVE_URL = "https://www.joelonsoftware.com/archives/"

OUTPUT_CSV_FILE = './output.csv'

##########################
# Web Scraping Utilities #
##########################

def _sleeping_range(upper_bound: int):
    for attempt_index in range(NUMBER_OF_ATTEMPTS_PER_SLEEP):
        if attempt_index // 10 and attempt_index % 10 == 0:
            time.sleep(60*(i//10))
        yield attempt_index

def assert_no_chrom(): # @todo get rid of this
    import subprocess
    try:
        pgrep_process = subprocess.Popen("pgrep chrom", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        chrom_processes_string, _ = pgrep_process.communicate()
        chrom_processes_string = chrom_processes_string.decode('utf-8')
        chrom_processes = eager_map(int,chrom_processes_string.split())
        pgrep_process.terminate()
        assert len(chrom_processes) == 0
    except subprocess.CalledProcessError as err:
        assert err.args[0] == 1

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

async def _launch_browser_page() -> Tuple[pyppeteer.browser.Browser, pyppeteer.page.Page]:
    browser: pyppeteer.browser.Browser = await pyppeteer.launch({
        'headless': BROWSER_IS_HEADLESS,
    })
    page: pyppeteer.page.Page = await browser.newPage()
    return browser, page

async def _possibly_close_browser_and_page(browser: pyppeteer.browser.Browser, page: pyppeteer.page.Page) -> None:
    for _ in _sleeping_range(MAX_NUMBER_OF_BROWSER_CLOSE_ATTEMPTS):
        try:
            if isinstance(page, pyppeteer.page.Page):
                if not page.isClosed():
                    await page.close()
        except Exception as err:
            warnings.warn(f'{time.strftime("%m/%d/%Y_%H:%M:%S")} {_possibly_close_browser_and_page} {err}')
        try:
            if isinstance(browser, pyppeteer.browser.Browser):
                await browser.close()
        except Exception as err:
            warnings.warn(f'{time.strftime("%m/%d/%Y_%H:%M:%S")} {_possibly_close_browser_and_page} {err}')
        break
    return

def scrape_function(func: Awaitable) -> Awaitable:
    async def decorating_function(*args, **kwargs):
        updated_kwargs = kwargs.copy()
        browser, page = None, None
        result = UNIQUE_BOGUS_RESULT_IDENTIFIER
        for _ in _sleeping_range(MAX_NUMBER_OF_BROWSER_LAUNCH_ATTEMPTS):
            try:
                browser, page = await _launch_browser_page()
                updated_kwargs['page'] = page
                result = await func(*args, **updated_kwargs)
            except (pyppeteer.errors.BrowserError,
                    pyppeteer.errors.ElementHandleError,
                    pyppeteer.errors.NetworkError,
                    pyppeteer.errors.PageError,
                    pyppeteer.errors.PyppeteerError,
                    pyppeteer.errors.TimeoutError) as err:
                warnings.warn(f'{time.strftime("%m/%d/%Y_%H:%M:%S")} {func} {err}')
            except Exception as err:
                raise
            finally:
                await _possibly_close_browser_and_page(browser, page)
            if result != UNIQUE_BOGUS_RESULT_IDENTIFIER:
                break
        assert_no_chrom()
        assert browser is not None
        assert page is not None
        return result
    return decorating_function

################
# Blog Scraper #
################

# Month Links

@scrape_function
async def _gather_month_links(*, page) -> List[str]:
    month_links: List[str] = []
    await page.goto(BLOG_ARCHIVE_URL)
    await page.waitForSelector("div.site-info")
    month_lis = await page.querySelectorAll('li.month')
    for month_li in month_lis:
        anchors = await month_li.querySelectorAll('a')
        anchor = only_one(anchors)
        link = await page.evaluate('(anchor) => anchor.href', anchor)
        month_links.append(link)
    return month_links

@trace
def gather_month_links() -> List[str]:
    month_links = EVENT_LOOP.run_until_complete(_gather_month_links())
    return month_links

# Blog Links from Month Links

@scrape_function
async def _blog_links_from_month_link(month_link: str, *, page: pyppeteer.page.Page) -> List[str]:
    blog_links: List[str] = []
    await page.goto(month_link)
    await page.waitForSelector("div.site-info")
    blog_h1s = await page.querySelectorAll('h1.entry-title')
    for blog_h1 in blog_h1s:
        anchors = await blog_h1.querySelectorAll('a')
        anchor = only_one(anchors)
        link = await page.evaluate('(anchor) => anchor.href', anchor)
        blog_links.append(link)
    return blog_links

@trace
def blog_links_from_month_link(month_link: str) -> Iterable[str]:
    return EVENT_LOOP.run_until_complete(_blog_links_from_month_link(month_link))

@trace
def blog_links_from_month_links(month_links: Iterable[str]) -> Iterable[str]:
    return itertools.chain(*eager_map(blog_links_from_month_link, month_links))

# Data from Blog Links

@scrape_function
async def _data_dict_from_blog_link(blog_link: str, *, page: pyppeteer.page.Page) -> dict:
    print("_data_dict_from_blog_link 0.1")
    data_dict = {'blog_link': blog_link}
    print("_data_dict_from_blog_link 0.2")
    await page.goto(blog_link)
    print("_data_dict_from_blog_link 0.3")
    await page.waitForSelector("div.site-info")
    print("_data_dict_from_blog_link 0.4")
    articles = await page.querySelectorAll('article.post')
    print("_data_dict_from_blog_link 0.5")
    article = only_one(articles)
    
    print("_data_dict_from_blog_link 1")
    
    entry_title_h1s = await article.querySelectorAll('h1.entry-title')
    entry_title_h1 = only_one(entry_title_h1s)
    title = await page.evaluate('(element) => element.textContent', entry_title_h1)
    data_dict['title'] = title
    
    print("_data_dict_from_blog_link 2")
    
    entry_date_divs = await article.querySelectorAll('div.entry-date')
    entry_date_div = only_one(entry_date_divs)
    
    print("_data_dict_from_blog_link 3")
    
    posted_on_spans = await entry_date_div.querySelectorAll('span.posted-on')
    posted_on_span = only_one(posted_on_spans)
    date = await page.evaluate('(element) => element.textContent', posted_on_span)
    data_dict['date'] = date
    
    print("_data_dict_from_blog_link 4")
    
    author_spans = await entry_date_div.querySelectorAll('span.author')
    author_span = only_one(author_spans)
    author = await page.evaluate('(element) => element.textContent', author_span)
    data_dict['author'] = author
    
    print("_data_dict_from_blog_link 5")
    
    entry_meta_divs = await article.querySelectorAll('div.entry-meta')
    entry_meta_div = only_one(entry_meta_divs)
    entry_meta_div_uls = await entry_meta_div.querySelectorAll('ul.meta-list')
    entry_meta_div_ul = only_one(entry_meta_div_uls)
    entry_meta_div_ul_lis = await entry_meta_div_ul.querySelectorAll('li.meta-cat')
    entry_meta_div_ul_li = only_one(entry_meta_div_ul_lis)
    blog_tags_text = await page.evaluate('(element) => element.textContent', entry_meta_div_ul_li)
    data_dict['blog_tags'] = blog_tags_text
    
    print("_data_dict_from_blog_link 6")
    
    entry_content_divs = await article.querySelectorAll('div.entry-content')
    entry_content_div = only_one(entry_content_divs)
    blog_text = await page.evaluate('(element) => element.textContent', entry_content_div)
    data_dict['blog_text'] = blog_text
    
    print("_data_dict_from_blog_link 7")
    
    return data_dict

@trace
def data_dict_from_blog_link(blog_link: str) -> Iterable[dict]:
    return EVENT_LOOP.run_until_complete(_data_dict_from_blog_link(blog_link))

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
