#!/usr/bin/python3

"""
"""

###########
# Imports #
###########

import asyncio
import pyppeteer

###########
# Globals #
###########

UNIQUE_BOGUS_RESULT_IDENTIFIER = object()

##########################
# Web Scraping Utilities #
##########################

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

def _indefinitely_attempt_task_until_success(coroutine, coroutine_args):
    result = UNIQUE_BOGUS_RESULT_IDENTIFIER
    while result == UNIQUE_BOGUS_RESULT_IDENTIFIER:
        task = coroutine(*coroutine_args)
        try:
            results = EVENT_LOOP.run_until_complete(asyncio.gather(task))
            if isinstance(results, list) and len(results) == 1:
                result = results[0]
        except Exception as err:
            print(f'Error: {err}')
            pass
        if result == UNIQUE_BOGUS_RESULT_IDENTIFIER:
            print("Attempting to execute {coroutine} on {coroutine_args} failed.".format(coroutine=coroutine, coroutine_args=coroutine_args))
    return result

async def _launch_browser_page():
    browser = await pyppeteer.launch({'headless': False})
    page = await browser.newPage()
    return page

BROWSER_PAGE = _indefinitely_attempt_task_until_success(_launch_browser_page, [])

################
# Blog Scraper #
################

async def _scrape():
    results = []
    uri = "https://paul-tqh-nguyen.github.io/about/"
    try:
        await BROWSER_PAGE.goto(uri)
        await BROWSER_PAGE.waitForSelector("section#services")
        stuff = await BROWSER_PAGE.querySelectorAll('section#services')
        results.append(stuff)
    except pyppeteer.errors.NetworkError:
        pass
    return results

##########
# Driver #
##########

def main():
    result = _indefinitely_attempt_task_until_success(_scrape, [])
    print(f"results {repr(results)}")

if __name__ == '__main__':
    main()
