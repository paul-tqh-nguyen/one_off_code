#!/usr/bin/python

"""

arXiv Scraping Utilities

Owner : paul-tqhnguyen

Created : 05/03/2019

File Organization:
* Misc Utilities
* arXiv Scraping Utilities

"""

from bs4 import BeautifulSoup
import requests

##################
# Misc Utilities #
##################

# @todo move these somewhere more general

def cached_function(func): 
    cache = dict()
    def func_cached(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    def cache_clearing_method():
        return cache.clear()
    return func_cached, cache_clearing_method

def execfile(file_location):
    '''Simulates EXECFILE from Python2'''
    return exec(open(file_location).read())

############################
# arXiv Scraping Utilities #
############################

ARXIV_URL = "https://arxiv.org/"

def arxiv_main_page_text_internal():
    get_response = requests.get(ARXIV_URL)
    arxiv_main_page_text = get_response.text
    return arxiv_main_page_text
arxiv_main_page_text, clear_arxiv_main_page_text_cache = cached_function(arxiv_main_page_text_internal) # @todo at some point we need to clear this cache

def print_arxiv_recent_pages_internal():
    text = arxiv_main_page_text()
    soup = BeautifulSoup(text)
    anchor_links = soup.find_all("a")
    return False
