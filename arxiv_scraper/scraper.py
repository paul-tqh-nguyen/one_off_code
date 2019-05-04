#!/usr/bin/python

"""

arXiv Scraping Utilities

Owner : paul-tqh-nguyen

Created : 05/03/2019

File Organization:
* Misc Utilities
* arXiv Scraping Utilities

"""

# @todo update access privileges
# @todo make sure everything is used
# @todo write tests

# Standard Library Imports
import time
from functools import lru_cache
import re

#Third Party Imports
from bs4 import BeautifulSoup
import requests

##################
# Misc Utilities #
##################

# @todo move these somewhere more general

def p1(iterable):
    for element in iterable:
        print(element)

def timer(function): # @todo make this only do something when we're in debug mode
    def timed(*args, **kw):
        start_time = time.time()
        answer = function(*args, **kw)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # @todo make this prettier and look like what the code would look like
        print("func:{function_name} args:[{args}, {keywords}] took: {elapsed_time:.6f} seconds.".format(function_name=function.__name__, args=args, keywords=kw, elapsed_time=elapsed_time)) 
        return answer
    return timed

def execfile(file_location):
    '''Simulates EXECFILE from Python2'''
    return exec(open(file_location).read())

############################
# arXiv Scraping Utilities #
############################

ARXIV_URL = "https://arxiv.org/"

@lru_cache(maxsize=1)
def arxiv_main_page_text():
    get_response = requests.get(ARXIV_URL)
    arxiv_main_page_text = get_response.text
    return arxiv_main_page_text

def arxiv_recent_page_title_and_page_link_string_iterator():
    text = arxiv_main_page_text()
    soup = BeautifulSoup(text, features="lxml")
    anchor_links = soup.find_all("a")
    arxiv_recent_page_relevant_anchor_link_iterator = filter(anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text, anchor_links)
    arxiv_recent_page_title_and_page_link_string_iterator = map(extract_text_and_link_string_from_anchor_link, arxiv_recent_page_relevant_anchor_link_iterator)
    return arxiv_recent_page_title_and_page_link_string_iterator

def extract_text_and_link_string_from_anchor_link(anchor_link):
    link_text = anchor_link.text
    link_string = anchor_link.get("href")
    return (link_text, link_string)

def anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text(anchor_link):
    href_attribute = anchor_link.get("href")
    link_text = anchor_link.text
    anchor_link_is_arxiv_recent_page_link = is_arxiv_recent_page_link(href_attribute)
    anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text = None
    if anchor_link_is_arxiv_recent_page_link:
        anchor_link_has_research_field_denoting_text = not (link_text == "recent") # @todo expand as needed
        anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text = anchor_link_has_research_field_denoting_text
    else:
        anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text = False
    return anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text

arxiv_recent_page_link_reg_ex = re.compile("/list/.+/recent")

def is_arxiv_recent_page_link(link_string):
    string_pattern_match_result = arxiv_recent_page_link_reg_ex.match(link_string)
    is_arxiv_recent_page_link = (string_pattern_match_result is not None)
    return is_arxiv_recent_page_link
