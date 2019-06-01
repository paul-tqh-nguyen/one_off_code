#!/usr/bin/python

"""

arXiv Scraping Utilities

Owner : paul-tqh-nguyen

Created : 05/03/2019

File Organization:
* Misc Utilities
* arXiv Scraping Utilities
** Top Level arXiv Scraping Utilities
** "Recent" Pages arXiv Scraping Utilities
* Main Runner

"""

# @todo update access privileges
# @todo make sure everything is used

# Imports
import time
import functools
from functools import lru_cache
import re
import urllib.parse
import itertools
from warnings import warn
from bs4 import BeautifulSoup
import requests

# Debugging Imports

import pdb
import copy

##################
# Misc Utilities #
##################

# @todo move these somewhere more general

def p1(iterable, number_of_newlines=1):
    number_of_necessary_additional_newlines = (number_of_newlines - 1)
    for element in iterable:
        element_string = str(element)
        additional_new_lines = ("\n"*number_of_necessary_additional_newlines)
        print(element_string+additional_new_lines)

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

@lru_cache(maxsize=256)
def get_text_at_url(url):
    get_response = requests.get(url)
    text = get_response.text
    return text

def arxiv_main_page_text():
    arxiv_main_page_text = get_text_at_url(ARXIV_URL)
    return arxiv_main_page_text

######################################
# Top Level arXiv Scraping Utilities #
######################################

def arxiv_recent_page_title_and_page_link_string_iterator():
    # @todo add comment with the types of values it returns and from what links it gets
    text = arxiv_main_page_text()
    soup = BeautifulSoup(text, features="lxml")
    anchor_links = soup.find_all("a")
    arxiv_recent_page_relevant_anchor_link_iterator = filter(anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text, anchor_links)
    arxiv_recent_page_title_and_page_link_string_iterator = map(extract_text_and_link_string_from_arxiv_anchor_link, arxiv_recent_page_relevant_anchor_link_iterator)
    return arxiv_recent_page_title_and_page_link_string_iterator

def concatenate_relative_link_to_arxiv_base_url(relative_link):
    return urllib.parse.urljoin(ARXIV_URL, relative_link)

def extract_text_and_link_string_from_arxiv_anchor_link(anchor_link):
    link_text = anchor_link.text
    relative_link_string = anchor_link.get("href")
    absolute_relative_link_string = concatenate_relative_link_to_arxiv_base_url(relative_link_string)
    return (link_text, absolute_relative_link_string)

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
    assert anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text is not None, "{function} logic is flawed.".format(function=anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text)
    return anchor_link_is_arxiv_recent_page_link_with_research_field_denoting_text

arxiv_recent_page_link_reg_ex = re.compile("/list/.+/recent")

def is_arxiv_recent_page_link(link_string):
    string_pattern_match_result = arxiv_recent_page_link_reg_ex.match(link_string)
    is_arxiv_recent_page_link = (string_pattern_match_result is not None)
    return is_arxiv_recent_page_link

###########################################
# "Recent" Pages arXiv Scraping Utilities #
###########################################

def extract_info_from_recent_page_url(recent_page_url): # @todo consider returning JSON
    '''The "Recent" page includes a bunch of papers. We return an iterator yielding tuples. The tuples are of the form (LINK_TO_PAPER_PAGE, TITLE, AUTHOR_TO_AUTHOR_LINK_DOUBLES, PRIMARY_SUBJECT, SECONDARY_SUBJECTS).'''
    text = get_text_at_url(recent_page_url)
    soup = BeautifulSoup(text, features="lxml")
    definition_lists = soup.find_all('dl')
    info_tuple_iterators = map(extract_info_tuple_iterator_from_recent_pages_definition_list, definition_lists)
    result_iterator = functools.reduce(itertools.chain, info_tuple_iterators)
    return result_iterator

def extract_info_tuple_iterator_from_recent_pages_definition_list(definition_list):
    info_tuple_iterator = None
    definition_terms = definition_list.find_all("dt")
    definition_descriptions = definition_list.find_all("dd")
    if not len(definition_terms) == len(definition_descriptions):
        warn("{definition_list} could not be parsed properly".format(definition_list=definition_list), RuntimeWarning)
    else:
        term_description_doubles = zip(definition_terms, definition_descriptions)
        info_tuple_iterator = extract_info_tuple_iterator_from_definition_term_description_doubles(term_description_doubles)
    return info_tuple_iterator

def extract_info_tuple_iterator_from_definition_term_description_doubles(term_description_doubles):
    result = map(extract_info_tuple_from_definition_term_description_double, term_description_doubles)
    return result

def extract_info_tuple_from_definition_term_description_double(term_description_double):
    definition_term, definition_description = term_description_double
    
    anchor_with_relative_link_to_paper_page = definition_term.find("a", title="Abstract")
    relative_link_to_paper_page = anchor_with_relative_link_to_paper_page.get("href")
    link_to_paper_page = concatenate_relative_link_to_arxiv_base_url(relative_link_to_paper_page)
    
    title_division = definition_description.find("div", attrs={"class":"list-title"})
    title_header_span = title_division.find("span", text="Title:", attrs={"class":"descriptor"})
    title_text_untrimmed = title_header_span.next_sibling
    title = title_text_untrimmed.strip()
    
    authors_division = definition_description.find("div", attrs={"class":"list-authors"})
    authors_division_anchors = authors_division.find_all("a")
    authors = map(lambda link: link.get("href"), authors_division_anchors)
    author_links = map(lambda link: link.text, authors_division_anchors)
    author_to_author_link_doubles = zip(authors, author_links)
    
    subjects_division = definition_description.find("div", attrs={"class":"list-subjects"})
    primary_subject_span = subjects_division.find("span", attrs={"class":"primary-subject"})
    primary_subject = primary_subject_span.text
    
    # secondary_subjects = primary_subject_span.next_sibling
    secondary_subjects = None # stub
    
    result = (link_to_paper_page, title, author_to_author_link_doubles, primary_subject, secondary_subjects)
    return result

###############
# Main Runner #
###############

# @todo get rid of this section after package becomes stable

def main():
    #print("Research Fields & Recent Page Links")
    #p1(arxiv_recent_page_title_and_page_link_string_iterator())
    print("\n\n")
    print("Testing")
    recent_link = "https://arxiv.org/list/econ.TH/recent"
    print("recent_link : {0}".format(recent_link))
    print(list(extract_info_from_recent_page_url(recent_link)))
    return None

if __name__ == '__main__':
    main()
