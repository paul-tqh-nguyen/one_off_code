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
* MongoDB Connection Utilities
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
import json
import pymongo
import getpass

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

@lru_cache(maxsize=1024)
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

def extract_info_from_recent_page_url_as_json(recent_page_url):
    info_tuples = extract_info_from_recent_page_url(recent_page_url)
    json_iterator = map(recent_page_url_info_tuple_to_json, info_tuples)
    return json_iterator

def recent_page_url_info_tuple_to_json(info_tuple):
    link_to_paper_page, title, author_to_author_link_dictionary, primary_subject, secondary_subjects_iterator, abstract = info_tuple
    secondary_subjects = secondary_subjects_iterator
    json_dict = {"page_link" : link_to_paper_page,
                 "research_paper_title" : title,
                 "author_info" : author_to_author_link_dictionary, 
                 "primary_subject" : primary_subject, 
                 "secondary_subjects" : secondary_subjects,
                 "abstract" : abstract}
    json_string = json.dumps(json_dict)
    return json_string

def extract_info_from_recent_page_url(recent_page_url):
    tuple_without_abstract_iterator = extract_info_without_abstract_from_recent_page_url(recent_page_url)
    result_iterator = map(append_abstract_to_info_extracted_from_recent_page_url, tuple_without_abstract_iterator)
    return result_iterator

def append_abstract_to_info_extracted_from_recent_page_url(info_tuple):
    link_to_paper_page, title, author_to_author_link_dictionaries, primary_subject, secondary_subjects = info_tuple
    abstract = abstract_text_from_arxiv_paper_url(link_to_paper_page)
    info_tuple = (link_to_paper_page, title, author_to_author_link_dictionaries, primary_subject, secondary_subjects, abstract)
    return info_tuple

def abstract_text_from_arxiv_paper_url(paper_url):
    text = get_text_at_url(paper_url)
    soup = BeautifulSoup(text, features="lxml")
    abstract_block_quote = soup.find("blockquote", {"class": "abstract mathjax"})
    abstract_span = abstract_block_quote.find("span", {"class": "descriptor"})
    abstract_text_raw = abstract_span.next_sibling
    abstract_text = abstract_text_raw.replace("\n", " ").strip()
    return abstract_text

def extract_info_without_abstract_from_recent_page_url(recent_page_url):
    '''The "Recent" page includes a bunch of papers. We return an iterator yielding tuples. The tuples are of the form (LINK_TO_PAPER_PAGE, TITLE, AUTHOR_TO_AUTHOR_LINK_DICTIONARIES, PRIMARY_SUBJECT, SECONDARY_SUBJECTS).'''
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
    authors = map(lambda link: link.text, authors_division_anchors)
    author_relative_links = map(lambda link: link.get("href"), authors_division_anchors)
    author_links = map(concatenate_relative_link_to_arxiv_base_url, author_relative_links)
    author_to_author_link_doubles = zip(authors, author_links)
    author_to_author_link_dictionary_iterator = map(author_to_author_link_double_to_author_to_author_link_dictionary, author_to_author_link_doubles)
    author_to_author_link_dictionaries = list(author_to_author_link_dictionary_iterator)
    
    subjects_division = definition_description.find("div", attrs={"class":"list-subjects"})
    primary_subject_span = subjects_division.find("span", attrs={"class":"primary-subject"})
    primary_subject = primary_subject_span.text

    secondary_subjects_unprocessed = primary_subject_span.next_sibling
    secondary_subjects = list(filter(bool, (map(lambda string : string.strip(), secondary_subjects_unprocessed.split(";")))))
    
    result = (link_to_paper_page, title, author_to_author_link_dictionaries, primary_subject, secondary_subjects)
    return result

def author_to_author_link_double_to_author_to_author_link_dictionary(author_to_author_link_double):
    author, author_link = author_to_author_link_double
    author_to_author_link_dictionary = {"author" : author,
                                        "author_link" : author_link}
    return author_to_author_link_dictionary

################################
# MongoDB Connection Utilities #
################################

MONGO_DB_CONNECTION_URL_FORMAT_STRING = "mongodb+srv://{username}:{password}@arxiv-news-paper-v2tf1.mongodb.net/test?retryWrites=true&w=majority"

def arxiv_mongo_db_connection_url(username0, password0):
    username = urllib.parse.quote_plus(username0)
    password = urllib.parse.quote_plus(password0)
    mongo_db_connection_url = "mongodb+srv://{username}:{password}@arxiv-news-paper-v2tf1.mongodb.net/test?retryWrites=true&w=majority".format(username=username, password=password)
    return mongo_db_connection_url

@lru_cache(maxsize=32)
def arxiv_mongo_db_connection(username=None, password=None):
    if username is None:
        username = input("Username: ")
    if password is None:
        password = getpass.getpass("Password: ")
    mongo_db_connection_url = arxiv_mongo_db_connection_url(username, password)
    client = pymongo.MongoClient(mongo_db_connection_url)
    db = client["arxivRecentPapers"]
    return db

def arxiv_recent_papers_collection(username=None, password=None):
    db = arxiv_mongo_db_connection(username, password)
    arxiv_recent_papers_collection = db.recentPapers
    return arxiv_recent_papers_collection

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
    #print(list(extract_info_from_recent_page_url(recent_link)))
    p1(extract_info_from_recent_page_url_as_json(recent_link), 4)
    return None

if __name__ == '__main__':
    main()
