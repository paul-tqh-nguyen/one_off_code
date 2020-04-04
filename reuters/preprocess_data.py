#!/usr/bin/python3 -OO

"""
Sections:
* Imports
* Globals
* Preprocessing Utilities
* Driver
"""

###########
# Imports #
###########

import os
import pandas as pd
from bs4 import BeautifulSoup
from typing import Iterable
from misc_utilites import debug_on_error, eager_map, at_most_one, tqdm_with_message

###########
# Globals #
###########

DATA_DIRECTORY = "./data/"
PREPROCESSED_DATA_DIR = './preprocessed_data/'
ALL_DATA_OUTPUT_CSV_FILE = os.path.join(PREPROCESSED_DATA_DIR,'extracted_data.csv')

###########################
# Preprocessing Utilities #
###########################

def gather_sgm_files() -> Iterable[str]:
    all_data_entries = os.listdir('./data/')
    sgm_files = map(lambda sgm_file_name: os.path.join(DATA_DIRECTORY, sgm_file_name), filter(lambda entry: '.' in entry and entry.split('.')[-1]=='sgm', all_data_entries))
    return sgm_files

def parse_sgm_files() -> pd.DataFrame:
    rows = []
    for sgm_file in gather_sgm_files(): # @todo parallelize this
        with open(sgm_file, 'rb') as sgm_text:
            soup = BeautifulSoup(sgm_text,'html.parser')
            reuters_elements = soup.find_all('reuters')
            for row_index, reuters_element in enumerate(tqdm_with_message(reuters_elements, pre_yield_message_func=lambda index: f'Processing {sgm_file}', bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')):
                date_element = at_most_one(reuters_element.find_all('date'))
                topics_element = at_most_one(reuters_element.find_all('topics'))
                topic_elements = topics_element.find_all('d')
                places_element = at_most_one(reuters_element.find_all('places'))
                place_elements = places_element.find_all('d')
                people_element = at_most_one(reuters_element.find_all('people'))
                person_elements = people_element.find_all('d')
                orgs_element = at_most_one(reuters_element.find_all('orgs'))
                org_elements = orgs_element.find_all('d')
                exchanges_element = at_most_one(reuters_element.find_all('exchanges'))
                exchange_elements = exchanges_element.find_all('d')
                companies_element = at_most_one(reuters_element.find_all('companies'))
                company_elements = companies_element.find_all('d')
                unknown_elements = reuters_element.find_all('unknown')
                text_element = at_most_one(reuters_element.find_all('text'))
                text_element_title = at_most_one(text_element.find_all('title'))
                text_element_dateline = at_most_one(text_element.find_all('dateline'))
                text_element_body = at_most_one(text_element.find_all('body'))
                get_element_text = lambda element: element.text
                row = {
                    'date': date_element.text.strip(),
                    'topics': eager_map(get_element_text, topic_elements),
                    'places': eager_map(get_element_text, place_elements),
                    'people': eager_map(get_element_text, person_elements),
                    'orgs': eager_map(get_element_text, org_elements),
                    'exchanges': eager_map(get_element_text, exchange_elements),
                    'companies': eager_map(get_element_text, company_elements),
                    'unknown': eager_map(get_element_text, unknown_elements),
                    'text_title': text_element_title.text if text_element_title else None,
                    'text_dateline': text_element_dateline.text if text_element_dateline else None,
                    'text': text_element_body.text if text_element_body else None,
                    'file': sgm_file,
                    'reuter_element_position': row_index,
                }
                rows.append(row)
    df = pd.DataFrame(rows)
    return df

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    if not os.isdir(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)
    df = parse_sgm_files()
    df.to_csv(ALL_DATA_OUTPUT_CSV_FILE, index=False)
    return

if __name__ == '__main__':
    main()
