#!/usr/bin/python3
"#!/usr/bin/python3 -OO"

"""
Sections:
* Imports
* Preprocessing Utilities
* Driver
"""

# @todo finish the top-level doc string.

###########
# Imports #
###########

import os
from bs4 import BeautifulSoup
from typing import Iterable, List
from misc_utilites import debug_on_error, eager_map, at_most_one

###########
# Globals #
###########

DATA_DIRECTORY = "./data/"

###########################
# Preprocessing Utilities #
###########################

def gather_sgm_files() -> Iterable[str]:
    all_data_entries = os.listdir('./data/')
    sgm_files = map(lambda sgm_file_name: os.path.join(DATA_DIRECTORY, sgm_file_name), filter(lambda entry: '.' in entry and entry.split('.')[-1]=='sgm', all_data_entries))
    return sgm_files

def parse_sgm_files() -> List:
    rows = []
    for sgm_file in gather_sgm_files():
        with open(sgm_file, 'r') as sgm_text:
            soup = BeautifulSoup(sgm_text,'html.parser')
            reuters_elements = soup.find_all('reuters')
            for reuters_element in reuters_elements:
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
                unknown_element = at_most_one(reuters_element.find_all('unknown'))
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
                    'unknown': unknown_element.text,
                    'text_title': text_element_title.text if text_element_title else None,
                    'text_dateline': text_element_dateline.text if text_element_dateline else None,
                    'text': text_element_body.text if text_element_body else None,
                    'file': sgm_file,
                }
                rows.append(row)
                print(f'row {row}')
    return rows

##########
# Driver #
##########

@debug_on_error # @todo get rid of this
def main() -> None:
    print(parse_sgm_files())
    return

if __name__ == '__main__':
    main()
