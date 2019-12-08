#!/usr/bin/python3 -O

"""

Utilities for named entity recognition via Wikidata scraping (to avoid banning).

Owner : paul-tqh-nguyen

Created : 11/19/2019

File Name : named_entity_recognition_via_wikidata.py

File Organization:
* Imports
* Async IO Utilities
* Wikidata Search Utilities
* Wikidata Query Service Utilities
* Most Abstract Interface

"""

###########
# Imports #
###########

import requests
import urllib.parse
import asyncio
import pyppeteer
import unicodedata
import re
import bidict
import string
import time
import warnings
from typing import List, Tuple, Set
import subprocess # @todo Remove after done debugging

LOGGING_FILE = "/home/pnguyen/Desktop/log.txt"

@profile
def _logging_print(input_string: str) -> None:
    with open(LOGGING_FILE, 'a') as f:
        f.write(input_string+'\n')
    return None

@profile
def _print_all_gnome_shell_processes() -> None:
    ps_e_process = subprocess.Popen("ps -e", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ps_e_stdout_string, _ = process.communicate()
    list(map(_logging_print, filter(lambda line: 'chrom' in line or 'gnome-shell' in line, ps_e_stdout_string.decode("utf-8").split('\n'))))
    return None

######################
# Async IO Utilities #
######################

UNIQUE_BOGUS_RESULT_IDENTIFIER = (lambda x: x)

@profile
def _indefinitely_attempt_task_until_success(coroutine, coroutine_args):
    result = UNIQUE_BOGUS_RESULT_IDENTIFIER
    while result == UNIQUE_BOGUS_RESULT_IDENTIFIER:
        task = coroutine(*coroutine_args)
        event_loop = asyncio.new_event_loop()
        from datetime import datetime; _logging_print("_indefinitely_attempt_task_until_success attempt start time {}".format(datetime.now()))
        _logging_print("All shell processes 1")
        _print_all_gnome_shell_processes()
        try:
            asyncio.set_event_loop(event_loop)
            results = event_loop.run_until_complete(asyncio.gather(task))
            if isinstance(results, list) and len(results) == 1:
                result = results[0]
        except Exception as err:
            _logging_print("err :: {}".format(err))
            pass
        finally:
            _logging_print("All shell processes 2")
            _print_all_gnome_shell_processes()
            pending_tasks = asyncio.Task.all_tasks()
            _logging_print("len(pending_tasks) {}".format(len(pending_tasks)))
            for pending_task in pending_tasks:
                print("pending_task {}".format(pending_task))
            event_loop.close()
            _logging_print("All shell processes 3")
            _print_all_gnome_shell_processes()
        if result == UNIQUE_BOGUS_RESULT_IDENTIFIER:
            warnings.warn("Attempting to execute {coroutine} on {coroutine_args} failed.".format(coroutine=coroutine, coroutine_args=coroutine_args))
            time.sleep(1)
    return result

#############################
# Wikidata Search Utilities #
#############################

WIKIDATA_SEARCH_URI_TEMPLATE = 'https://www.wikidata.org/w/index.php?sort=relevance&search={encoded_string}'

@profile
def _normalize_string_wrt_unicode(input_string: str) -> str:
    normalized_string = unicodedata.normalize('NFKD', input_string).encode('ascii', 'ignore').decode('utf-8')
    return normalized_string

PUNUCTION_REMOVING_TRANSLATION_TABLE = str.maketrans('', '', string.punctuation)

@profile
def _normalize_string_for_wikidata_entity_label_comparison(input_string: str) -> str:
    normalized_string = input_string
    normalized_string = _normalize_string_wrt_unicode(normalized_string)
    normalized_string = normalized_string.lower()
    normalized_string = normalized_string.translate(PUNUCTION_REMOVING_TRANSLATION_TABLE)
    return normalized_string

@profile
async def _most_relevant_wikidata_entities_corresponding_to_string(input_string: str) -> str:
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 1")
    _print_all_gnome_shell_processes()
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 1")
    wikidata_entities_corresponding_to_string = []
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 1.1")
    browser = await pyppeteer.launch({'headless': True})
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 1.2")
    page = await browser.newPage()
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 2")
    input_string_encoded = urllib.parse.quote(input_string)
    uri = WIKIDATA_SEARCH_URI_TEMPLATE.format(encoded_string=input_string_encoded)
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 2")
    _print_all_gnome_shell_processes()
    try:
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 3")
        _logging_print("uri {}".format(uri))
        await page.goto(uri)
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 3.5")
        await page.waitForSelector('div#mw-content-text')
        search_results_div = await page.waitForSelector('div.searchresults')
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 4")
        search_results_paragraph_elements = await search_results_div.querySelectorAll('p')
        search_results_have_shown_up = None
        for paragraph_element in search_results_paragraph_elements:
            _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 5")
            paragraph_element_classname_string = await page.evaluate('(p) => p.className', paragraph_element)
            paragraph_element_classnames = paragraph_element_classname_string.split(' ')
            _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 6")
            for paragraph_element_classname in paragraph_element_classnames:
                if paragraph_element_classname == 'mw-search-nonefound':
                    search_results_have_shown_up = False
                elif paragraph_element_classname == 'mw-search-pager-bottom':
                    search_results_have_shown_up = True
                if search_results_have_shown_up is not None:
                    break
            _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 7")
            if search_results_have_shown_up is not None:
                break
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 8")
        if search_results_have_shown_up:
            search_results_divs = await page.querySelectorAll('div.mw-search-result-heading')
            # _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 9")
            for search_results_div in search_results_divs:
                search_results_div_text_content = await page.evaluate('(search_results_div) => search_results_div.textContent', search_results_div)
                parsable_text_match = re.match(r'^.+\(Q[0-9]+\) +$', search_results_div_text_content)
                # _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 10")
                if parsable_text_match:
                    parsable_text = parsable_text_match.group()
                    parsable_text = parsable_text.replace(')','')
                    parsable_text_parts = parsable_text.split('(')
                    # _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 11")
                    if len(parsable_text_parts)==2:
                        (label, term_id) = parsable_text_parts
                        label = label.strip()
                        term_id = term_id.strip()
                        # _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 12")
                        if _normalize_string_for_wikidata_entity_label_comparison(label) == _normalize_string_for_wikidata_entity_label_comparison(input_string):
                            wikidata_entities_corresponding_to_string.append(term_id)
                            if len(wikidata_entities_corresponding_to_string)>5:
                                break
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 13")
    except pyppeteer.errors.NetworkError:
        pass
    finally:
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 3")
        _print_all_gnome_shell_processes()
        await page.close()
        await browser.close()
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 4")
        _print_all_gnome_shell_processes()
        _logging_print("before communicate browser.process {}".format(browser.process))
        _, errs = browser.process.communicate()
        assert errs is None
        _logging_print("after communicate browser.process {}".format(browser.process))
        _logging_print("errs {}".format(errs))
        process_is_still_running = browser.process.poll() is None
        _logging_print("process_is_still_running {}".format(process_is_still_running))
        assert not process_is_still_running
        _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 5")
        _print_all_gnome_shell_processes()
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string 14")
    _logging_print("_most_relevant_wikidata_entities_corresponding_to_string All shell processes 6")
    _print_all_gnome_shell_processes()
    return wikidata_entities_corresponding_to_string

@profile
def _string_corresponding_commonly_known_entities(input_string: str) -> List[str]:
    _logging_print("")
    _logging_print("_string_corresponding_commonly_known_entities")
    _logging_print("input_string {}".format(input_string))
    result = _indefinitely_attempt_task_until_success(_most_relevant_wikidata_entities_corresponding_to_string, [input_string])
    return result

####################################
# Wikidata Query Service Utilities #
####################################

TYPE_TO_ID_MAPPING = bidict.bidict({
    'Organization': 'Q43229',
    'Anthroponym': 'Q10856962',
    'Work': 'Q386724',
    'Natural Geographic Entity': 'Q27096220',
})

QUERY_TEMPLATE_FOR_ENTITY_COMMONLY_KNOWN_ISAS = '''
SELECT ?VALID_GENLS ?TERM
WHERE 
{{
  VALUES ?TERM {{ {space_separated_term_ids} }}.
  ?TERM wdt:P31 ?IMMEDIATE_GENLS.
  ?IMMEDIATE_GENLS 	wdt:P279* ?VALID_GENLS.
  VALUES ?VALID_GENLS {{ '''+' '.join(map(lambda type_string: 'wd:'+type_string, TYPE_TO_ID_MAPPING.values()))+''' }}.
  MINUS {{
    ?TERM wdt:P31 wd:Q4167410 .
  }}
}}
'''

WIKI_DATA_QUERY_SERVICE_URI = 'https://query.wikidata.org'

@profile
def _sparql_query_queried_variables(sparql_query:str) -> List[str]:
    queried_variables = []
    sparql_tokens = sparql_query.split()
    assert sparql_tokens[0].lower()=='select'
    for sparql_token in sparql_tokens[1:]:
        if sparql_token[0]=='?':
            queried_variables.append(sparql_token)
        else:
            break
    return queried_variables

@profile
async def _query_wikidata_via_web_scraper(sparql_query:str) -> List[dict]:
    _logging_print("_query_wikidata_via_web_scraper All shell processes 1")
    _print_all_gnome_shell_processes()
    results = []
    sparql_query_encoded = urllib.parse.quote(sparql_query)
    uri = WIKI_DATA_QUERY_SERVICE_URI+'/#'+sparql_query_encoded
    browser = await pyppeteer.launch({'headless': True})
    page = await browser.newPage()
    sparql_query_queried_variables = _sparql_query_queried_variables(sparql_query)
    number_of_variables_queried = len(sparql_query_queried_variables)
    _logging_print("_query_wikidata_via_web_scraper All shell processes 2")
    _print_all_gnome_shell_processes()
    try:
        await page.goto(uri)
        selector_query_for_arbitrary_text_inside_query_box = 'span.cm-variable-2'
        await page.waitForSelector(selector_query_for_arbitrary_text_inside_query_box)
        button = await page.querySelector('button#execute-button')
        await page.evaluate('(button) => button.click()', button)
        await page.waitForSelector('div.th-inner.sortable.both')
        column_header_divs = await page.querySelectorAll('div.th-inner.sortable.both')
        assert len(column_header_divs) == number_of_variables_queried
        variable_names = []
        for column_header_div in column_header_divs:
            variable_name = await page.evaluate('(column_header_div) => column_header_div.textContent', column_header_div)
            variable_names.append(variable_name)
        assert sparql_query_queried_variables == list(map(lambda variable_name: '?'+variable_name, variable_names))
        anchors = await page.querySelectorAll('a.item-link')
        result = dict()
        for anchor_index, anchor in enumerate(anchors):
            anchor_variable = variable_names[anchor_index%number_of_variables_queried]
            anchor_link = await page.evaluate('(anchor) => anchor.href', anchor)
            assert len(re.findall(r"^http://www.wikidata.org/entity/\w+$", anchor_link))==1
            entity_id = anchor_link.replace('http://www.wikidata.org/entity/','')
            anchor_variable_with_question_mark_prefix = '?'+anchor_variable
            result[anchor_variable_with_question_mark_prefix] = entity_id
            if (1+anchor_index)%number_of_variables_queried==0:
                assert len(result) == number_of_variables_queried
                results.append(result)
                result = dict()
    except pyppeteer.errors.NetworkError:
        pass
    finally:
        _logging_print("_query_wikidata_via_web_scraper All shell processes 3")
        _print_all_gnome_shell_processes()
        await page.close()
        await browser.close()
        _logging_print("_query_wikidata_via_web_scraper All shell processes 4")
        _print_all_gnome_shell_processes()
        _logging_print("before communicate browser.process {}".format(browser.process))
        _, errs = browser.process.communicate()
        assert errs is None
        _logging_print("after communicate browser.process {}".format(browser.process))
        _logging_print("errs {}".format(errs))
        process_is_still_running = browser.process.poll() is None
        _logging_print("process_is_still_running {}".format(process_is_still_running))
        assert not process_is_still_running
        _logging_print("_query_wikidata_via_web_scraper All shell processes 5")
        _print_all_gnome_shell_processes()
    return results

###########################
# Most Abstract Interface #
###########################

@profile
def execute_sparql_query_via_wikidata(sparql_query:str) -> List[dict]:
    _logging_print("")
    _logging_print("execute_sparql_query_via_wikidata")
    _logging_print("sparql_query {}".format(sparql_query))
    result = _indefinitely_attempt_task_until_success(_query_wikidata_via_web_scraper, [sparql_query])
    return result

@profile
def _find_commonly_known_isas(term_ids_without_item_prefix: List[str]) -> Set[Tuple[str, str]]:
    term_type_pairs = set()
    if len(term_ids_without_item_prefix) != 0:
        term_ids = map(lambda raw_term_id: 'wd:'+raw_term_id, term_ids_without_item_prefix)
        space_separated_term_ids = ' '.join(term_ids)
        sparql_query = QUERY_TEMPLATE_FOR_ENTITY_COMMONLY_KNOWN_ISAS.format(space_separated_term_ids=space_separated_term_ids)
        results = execute_sparql_query_via_wikidata(sparql_query)
        for result in results:
            term = result['?TERM']
            term_type = result['?VALID_GENLS']
            term_type_pair = (term, term_type)
            term_type_pairs.add(term_type_pair)
    return term_type_pairs

@profile
def string_corresponding_wikidata_term_type_pairs(input_string: str) -> Set[Tuple[str, str]]:
    _logging_print("string_corresponding_wikidata_term_type_pairs All shell processes 1")
    _print_all_gnome_shell_processes()
    term_ids = _string_corresponding_commonly_known_entities(input_string)
    _logging_print("")
    _logging_print("string_corresponding_wikidata_term_type_pairs")
    _logging_print("input_string {}".format(input_string))
    _logging_print("term_ids {}".format(term_ids))
    term_type_pairs = _find_commonly_known_isas(term_ids)
    term_type_pairs = [(term, TYPE_TO_ID_MAPPING.inverse[type_id]) for term, type_id in term_type_pairs]
    _logging_print("string_corresponding_wikidata_term_type_pairs All shell processes 2")
    _print_all_gnome_shell_processes()
    return term_type_pairs

@profile
def main():
    _logging_print("This module contains utilities for named entity recognition via Wikidata scraping.")
    big_string = "Barbecue in the United States From Wikipedia, the free encyclopedia Jump to navigationJump to search Part of a series on American cuisine Regional cuisines[show] History[show] Ingredients and foods[show] Styles[show] Ethnic and cultural[show] Holidays and festivals[show] Flag of the United States.svg United States portalFoodlogo2.svg Food portal vteA Southern Barbecue, 1887, by Horace Bradley In the United States, barbecue refers to a technique of cooking meat outdoors over a fire; often this is called pit barbecue, and the facility for cooking it is the barbecue pit. This form of cooking adds a distinctive smoky taste to the meat; barbecue sauce, while a common accompaniment, is not required for many styles.[1]Often the proprietors of Southern-style barbecue establishments in other areas originate from the South. In the South, barbecue is more than just a style of cooking, but a subculture with wide variation between regions, and fierce rivalry for titles at barbecue competitions.[1][2] Contents 1 Description 2 The barbecue region 3 Barbecue tradition 4 Main regional styles 4.1 Carolinas 4.2 Kansas City 4.3 Memphis 4.4 Texas 5 Other regions 5.1 Alabama 5.2 California 5.3 Hawaii 5.4 St. Louis 5.5 Other states 6 Competitions 7 See also 8 References 9 External links Description There are 3 ingredients to barbecue. Meat and wood smoke are essential. The use of a sauce or seasoning varies widely between regional traditions.The first ingredient in the barbecue tradition is the meat. The most widely used meat in most barbecue is pork, particularly the pork ribs, and also the pork shoulder for pulled pork.[1] The techniques used to cook the meat are hot smoking and smoke cooking. These cooking processes are distinct from the cold smoking preservation process. Hot smoking is where the meat is cooked with a wood fire, over indirect heat, at temperatures between 120 and 180 °F (50 and 80 °C), and smoke cooking (the method used in barbecue) is cooking over indirect fire at higher temperatures, often in the range of 250°F (121°C) ±50°F (±28°C). The long, slow cooking process take hours, as many as 18, and leaves the meat tender and juicy.[2][3] Characteristically, this process leaves a distinctive line of red just under the surface, where the myoglobin in the meat reacts with carbon monoxide from the smoke, and imparts the smoky taste essential to barbecue.[2][4][5]The second ingredient in barbecue is the wood used to smoke the meat. Since the wood smoke flavors the food, the particular type of wood used influences the process. Different woods impart different flavors, so the regional availability of the various woods for smoking influences the taste of the region's barbecue. Smoking the meat is the key, as otherwise cooking meat over an open flame is simply grilling the meat, whereas barbecue is the actual process of smoking it.Hard woods such as hickory, mesquite and the different varieties of oak impart a strong smoke flavor. Maple, alder, pecan and fruit woods such as apple, pear, and cherry impart a milder, sweeter taste. Stronger flavored woods are used for pork and beef, while the lighter flavored woods are used for fish and poultry. More exotic smoke generating ingredients can be found in some recipes; grapevine adds a sweet flavor, and sassafras, a major flavor in root beer, adds its distinctive taste to the smoke.The last, and in many cases optional, ingredient is the barbecue sauce. There are no constants, with sauces running the gamut from clear, peppered vinegars to thick, sweet, tomato and molasses sauces to mustard-based barbecue sauces, which themselves range from mild to painfully spicy. The sauce may be used as a marinade before cooking, applied during cooking, after cooking, or used as a table sauce. An alternate form of barbecue sauce is the dry rub, a mixture of salt and spices applied to the meat before cooking.[6]The barbecue region The origins of American barbecue date back to colonial times, with the first recorded mention in 1672[7] and George Washington mentions attending a barbicue in Alexandria, Virginia, in 1769. As the country expanded westwards along the Gulf of Mexico and north along the Mississippi River, barbecue went with it.[1] A slab of barbecued pork ribs at Oklahoma Joe's in Tulsa. The core region for barbecue is the southeastern region of the United States, an area bordered on the west by Texas and Oklahoma, on the north by Missouri, Kentucky, and Virginia, on the south by the Gulf of Mexico, and on the east by the Atlantic Ocean. While barbecue is found outside of this region, the fourteen core barbecue states contain 70 of the top 100 barbecue restaurants, and most top barbecue restaurants outside the region have their roots there.[1]Barbecue in its current form grew up in the South, where cooks learned to slow-roast tough cuts of meat over fire pits to make them tender.These humble beginnings are still reflected in the many barbecue restaurants that are operated out of hole-in-the-wall (or dive) locations; the rib joint is the purest expression of this. Many of these will have irregular hours, and remain open only until all of a day's ribs are sold; they may shut down for a month at a time as the proprietor goes on vacation. Despite these unusual traits, rib joints will have a fiercely loyal clientele.[1]Barbecue is strongly associated with Southern cooking and culture due to its long history and evolution in the region. Indian corn cribs, predecessors to Southern barbecue, were described during the Hernando de Soto expedition in southwest Georgia, and were still around when English settlers arrived two centuries later. Early usage of the verb barbecue, derived from Spanish barbacoa, meant to preserve (meat) by drying or slowly roasting; the meaning became closer to that of its modern usage as a specific cooking technique by the time Georgia was colonized.[8] Today, barbecue has come to embody cultural ideals of communal recreation and faithfulness in certain areas. These ideals were historically important in farming and frontier regions throughout the South and parts of the Midwest with influences from the South.[9] As such, due to the strong cultural associations that it holds in these areas, barbecue has attained an important position in America's culinary tradition.Parts of the Midwest also incorporate their own styles of barbecue into their culinary traditions. For example, in Kansas City, barbecue entails a wide variety of meats, sweet and thick sauces, dry rubs, and sliced beef brisket. Kansas City barbecue is a result of the region’s history; a combination of the cooking techniques brought to the city by freed slaves and the Texas cattle drives during the late nineteenth century has led to the development of the region's distinctive barbecue style.[10] Barbecue as a cultural tradition spread from the South and was successfully incorporated into several Midwestern regions such as western Missouri, again owing to the cultural ideals that the barbecue tradition represents and the need for locals to express those ideals. Variations of these ideals by region are reflected in the great diversity of barbecue styles and traditions within the United States.Barbecue tradition Barbecue has been a staple of American culture, especially Southern American culture, since colonial times. As it has emerged through the years many distinct traditions have become prevalent in the United States. The pig, the essential ingredient to any barbecue, became a fundamental part of food in the South in the 18th century because the pig requires little maintenance and is able to efficiently convert feed to meat (six times quicker than beef cattle).[11] As a result of the prevalence of hogs in the South, the pig became synonymous with Southern culture and barbecue. The origins of the pig symbol with Southern Culture began as a result of its value as an economic commodity. By 1860, hogs and southern livestock were valued at double the cotton crop, at a price of half a billion dollars.[11] The majority of pigs were raised by residents of the South and as a result the pigs contributed considerably to the economic well-being of many Southerners.Pigs and barbecue were not only valuable for economic reasons but barbecue scores of hog were set aside for large gatherings and often used as an enticement for political rallies, church events, as well as harvest festival celebrations.[11] Barbecues have been a part of American history and tradition from as early as the first Independence Day celebration.[12] In the early years, Independence Day was celebrated as a formal civil gathering, in which egalitarian principles were reinforced. The traditions of Independence Day moved across the country as settlers traveled to western territories. By the 19th century, the role of barbecue in public celebration and political institutions increased significantly and it became the leading practice of communal celebrations in the South as well as the Midwest.[12] The important social, political, and cultural gatherings of barbecues have spanned three centuries and its cultural significance remains important today.Main regional styles See also: Regional variations of barbecue While the wide variety of barbecue styles makes it difficult to break barbecue styles down into regions, there are four major styles commonly referenced, Carolina and Memphis, which rely on pork and represent the oldest styles, and Kansas City and Texas, which use beef as well as pork, and represent the later evolution of the original Deep South barbecue. Pork is the most common meat used, followed by beef and veal, often with chicken or turkey in addition. Lamb and mutton are found in some areas, such as Owensboro, Kentucky (International Bar-B-Q Festival), and some regions will add other meats.[2][4]Carolinas Further information: Barbecue in North Carolina Carolina barbecue is usually pork, served pulled, shredded, or chopped, but sometimes sliced. It may also be rubbed with a spice mixture before smoking and mopped with a spice and vinegar liquid during smoking. It is probably the oldest form of American barbecue. The wood used is usually a hardwood such as oak or hickory.Two styles predominate in different parts of North Carolina. Eastern North Carolina barbecue is normally made by the use of the whole hog, where the entire pig is barbecued and the meat from all parts of the pig are chopped and mixed together. Eastern North Carolina barbecue uses a thin sauce made of vinegar and spices (often simply cayenne pepper). Western North Carolina barbecue is made from only the pork shoulder, which is mainly dark meat, and uses a vinegar-based sauce that includes the addition of varying amounts of tomato. Western North Carolina barbecue is also known as Piedmont style or Lexington style barbecue, after the town of Lexington, North Carolina, home to many barbecue restaurants and a large barbecue festival, the Lexington Barbecue Festival.[13][14][15]South Carolina has its own distinct sauce. Throughout the Columbia to Charleston corridor, barbecue is characterized by the use of a yellow Carolina Gold sauce, made from a mixture of yellow mustard, vinegar, brown sugar and other spices.Kansas City Main article: Kansas City-style barbecueKansas City-style barbecue Barbecue was brought to Kansas City, Missouri by Memphian Henry Perry. Despite these origins, the Kansas City style is characterized by a wide variety in meat, particularly including beef, pork, and lamb; and a strong emphasis on the signature ingredient, the sauce and the french fries. The meat is smoked with a dry rub, and the sauce served as a table sauce. Kansas City barbecue is rubbed with spices, slow-smoked over a variety of woods and served with a thick tomato-based barbecue sauce,[16] which is an integral part of KC-style barbecue. Major Kansas City-area barbecue restaurants include Arthur Bryant's, which is descended directly from Perry's establishment and Gates and Sons Bar-B-Q, notably spicier than other KC-style sauces with primary seasonings being cumin and celery salt.Memphis Main article: Memphis-style barbecue Memphis barbecue is primarily two different dishes: ribs, which come wet and dry, and barbecue sandwiches. Wet ribs are brushed with sauce before and after cooking, and dry ribs are seasoned with a dry rub. Barbecue sandwiches in Memphis are typically pulled pork (that is shredded by hand and not chopped with a blade) served on a simple bun and topped with barbecue sauce, and cole slaw. Of note is the willingness of Memphians to put this pulled pork on many non-traditional dishes, creating such dishes as barbecue salad, barbecue spaghetti, barbecue pizza, or barbecue nachos.[2][4]Texas Main article: Barbecue in Texas There are four generally recognized regional styles of barbecue in Texas:East Texas style, which is essentially Southern barbecue and is also found in many urban areas; Central Texas meat market style, which originated in the butcher shops of German and Czech immigrants to the region; West Texas cowboy style, which involves direct cooking over mesquite and uses goat and mutton as well as beef; and South Texas barbacoa, in which the head of a cow is cooked (originally underground).[17][18] Other regions Alabama Alabama is known for its smoked chicken which is traditionally served with Alabama white sauce, a mayonnaise-based sauce including vinegar, black pepper, and other spices. The sauce was created by Bob Gibson in Decatur, Alabama during the 1920s and served at the restaurant bearing his name, Big Bob Gibson’s Barbecue.[19] Chicken is first smoked in the pit and then coated or dunked in the white sauce. The sauce is also served at the table where it is eaten on a variety of other foods.[20]California Main article: Santa Maria-style barbecue The original use of buried cooking in barbecue pits in North America was done by the Native Americans for thousands of years, including by the tribes of California. In the late 18th and early 19th centuries eras, when the territory became Spanish Las Californias and then Mexican Alta California, the Missions and ranchos of California had large cattle herds for hides and tallow use and export. At the end of the culling and leather tanning season large pit barbecues cooked the remaining meat. In the early days of California statehood after 1850 the Californios continued the outdoor cooking tradition for fiestas.In California, the Santa Maria-style barbecue, which originated in the Central Coast region, is best known for its tri-tip beef rump, sometimes cut into steaks, which is grilled over a pit of red oak, and simply season it with salt and garlic. Versions made in towed trailers are frequently seen at farmers markets.[21] It is often served with pinto beans, pico de gallo salsa, and tortillas.HawaiiThis section does not cite any sources. Please help improve this section by adding citations to reliable sources. Unsourced material may be challenged and removed. Find sources: Barbecue in the United States – news · newspapers · books · scholar · JSTOR (September 2019) (Learn how and when to remove this template message) The cooking customs of the indigenous peoples of Polynesia became the traditional Hawaiian luau of the Native Hawaiians. It was brought to international attention by 20th century tourism to the islands.St. Louis Main article: St. Louis-style barbecue A staple of barbecuing in St. Louis is the pork steak,[22] which is sliced from the shoulder of the pig. Although now considered a part of the Midwest, Missouri was originally settled primarily by Southerners from Kentucky, Virginia, and Tennessee. These original settlers brought a strong barbecue tradition and even though successive waves of later, primarily German and Scandinavian, immigration obscured much of the state's Southern roots, the Southern influences persisted, especially throughout the Little Dixie enclave of central Missouri (connecting the Kansas City and St. Louis barbecue traditions).[citation needed]Other states Other regions of the core barbecue states tend to be influenced by the neighboring styles, and often will draw from more than one region. Southern barbecue is available outside of the core states; while far less common, the variety can be even greater. With no local tradition to draw on, these restaurants often bring together eclectic mixes of things such as Carolina pulled pork and Texas brisket on the same menu, or add in some original creations or elements of other types of cuisines.[2]CompetitionsThis section needs additional citations for verification. Please help improve this article by adding citations to reliable sources. Unsourced material may be challenged and removed. Find sources: Barbecue in the United States – news · newspapers · books · scholar · JSTOR (April 2010) (Learn how and when to remove this template message) Nationally and regionally sanctioned barbecue competitions occur. State organizations like the Florida Bar B Que Association often list competitions taking place throughout any given year. Visitors are welcome to visit these contests, and many of them hold judging classes where it is possible to become a certified barbecue judge on site.[citation needed]There are hundreds of barbecue competitions across the region every year, from small local affairs to large festivals that draw from all over the region. The American Royal World Championship contest, with over 500 teams competing, is the largest in the United States. Another major contest is the Houston BBQ world championship contest in Texas. Memphis in May World Championship Barbecue Cooking Contest is another one of the largest, and there is even a contest dedicated to sauces, the Diddy Wa Diddy National Barbecue Sauce Contest.[2][6] The nonprofit Kansas City Barbeque Society, or KCBS, sanctions over 300 barbecue contests per year, in 44 different states. Despite the Kansas City name, the KCBS judges all styles of barbecue, which is broken down into classes for ribs, brisket, pork, and chicken. "
    words = big_string.split()
    for _ in range(9999999):
        for word_index, raw_word in enumerate(words):
            word = raw_word.lower()
            _logging_print("word_index {}".format(word_index))
            _logging_print("word ]{}[".format(word))
            _string_corresponding_commonly_known_entities(word)

if __name__ == '__main__':
    main()
