#!/usr/bin/python

"""

TODO:
    Get the authentication working with the api

Given links to an intial start page, this script searches the webpage for links to tumblr pages, attempts to download all images linked to by those webpages, searches those tumblr pages for links to more tumblr pages, and so on.

"""

import sys
import os
import urllib
import re
import filecmp
import shutil
import Image
import numpy
import pytumblr

EXTENSIONS = ['.jpg','.png','.gif']
MIN_IMAGE_HEIGHT = 500
MIN_IMAGE_WIDTH = 500

# Example Usage: clear; python img_downloader.py ./test https://www.tumblr.com/

def usage():
    print 
    print "usage: python "+__file__+" <output_directory> <link_to_webpage_to_crawl_1> <link_to_webpage_to_crawl_2> ... <link_to_webpage_to_crawl_n>"
    print 
    sys.exit(1)

def scrape_for_images(output_directory, list_of_links_to_scrape_for_images):
    
    print "The webpages we are scraping for images are: "+"".join(map(lambda x: "\n    "+x,list_of_links_to_scrape_for_images))
    
    for web_page_link in list_of_links_to_scrape_for_images:
        
        try:
            print 
            print 'Scraping '+web_page_link+' for images.'
            print 
            
            try:
                html_text = urllib.urlopen(web_page_link).read()
                print "    HTML text for "+web_page_link+" has been downloaded."
            except:
                print "    Could not HTML for "+web_page_link
                continue
            
            img_links = []
            for e in html_text.split():
                for extension in EXTENSIONS:
                    x = re.search('(?<=src=").+?(\\'+extension+')', e)
                    if x != None:
                        if x.group()[0:7] == 'http://':
                            img_links.append(x.group())
            
            print 
            print "    Images to download: "+"\n        ".join(img_links)
            print 
            for i,e in enumerate(img_links):
                print "    Downloading image at "+e
                temp_destination = os.path.join(output_directory,('z'*99)+'temp'+e.split('/')[-1][-4:])
                urllib.urlretrieve(e, temp_destination)
                destination = os.path.join(output_directory,e.split('/')[-1])
                count = 0
                need_to_replace_destination_with_temp_image = True
                downloaded_image_array = numpy.asarray(Image.open(temp_destination))
                if downloaded_image_array.shape == ():
                    print "        Bad Image."
                    os.remove(temp_destination)
                    continue
                if downloaded_image_array.shape[0] < MIN_IMAGE_HEIGHT or downloaded_image_array.shape[1] < MIN_IMAGE_WIDTH:
                    print "        Image is too small."
                    os.remove(temp_destination)
                    continue
                while os.path.isfile(destination):
                    existing_image_array = numpy.asarray(Image.open(destination))
                    are_exact_same_image = existing_image_array.shape == downloaded_image_array.shape and numpy.sum(existing_image_array - downloaded_image_array) == 0
                    if are_exact_same_image: 
                        need_to_replace_destination_with_temp_image = False
                        break
                    count += 1
                    destination = destination[:-4]+'('+str(count)+')'+destination[-4:]
                if need_to_replace_destination_with_temp_image: 
                    shutil.move(temp_destination, destination)
                else:
                    os.remove(temp_destination)
        except:
            pass

def crawl_for_more_tumblr_links(base_link):
    # returns a set of all tumblr pages linked to in the base_link page
    tumblr_links_list = set()
    html_text = urllib.urlopen(base_link).read()
    
    for e in html_text.split():
        x = re.search('(?<=http://).+?(\.tumblr\.com)', e)
        if x != None:
            tumblr_links_list.add('http://'+x.group())
    return tumblr_links_list

def post_photo():
    client = pytumblr.TumblrRestClient(###########)

    print 
    print client.info()
    print 
    print client.info()['user']['name']
    print 
    print client.create_quote("codingjester", state="queue", quote="I am the Walrus", source="Ringo")
    print 
    print client.create_text("codingjester", state="published", slug="testing-text-posts", title="Testing", body="testing1 2 3 4")
    print 
    
    
    exit()
    

def main():
    if (len(sys.argv) < 3):
        usage()
    
    post_photo()
    
    output_directory = os.path.abspath(sys.argv[1])
    if (not os.path.isdir(output_directory)): 
        os.makedirs(output_directory)
    
    web_pages_to_scrape_for_images = set(sys.argv[2:])
    
    while len(web_pages_to_scrape_for_images) > 0: # this will pretty much guarantee that it will run forever
        scrape_for_images(output_directory, web_pages_to_scrape_for_images)
        new_web_pages_to_scrape_for_images = set()
        for e in web_pages_to_scrape_for_images:
            new_tumblr_links = crawl_for_more_tumblr_links(e)
            for e2 in new_tumblr_links:
                new_web_pages_to_scrape_for_images.add(e2)
        web_pages_to_scrape_for_images = new_web_pages_to_scrape_for_images

if __name__ == '__main__':
    main()

