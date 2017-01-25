from __future__ import division
import requests
import unicodedata
from bs4 import BeautifulSoup
import pandas as pd
import unicodedata
from string import punctuation
import json

''' NYT Article Scraper '''

def convert_unicode_to_str(unicode):
    ''' Converts unicode to string '''
    return unicodedata.normalize('NFKD', unicode).encode('ascii','ignore')

def urls_from_dataframe(df, article_nums, filename):
    '''Grabs urls from df.read_json. Each data point is an article from the NYT api'''
    urls = []
    for article in xrange(article_nums):
        print "Getting url {}".format(article)
        urls.append(df.response[0][article]['web_url'])
        save_to_json(filename, urls)
    print "Done appending urls!"

def get_soup(url):
    r=requests.get(url)
    return BeautifulSoup(r.text, 'html.parser')

def get_h1_title(code_section):
    return code_section.h1.text

def get_unfiltered_code(code_section,klass):
    return code_section.find_all(class_=klass)

def save_to_json(filename, datapoints):
  with open(filename, 'w') as f:
      json.dump(datapoints, f)

def open_json(filename):
    with open(filename) as f:
        file_ = json.load(f)
    return file_

def get_datapoint_info_pre(urls, filename, news_site):
    '''Given a url, uses BeautifulSoup and requests to scrape page and return url, article body, and title'''
    '''works for nyt pre 1995'''
    datapoints=[]
    counter = 0
    for url in urls:
        counter += 1
        print "Iteration: {}".format(counter)
        datapoint = {}
        soup = get_soup(url)
        # code_section =  soup.article
        try:
            datapoint["title"] = get_h1_title(soup)
            code_text = get_unfiltered_code(soup,'articleBody')
            datapoint["article"] = code_text
            datapoint["url"] = url
            datapoints.append(str(datapoint))
            save_to_json(filename, datapoints)
        except:
            pass

    return datapoints

def get_datapoint_info(urls, filename, news_site):
    '''Given a url, uses BeautifulSoup and requests to scrape page and return url, article body, and title'''
    '''works for nyt post 1995'''
    datapoints=[]
    counter = 0
    for url in urls:
        counter += 1
        print "Iteration: {}".format(counter)
        datapoint = {}
        soup = get_soup(url)
        code_section =  soup.article
        try:
            datapoint["title"] = get_h1_title(code_section)
            code_text = get_unfiltered_code(code_section,'story-content')
            datapoint["article"] = code_text
            datapoint["url"] = url
            datapoints.append(str(datapoint))
            save_to_json(filename, datapoints)
        except:
            pass

    return datapoints


if __name__ == '__main__':
    df = pd.read_json(filename)
    urls_from_dataframe(df, article_numbers, filename)
    urls = pd.read_json(filename)
    nyt_datapoints = get_datapoint_info(urls, filename, source)
