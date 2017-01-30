from __future__ import division
import requests
import unicodedata
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import json
import csv

def read_file(path):
    ''' Reads csv file. '''
    return pd.read_csv(path)

def open_json(path):
    ''' Reads json file. '''
    with open(path) as f:
        file_ = json.load(f)
    return file_

def convert_unicode_to_str(unicode):
    ''' Converts unicode to string. '''
    return unicodedata.normalize('NFKD', unicode).encode('ascii','ignore')

def json_to_list(json_file):
    ''' Appends each entry (article) in a json file to a list. '''
    json_file_new = []
    for json_entry in json_file:
        json_entry = json_entry.replace("\'title\'", '///')
        json_entry = json_entry.replace("\'article\'", '///')
        json_entry = json_entry.replace("\'url\'", '///')
        json_entry = json_entry.split('///')
        for json in json_entry:
            json = convert_unicode_to_str(json)
        json_file_new.append(json_entry[1:])
    return json_file_new

def list_to_csv(list_of_lists, filename):
    ''' Cleans each entry of a list (the url, title and article body), 
    creates a date as a datetime object. Formats 4 columns. Writes to a csv file.

    Different html formatting for NYT before and after 1996. '''
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('Title', 'Article', 'Date', 'URL'))
        for lst in list_of_lists:
            if lst[0][11:25] == 'learning.blogs':
                date_correct_format = lst[0][43:45]+'-'+lst[0][46:48]+'-'+lst[0][38:42]
                date = datetime.strptime(date_correct_format, '%m-%d-%Y')
                lst[2] = lst[2].replace(': u','').replace('}','')
                lst[2] = lst[2][1:-1]
                lst[0] = lst[0].replace(': u','')
                lst[0] = lst[0][1:-3]
                lst[1] = lst[1].replace(': [','')
                lst[1] = lst[1][0:-3]
                soup = BeautifulSoup(lst[1], 'html.parser')
                lst[1] = ''.join(str(i.string) for i in soup)
                if date.year < 1996:
                    lst[1] = ''.join(soup.text.replace('\\n',''))
                else:
                    lst[1] = ''.join(str(i.string) for i in soup)
                writer.writerow((lst[2], lst[1], date.date(), lst[0]))
            elif len(lst) < 3:
                print lst[0]
            elif lst[0][32:34]+'-'+lst[0][35:37]+'-'+lst[0][27:31] == 'm/-00-es.c':
                print lst[0]
            else:
                date_correct_format = lst[0][32:34]+'-'+lst[0][35:37]+'-'+lst[0][27:31]
                date = datetime.strptime(date_correct_format, '%m-%d-%Y')
                lst[2] = lst[2].replace(': u','').replace('}','')
                lst[2] = lst[2][1:-1]
                lst[0] = lst[0].replace(': u','')
                lst[0] = lst[0][1:-3]
                lst[1] = lst[1].replace(': [','')
                lst[1] = lst[1][0:-3]
                soup = BeautifulSoup(lst[1], 'html.parser')
                if date.year < 1996:
                    lst[1] = ''.join(soup.text.replace('\\n',''))
                else:
                    lst[1] = ''.join(str(i.string) for i in soup)
                writer.writerow((lst[2], lst[1], date.date(), lst[0]))

if __name__ == '__main__':
    ''' Cleans data and writes it to a csv file with 4 columns: 
    Title, Article (content), Date (datetime object), URL. '''
    json_file = open_json(path)
    list_of_jsons = json_to_list(json_file)
    list_to_csv(list_of_jsons, filename)
