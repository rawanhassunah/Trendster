import requests
import json
from datetime import datetime, timedelta
import time

def single_query(page_offset, link, payload):
   response = requests.get(link+'?page={}'.format(page_offset), params=payload)
   if response.status_code != 200:
       print 'Failed: status_code', response.status_code
   else:
       return response.json()

def get_many(days, link, filename, api_key, fq):
    responses = []
    not_scraped = []
    today = datetime(2017, 1, 14)
    yesterday = datetime(2017, 1, 13)
    for day in range(days):
        end = str(today).replace('-','')
        day = timedelta(hours=24)
        begin = str(yesterday).replace('-','')
        payload = {'api-key': api_key, 'end_date': end[:8], 'begin_date': begin[:8], 'fq': fq}
        print 'Scraping period: %s - %s ' % (str(yesterday), str(today))

        for i in xrange(120):
           offset = i
           payload['offset'] = i
           time.sleep(7)
           response = requests.get(link+'?page={}'.format(i), params=payload)
           if response.status_code != 200:
               not_scraped.append(i)
               print 'Offset:', i
               print 'Failed: status_code', response.status_code
           else:
               print 'page_number', i
               print '-------------------------------------------'
               for item in response.json()["response"]["docs"]:
                   responses.append(item)
        save_to_json(filename, responses)
        today = today - day
        yesterday = yesterday - day
    return not_scraped, responses

def save_to_json(filename, datapoints):
  with open(filename, 'w') as f:
      json.dump(datapoints, f)

def open_json_file(filename):
  with open(filename) as f:
      data = json.load(f)
  return data

if __name__ == '__main__':
    api_key = 'Enter API key here'
    link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    filename = 'Enter json file dump file here'
    fq = 'section_name:("Arts" "Blogs" "Business" "Business Day" "Education" "Fashion & Style" "Health" "International Home" "Learning" "Magazine" "National" "Opinion" "Science" "Style" "Sunday Magazine" "T Magazine" "T:Style" "Technology" "Times Topics" "TimesMachine" "Travel" "U.S." "Universal" "Washington" "World" "New York" "N.Y./Region" "N.Y. / Region")'

    not_scraped, responses = get_many(days=365,link=link,filename=filename,api_key=api_key,fq=fq)
