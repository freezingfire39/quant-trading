import os
import re
import urllib

import requests as req
from bs4 import BeautifulSoup as bs

BASE_FOLDER = '/Users/yiluntong/Downloads/scrape_cboe-master'
if not os.path.exists(BASE_FOLDER):
    os.mkdir(BASE_FOLDER)

# url for base of download
BASE_URL = 'http://cfe.cboe.com/'
# where are data is listed
ARCHIVE_URL = 'http://cfe.cboe.com/market-data/historical-data-archive#VX'

res = req.get(ARCHIVE_URL)

soup = bs(res.content, 'lxml')

link_lists = soup.findAll('ul', {'class': ['inline-list', 'list', 'bluelinks']})

for l in link_lists:
    links = l.findAll('a')
    csv_links = [lnk.attrs['href'] for lnk in links]
    for c in csv_links:
        # some of the first few links are for the datashop, etc
        if '.csv' not in c:
            continue
        # skip open interest and volume summaries for now; these update daily
        if 'cfeoi.csv' in c or 'cfevoloi.csv' in c:
            continue

        filename = c.split('/')[-1]
        full_contract_name = filename.split('.')[0].split('_')[-1]
        # extract just the part with letters
        contract_name = re.search('[A-Za-z]+', full_contract_name)[0]
        folder = BASE_FOLDER + contract_name
        if not os.path.exists(folder):
            os.mkdir(folder)

        # if any numbers, put in special folder
        if bool(re.search(r'\d', full_contract_name)):
            folder = folder + '/weekly_closes'
            if not os.path.exists(folder):
                os.mkdir(folder)

        # if no numbers, goes in base folder
        urllib.request.urlretrieve(BASE_URL + c, folder + '/' + filename)
