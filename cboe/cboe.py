from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager


import re
import time

class cboe():

	def __init__(self, symbol, month):
		self.symbol = symbol
		self.month = month
		self.source = self.getPageSource()

	def getPageSource(self):
		driver = webdriver.Chrome(ChromeDriverManager().install())
		driver.get('http://www.cboe.com/delayedquote/quote-table')
		ticker = driver.find_element_by_id('txtSymbol')
		ticker.send_keys(self.symbol)
		ticker.send_keys(Keys.ENTER)

		time.sleep(10)

		optionsrange = driver.find_element_by_id('ddlRange')
		Select(optionsrange).select_by_index(0)

		expiration = driver.find_element_by_id('ddlMonth')
		Select(expiration).select_by_index(self.month)

		time.sleep(5)

		filterbtn = driver.find_element_by_id('btnFilter')
		filterbtn.click()

		time.sleep(10)

		return driver.page_source

