from bs4 import BeautifulSoup
import re
from cboe import cboe


class convert(cboe):

	def __init__(self, symbol, month):
		super().__init__(symbol, month)
		self.filename = self.symbol + '.csv'
		self.setfile()

	def getRows(self, tbody, date, o_type):
		with open(self.filename, 'a') as csv:
			rows = tbody.find_all('tr')
			for cells in rows:
				if len(cells) < 4:
					break
				cells.find_all('td')
				s = date + ',' + o_type
				for td in cells:
					text = re.sub(',','', td.text)
					s+=',' + text
				csv.write(s + '\n')
	
	def getTable(self, component):
		title = component.find('h2', class_='title')
		if title != None:
			tbody = component.find('tbody')
			o_type = 'C' if 'Calls' in title.text else 'P'
			date = re.findall('[0-9]{2}/+[0-9]{2}/[0-9]{4}', title.text)
			if len(date) > 0:
				self.getRows(tbody, date[0], o_type)


	def setfile(self):
		page = BeautifulSoup(self.source, 'lxml')

		tables = page.find_all('div', class_='component')

		with open(self.filename, 'w') as csv:
			csv.write('Date,Type,Strike,Last,Net,Bid,Ask,Vol,IV,Delta,Gamma,Int\n')

		for component in tables:
			self.getTable(component)

convert('vxx', 1)
