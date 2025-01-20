import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import time
import threading
import datetime

def nytimes_scraper(soup):
   articles = []
   for x in soup.findAll("article"):
      articles.append(x.find('a').text)
   return articles

def cnn_scraper(soup):
   articles = []
   for x in soup.findAll(class_ = 'container__headline-text'):
      articles.append(x.text)
   return articles

def yh_scraper(soup):
   articles = []
   for x in soup.find_all(class_ = 'content'):
      articles.append(x.text)
   return articles

def stat_scraper(soup):
   articles = []
   for x in soup.find_all(class_ = 'topic-block__preview-title'):
      articles.append(x.text.strip())
   return articles

def ap_scraper(soup):
   articles = []
   for x in soup.find_all(class_ = 'PagePromoContentIcons-text'):
      articles.append(x.text)
   return articles[-1:1]

def abc_scraper(soup):
   articles = []
   for x in soup.find_all(class_ = 'PFoxV eBpQD rcQBv bQtjQ lQUdN GpQCA mAkiF FvMyr WvoqU nPLLM tuAKv ZfQkn GdxUi'):
      articles.append(x.text)
   return articles


class Scraper:
    def __init__(self, urls, update_time, path_download, path_init=None):
        """
        urls: dictionary with key being news source (nytimes, cnn, stat, ap, abc) and value being list of URLs
        update_time: int, how often to scrape in seconds
        path_download: string, where to store the DataFrame as JSON
        path_init: string, path to an existing JSON with headline information
        """
        self.urls = urls
        self.update_time = update_time
        self.path_download = path_download
        self.path_init = path_init
        self.timer = None
        self.running = False

    def start(self):
        if self.running:
            print("Scraping is already running.")
            return

        self.running = True
        self._scrape()
    
    def _scrape(self):
        print(f"Started scraping at {datetime.datetime.now().time()}")
        
        # Initialize or load the DataFrame
        if self.path_init and pd.io.common.file_exists(self.path_init):
            df = pd.read_json(self.path_init)
        else:
            df = pd.DataFrame(columns=['headline', 'date', 'source'])
        
        timestamp = time.time()
        scrap_dict = {
            'nytimes': nytimes_scraper,
            'cnn': cnn_scraper,
            'stat': stat_scraper,
            'ap': ap_scraper,
            'abc': abc_scraper
        }
        
        date, headline, sources = [], [], []
        for source, urls in self.urls.items():
            for url in urls:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                    new_headlines = scrap_dict[source](soup)
                    date.extend([timestamp] * len(new_headlines))
                    headline.extend(new_headlines)
                    sources.extend([source] * len(new_headlines))
                except requests.RequestException as e:
                    print(f"Error fetching headlines from {source}: {e}")
                time.sleep(5)  
        
        new_data = pd.DataFrame({"headline": headline, "date": date, "source": sources})
        df = pd.concat([df, new_data], ignore_index=True).drop_duplicates(subset='headline')
        
        df.to_json(self.path_download)
        print(f"Finished scraping at {datetime.datetime.now().time()}")
        
        if self.running:
            self.timer = threading.Timer(self.update_time, self._scrape)
            self.timer.start()

    def stop(self):
        """Stops the scraping process."""
        if self.timer:
            self.timer.cancel()
        self.running = False
        print("Scraping stopped.")
         
