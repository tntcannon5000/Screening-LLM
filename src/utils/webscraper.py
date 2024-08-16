import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse
from langchain.document_loaders import WebBaseLoader
import bs4

class WebScraper:
    """A class for web scraping based on Google search results."""
    def __init__(self, search_query, num_pages=5):
        """
        Initialize the WebScraper.

        Args:
            search_query (str): The search query to use for Google search.
            num_pages (int, optional): The number of pages to scrape. Defaults to 5.
        """
        self.search_query = search_query
        self.num_pages = num_pages

    def search_and_scrape(self):
        """
        Perform a Google search and scrape the resulting web pages.

        Returns:
            list: A list of documents containing the scraped content.

        Raises:
            Exception: For any other errors during scraping.
        """
        search_url = f"https://www.google.com/search?q={self.search_query}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = soup.find_all('div', class_='yuRUbf')
        urls = [link.find('a')['href'] for link in links[:self.num_pages]]
        all_documents = []
        for i, url in enumerate(urls):
            try:
                loader = WebBaseLoader(
                    web_paths=(url,),
                    bs_kwargs=dict(
                        parse_only=bs4.SoupStrainer('body')
                    ),
                )
                all_documents.extend(loader.load())
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
        
        return all_documents

    def get_scraped_data(self):
        """
        Get the scraped data from the search results.

        Returns:
            list: A list of documents containing the scraped content.
        """
        return self.search_and_scrape()