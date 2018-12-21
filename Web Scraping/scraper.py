from urllib.request import Request, urlopen as uReq
from bs4 import BeautifulSoup as soup
# import pandas as pd


def make_soup(website):
    req = Request(website, headers={"User-Agent": "Mozilla/5.0"})
    uClient = uReq(req)
    page_html = uClient.read()
    page_soup = soup(page_html, "html.parser")
    return page_soup


SINAR_NEWS_URL = "http://www.sinarharian.com.my/"


def forge_url(q):
    return SINAR_NEWS_URL.format(q.replace(' ', '.'))


news_url = forge_url('avengers')
make_soup(news_url)
website = make_soup(news_url)


print(website.h3)


