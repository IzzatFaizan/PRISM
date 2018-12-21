import mysql.connector
import requests
from bs4 import BeautifulSoup


def get_link():
    url = 'http://www.sinarharian.com.my/politik/kelab-bunga-raya-komited-lahir-pelapis-tun-mahathir-1.883465'
    source = requests.get(url)
    soup = BeautifulSoup(source.text, 'lxml')
    # Get title
    title = soup.h1.text
    # Get Content
    content = soup.find('div', {'class': 'articleBody'}).find_all('p')
    for p in content:
        content_real = p.text
    # Get Category
    category = soup.find_all('div', {'class': 'header'})
    for item in category:
        category_news = item.contents[1].find_all('a')[15].text

    t = "".join(title)
    cr = "".join(content_real)
    c = "".join(category_news)

    # Dataset Storage Connection
    conn = mysql.connector.connect(host="127.0.0.1", user="root", passwd="", database="news_dataset", charset="utf8")
    query = "INSERT INTO datacollection (url, fakenews, realnews, category) VALUES ""(\"%s\", \"%s\", \"%s\", \"%s\")"

    args = (response.url, t, cr, c)

    cur = conn.cursor()
    cur.execute(query, args)
    cur.connection.commit()
