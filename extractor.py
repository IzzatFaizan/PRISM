import base64
import requests
from interface import implements, Interface


class IExtractor(Interface):
    def extract_news(self, url):
        pass


class Extractor(implements(IExtractor)):

    # return status code, indicating whether a successful connection can be made or not
    def check_url(self, url):
        decoded_url = base64.b64decode(url.encode()).decode()
        return requests.get(decoded_url).status_code

    def extract_news(self, url):

        if self.check_url(url) == 200:
            try:
                extracted_news = requests.get('http://miclip.ddns.net/readibility/vbtext.php?base64url=' + url).text
                return extracted_news

            except:
                return 'error extraction'

        else:
            return 'invalid URL'
