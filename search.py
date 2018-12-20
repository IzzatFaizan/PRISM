import base64
from extractor import Extractor
from googleapiclient.discovery import build
from interface import implements, Interface


class ISearch(Interface):
    def search_input(self, keyword):
        pass


class Search(implements(ISearch)):
    def search_input(self, keyword):
        # Build a service object for interacting with the API. Visit
        # the Google APIs Console <http://code.google.com/apis/console>
        # to get an API key for your own application.
        service = build("customsearch", "v1",
                        developerKey="AIzaSyBfCFTuj3KKqSRaRllZZWuwKi3iecW8rQg")

        # respond
        res = service.cse().list(
            q=keyword,
            cx='011556233041669001480:ef-zw8gak5e',
        ).execute()

        return res['items']


# import
search_object = Search()
related_object = search_object.search_input(keyword='dr mahathir')
print(related_object)

for data in related_object:
    print(data['link'])

    encode_link = base64.b64encode(data['link'].encode()).decode()

    extract_link = Extractor()
    extracted_content = extract_link.extract_news(encode_link)

    print(extracted_content)

# API_Key Fake News Detection = AIzaSyBfCFTuj3KKqSRaRllZZWuwKi3iecW8rQg
# API_Key Test = AIzaSyBbSbWfFcHB-T4UuhddNKx_as4xseKVFME
# API_Key Test 2 = AIzaSyAHJY8s-3xjk6EMVUeAJu1yzk-3bo7lB8U
# API_Key My Project = AIzaSyCjVnWiiZTcrrKR4QOSG7pEGZzbNwaTFAE
# API_Key Ecommerce = AIzaSyCIQS3kHrZ5IuvkRMSgPiSLA2f3ZFXH5-U


# Search Engine ID = 011556233041669001480:f5q80m15zyy
