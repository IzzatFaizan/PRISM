from verification import Verification
from extractor import Extractor
from flask import Flask, jsonify, render_template
from search import Search

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/verificationfacade/url/<string:url>', methods=['GET'])
def execute_detection_url(url):
    extractor_obj = Extractor()
    extracted_news = extractor_obj.extract_news(url)
    print(extracted_news)

    if extracted_news == 'error extract':
        return jsonify({'error': 'Maaf, masalah dengan laman web'})

    elif extracted_news == 'invalid URL':
        return jsonify({'error': 'Maaf, salah URL laman web'})

    else:
        return execute_detection_news(extracted_news)


@app.route('/api/verificationfacade/news/<string:news>', methods=['GET'])
def execute_detection_news(news):
    detect_news = Verification()
    label, prob = detect_news.detect_fake_news(news)
    return jsonify({'result': {'label': label, 'probability': prob}})


@app.route('/api/verificationfacade/news/stance/<string:news>', methods=['GET'])
def execute_detection_news_stance(news):
    detect_news = Verification()

    search_object = Search()
    related_object = search_object.search_input(keyword=news)
    print(related_object)

    results = []
    fa, re = 0, 0
    for i in range(len(related_object)):
        print(related_object[i]['snippet'])
        label, prob = detect_news.detect_fake_news_stance([news], [related_object[i]['snippet']])
        fa = fa + prob[0]
        re = re + prob[1]
        print(fa, re)

        results.append({i: {'label': label, 'probability': prob[0]}})

    return jsonify({'result': results})


if __name__ == "__main__":
    app.run()
