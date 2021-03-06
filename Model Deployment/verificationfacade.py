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

    if extracted_news == 'error extraction of the input URL':
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
    global label, probs
    detect_news = Verification()

    search_object = Search()
    related_object = search_object.search_input(keyword=news)
    print(related_object)

    # results = []
    fa_count, re_count = 0, 0
    fa_prob, re_prob = 0, 0

    for i in range(len(related_object)):
        print(related_object[i]['snippet'])
        label, prob = detect_news.detect_fake_news_stance([news], [related_object[i]['snippet']])

        if label == 'Fake':
            fa_prob += prob[0]
            fa_count += 1

        elif label == 'Real':
            re_prob += prob[1]
            re_count += 1

    if fa_count > re_count:
        probs = fa_prob / 10
        label = 'Berita Palsu'
        print(probs)

    elif fa_count < re_count:
        probs = re_prob / 10
        label = 'Berita Benar'
        print(probs)

        # results.append({i: {'label': label, 'probability': probs}})

    '''
    resultsss = []
    labels = []
    for i in range(len(news)):
        resultsss.append(execute_detection_news_stance(news[i]))

    correct_count_label = 0
    correct_count_prob = 0

    for i in range(len(news)):

        if labels[i] == resultsss[i][0]:
            correct_count_label += 1

        if labels[i] == resultsss[i][1]:
            correct_count_prob += 1

        # print(labels[i])
        print(resultsss[i][0])
        print('\n', resultsss[i][1])
    '''

    return jsonify({'result': {'label': label, 'probability': probs}})


if __name__ == "__main__":
    app.run()
