import pickle
import numpy as np
from flask import jsonify
from search_api import Search

news = ['najib diikat jamin', 'kfc tidak halal',
        'vietnam menang dalam piala aff suzuki 2018',
        'malaysia kalah dalam piala aff suzuki 2018',
        'adib meninggal dunia',
        'tun dr mahathir mati',
        'najib mati',
        'himpunan icerd tidak wujud',
        'tun dr mahathir perdana menteri ke-7',
        'anwar bakal ganti mahathir',
        'cadburry diperbuat daripada babi',
        'sushiking haram dimakan',
        'sri serdang banjir tahun ini',
        'nurul izzah letak jawatan naib presiden pkr',
        'johor nafi ada ancaman tsunami bulan ini',
        'viral "muhyiddin letak jawatan" adalah palsu',
        'dewan rakyat lulus RRU mansuh akta anti berita tidak benar',
        'polis nafi serbu apartmen mewah',
        'pdrm beri jaminan proses pilihan raya berjalan lancar',
        'malaysia juara piala aff suzuki 2010',
        'pakatan harapan menang pru 14',
        'bn tewas pru 14',
        'pakatan harapan tewas pru 14',
        'bn menang pru 14']


def detect_fake_news_stance(self, news, source):
    load_vocab = pickle.load(open('vocab_word.pickle', 'rb'))
    load_model = pickle.load(open('model_stance.sav', 'rb'))

    news_tfidf = load_vocab.transform(news)
    source_tfidf = load_vocab.transform(source)

    print(news_tfidf)
    print(source_tfidf)
    concate_features = np.concatenate((news_tfidf.toarray(), source_tfidf.toarray()), axis=1)

    prediction = load_model.predict(concate_features)
    prob = load_model.predict_proba(concate_features)
    print(prediction)
    print(prob)

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

        results.append({i: {'label': label, 'probability': prob[1]}})

    return jsonify({'result': results})