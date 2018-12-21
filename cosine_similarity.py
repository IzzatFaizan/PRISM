import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from search import Search
import numpy as np

news = ['najib diikat jamin',
        'kfc tidak halal',
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
        'dewan rakyat lulus RUU mansuh akta anti berita tidak benar',
        'polis nafi serbu apartmen mewah',
        'pdrm beri jaminan proses pilihan raya berjalan lancar',
        'malaysia juara piala aff suzuki 2010',
        'pakatan harapan menang pru 14',
        'bn tewas pru 14',
        'pakatan harapan tewas pru 14',
        'bn menang pru 14']


def cosine_sim(text1, text2, vectorizer):
    tfidf_test = vectorizer.fit_transform([text1, text2])
    return (tfidf_test * tfidf_test.T).A[0, 1]


def cosine_sim2(u, v):
    return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))


raw1 = "najib diikat jamin"

raw2 = "najib mati dibunuh"

tfidf = pickle.load(open('vocab/vocab_char.pickle', 'rb'))

response1 = tfidf.transform([raw1])
response2 = tfidf.transform([raw2])

sim_1 = str(cosine_sim(raw1, raw2, tfidf))
sim_2 = str(cosine_sim2(response1.toarray()[0], response2.toarray()[0]))

# first function, pass the raw text and get vectors inside the function
print("similarity cosine_sim :" + sim_1)
# second function, pass the vectors calculated above
print("similarity cosine_sim2 :" + sim_2)

'''
def cos_sim(news):
    search_object = Search()
    related_object = search_object.search_input(keyword=news)

    tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                            max_features=5000)
    tfidf_matrix = tfidf_vect_ngram_char.fit_transform(news, related_object['snippet'])
    print(tfidf_matrix.shape(4, 11))

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()

    print(similarity)

    pass
'''''

'''
# the counts we computed above
sentence_m = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
sentence_h = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0])
sentence_w = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1])

# We should expect sentence_m and sentence_h to be more similar
print(cos_sim(sentence_m, sentence_h))  # 0.5
print(cos_sim(sentence_m, sentence_w))  # 0.25
'''
