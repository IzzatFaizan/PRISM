from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from search import Search
from verification import Verification

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

labels = ['Real', 'Fake', 'Real', 'Real', 'Real', 'Fake', 'Fake', 'Fake', 'Real', 'Real', 'Fake', 'Fake', 'Fake',
          'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Fake', 'Fake']


def execute_detection_news_stance(news):
    detect_news = Verification()

    search_object = Search()
    related_object = search_object.search_input(keyword=news)
    # print(related_object)

    fa, re = 0, 0
    fa_count, re_count = 0, 0

    for i in range(len(related_object)):
        # print(related_object[i]['snippet'])

        label, prob = detect_news.detect_fake_news_stance([news], [related_object[i]['snippet']])
        fa = fa + prob[0]
        re = re + prob[1]

        if prob[0] > prob[1]:
            fa_count += 1
        elif prob[0] < prob[1]:
            re_count += 1

    if fa_count > re_count:
        label = 'Fake'

    elif fa_count < re_count:
        label = 'Real'

    if fa > re:
        prob_label = 'Fake'

    elif fa < re:
        prob_label = 'Real'

    print(fa_count, fa, re_count, re, '\n\n')

    return label, prob_label


resultsss = []
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
    print(resultsss[i][1])

accuracy_label = correct_count_label / len(news)
accuracy_prob = correct_count_prob / len(news)

print('Label Accuracy : ', accuracy_label)
print('Probability Accuracy : ', accuracy_prob)

# refer cosine similarity