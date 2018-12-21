import pickle
from interface import Interface, implements
import numpy as np


class IVerification(Interface):
    def detect_fake_news(self, news):
        pass

    def detect_fake_news_stance(self, news, source):
        pass


class Verification(implements(IVerification)):
    def detect_fake_news(self, news):
        load_model = pickle.load(open('model/content_based_model.sav', 'rb'))
        prediction = load_model.predict([news])
        prob = load_model.predict_proba([news])

        if prediction == 0:
            return 'Berita Palsu', prob[0][0]
        else:
            return 'Berita Benar', prob[0][1]

    def detect_fake_news_stance(self, news, source):
        load_vocab = pickle.load(open('vocab/vocab_char.pickle', 'rb'))
        load_model = pickle.load(open('model/stance_based_model.sav', 'rb'))

        news_tfidf = load_vocab.transform(news)
        source_tfidf = load_vocab.transform(source)

        # print(news_tfidf)
        # print(source_tfidf)
        concate_features = np.concatenate((news_tfidf.toarray(), source_tfidf.toarray()), axis=1)

        prediction = load_model.predict(concate_features)
        prob = load_model.predict_proba(concate_features)

        print(prediction)
        print(prob)

        if prediction == 'Fake':
            return 'Fake', prob[0]
        else:
            return 'Real', prob[0]
