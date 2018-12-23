import numpy as np
from model import Model
from interface import Interface, implements


class IVerification(Interface):
    def detect_fake_news(self, news):
        pass

    def detect_fake_news_stance(self, news, source):
        pass


class Verification(implements(IVerification)):
    def detect_fake_news(self, news):
        load = Model()
        content_model = load.get_content_model()

        prediction = content_model.predict([news])
        prob = content_model.predict_proba([news])

        if prediction == 0:
            return 'Berita Palsu', prob[0][0]
        else:
            return 'Berita Benar', prob[0][1]

    def detect_fake_news_stance(self, news, source):
        load = Model()
        load_vocab = load.get_vocab_char()
        load_model = load.get_stance_model()

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
