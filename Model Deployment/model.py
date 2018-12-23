import pickle
from interface import implements, Interface


class IModel(Interface):
    def get_content_model(self):
        pass

    def get_stance_model(self):
        pass

    def get_vocab_char(self):
        pass


class Model(implements(IModel)):
    def get_content_model(self):
        load_content_model = pickle.load(open('model/content_based_model_v1.0.sav', 'rb'))

        return load_content_model

    def get_stance_model(self):
        load_stance_model = pickle.load(open('model/stance_based_model_v1.0.sav', 'rb'))

        return load_stance_model

    def get_vocab_char(self):
        load_vocab_char = pickle.load(open('vocab/vocab_char_v1.0.pickle', 'rb'))

        return load_vocab_char