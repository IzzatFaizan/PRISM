import pickle

from interface import implements, Interface


class IModel(Interface):
    def get_model(self):
        pass


class Model(implements(IModel)):
    def get_model(self):

        load__content_model = pickle.load(open('model/content_based_model.sav', 'rb'))

        load__stance_model = pickle.load(open('model/stance_based_model.sav', 'rb'))

        return load__content_model, load__stance_model
