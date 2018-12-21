import nltk
import numpy as np
import pandas as pd
import mysql.connector
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

# database connection
conn = mysql.connector.connect(host='127.0.0.1', user='root', password='', database='news')
cursor = conn.cursor()
cursor.execute("SELECT * from news")
content = cursor.fetchall()

# load the dataset
labels, texts = [], []

for row in content:
    texts.append(row[2])
    texts.append(row[3])

for i in range(239):
    labels.append('Fake')
    labels.append('Real')

# create a dataframe using data and label
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# we will start with simple bag of words technique
# creating feature vector - document term matrix
countV = CountVectorizer()
train_count = countV.fit_transform(trainDF['text'].values)

print(countV)
print(train_count)


# print training doc term matrix
# we have matrix of size of (10240, 12196) by calling below
def get_countVectorizer_stats():
    # vocab size
    var = train_count.shape
    print(var)

    # check vocabulary using below command
    print(countV.vocabulary_)

    # get feature names
    print(countV.get_feature_names()[:25])


# create tf-df frequency features
# tf-idf
tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(train_count)


def get_tfidf_stats():
    var = train_tfidf.shape
    # get train data feature names
    print(var.A[:10])


tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), use_idf=True, smooth_idf=True)
tagged_sentences = nltk.corpus.treebank.tagged_sents()
cutoff = int(.75 * len(tagged_sentences))
training_sentences = trainDF['text']

print(training_sentences)


# building Linear SVM classfier
svm_pipeline = Pipeline([
    ('svmCV', countV),
    ('svm_clf', svm.LinearSVC())
])

svm_pipeline.fit(trainDF['text'], trainDF['Label'])
predicted_svm = svm_pipeline.predict(trainDF['text'])
np.mean(predicted_svm == trainDF['Label'])


# User defined function for K-Fold cross validatoin
def build_confusion_matrix(classifier):
    k_fold = KFold(n=len(trainDF), n_folds=5)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])

    for train_ind, test_ind in k_fold:
        train_text = trainDF.iloc[train_ind]['text']
        train_y = trainDF.iloc[train_ind]['Label']

        test_text = trainDF.iloc[test_ind]['text']
        test_y = trainDF.iloc[test_ind]['Label']

        classifier.fit(train_text, train_y)
        predictions = classifier.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions)
        scores.append(score)

    return (print('Total statements classified:', len(trainDF)),
            print('Score:', sum(scores) / len(scores)),
            print('score length', len(scores)),
            print('Confusion matrix:'),
            print(confusion))


# K-fold cross validation for all classifiers
build_confusion_matrix(svm)
