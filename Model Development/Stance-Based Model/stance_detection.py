import pickle
import mysql.connector
import numpy as np
import pandas
from sklearn import model_selection, metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# establish connection with database
conn = mysql.connector.connect(host='localhost', database='stance_sebenarnya', user='root', password='')

# select element in sebenarnya_news database
sebenarnya_news = pandas.read_sql('SELECT title, content  FROM stance_sebenarnya WHERE label = "tidak setuju" '
                                  'LIMIT 500', con=conn)
# select element in utusan_news database
utusan_news = pandas.read_sql('SELECT title, content  FROM stance_utusan WHERE label = "setuju" LIMIT 500', con=conn)

# merge sebenarnya_news and utusan_news into single dataframe alternately
dataDF = pandas.concat([sebenarnya_news, utusan_news]).sort_index(kind='merge')

# reset index bcoz of alternate merging process before
dataDF = dataDF.reset_index(drop=True)

label = []

for i in range(500):
    label.append('Fake')
    label.append('Real')

# add column label in dataDF
dataDF['label'] = pandas.DataFrame(label, index=dataDF.index)

concateDF = pandas.concat([dataDF['title'], dataDF['content']]).sort_index(kind='merge')

# split the dataset into training and validation datasets
train_title, valid_title, train_content, valid_content, train_y, valid_y = model_selection.train_test_split(
    dataDF['title'], dataDF['content'],
    dataDF['label'], test_size=0.25,
    shuffle=False)

# characters level tf-idf
tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                        max_features=5000)
tfidf_vect_ngram_char.fit(concateDF)

# save vocab in pickle file
# final_model = "vocab_ngram.pickle"
# pickle.dump(tfidf_vect_ngram_char, open(final_model, 'wb'))

train_title_tfidf_ngram_char = tfidf_vect_ngram_char.transform(train_title)
valid_title_tfidf_ngram_char = tfidf_vect_ngram_char.transform(valid_title)
train_content_tfidf_ngram_char = tfidf_vect_ngram_char.transform(train_content)
valid_content_tfidf_ngram_char = tfidf_vect_ngram_char.transform(valid_content)

train_stance = np.concatenate((train_title_tfidf_ngram_char.toarray(), train_content_tfidf_ngram_char.toarray()),
                              axis=1)
valid_stance = np.concatenate((valid_title_tfidf_ngram_char.toarray(), valid_content_tfidf_ngram_char.toarray()),
                              axis=1)

linear_model = CalibratedClassifierCV(base_estimator=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                                                               intercept_scaling=1, loss='squared_hinge',
                                                               max_iter=1000,
                                                               multi_class='ovr', penalty='l2', random_state=None,
                                                               tol=0.0001, verbose=0), cv=5)


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    model_svm = classifier.fit(feature_vector_train, label)
    model_svm.score(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    print(classification_report(valid_y, predictions))

    return metrics.accuracy_score(predictions, valid_y), model_svm.score(feature_vector_train, label)


# Print Accuracy
train_accuracy, test_accuracy = train_model(linear_model, train_stance, train_y, valid_stance)
print("Train Accuracy : ", train_accuracy)
print("Test Accuracy : ", test_accuracy)

final_model = "stance_char.sav"
pickle.dump(linear_model, open(final_model, 'wb'))
