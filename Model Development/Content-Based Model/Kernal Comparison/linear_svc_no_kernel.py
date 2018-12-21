import mysql.connector
import pandas as pd
from sklearn import model_selection, preprocessing, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

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

# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], shuffle=False,
                                                                      test_size=0.25)

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                         max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

# SVM on Character Level TF IDF Vectors
# svm_model = svm.SVC(kernel='linear', C=1.0, gamma='auto')
linear_model = svm.LinearSVC(C=1.0)


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    model_svm = classifier.fit(feature_vector_train, label)
    model_svm.score(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    print(classification_report(valid_y, predictions))

    return metrics.accuracy_score(predictions, valid_y), model_svm.score(feature_vector_train, label)


# Print Accuracy
train_accuracy, test_accuracy = train_model(linear_model, xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("Train Accuracy : ", train_accuracy)
print("Test Accuracy : ", test_accuracy)