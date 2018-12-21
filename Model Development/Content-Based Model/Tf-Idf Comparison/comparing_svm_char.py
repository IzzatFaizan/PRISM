import mysql.connector
import pandas as pd
from sklearn import model_selection, preprocessing, metrics, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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

# using pipeline create linearSVC model
linear_svc_pipeline_ngram_char = Pipeline([
    ('svmCV', tfidf_vect_ngram_chars),
    ('svm_clf', CalibratedClassifierCV(base_estimator=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                                                                intercept_scaling=1, loss='squared_hinge',
                                                                max_iter=1000,
                                                                multi_class='ovr', penalty='l2', random_state=None,
                                                                tol=0.0001, verbose=0), cv=5))
])

req = linear_svc_pipeline_ngram_char.fit(train_x, train_y)
predicted_svm = linear_svc_pipeline_ngram_char.predict(valid_x)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Classification Report for LinearSVC using Char Level Tf Idf Vectorizer No Kernel\n")
print(classification_report(valid_y, predicted_svm))
print("Train Accuracy ", metrics.accuracy_score(predicted_svm, valid_y))
print("Test Accuracy ", req.score(train_x, train_y))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

# using pipeline create SVC model with linear kernel
svm_pipeline_ngram_char = Pipeline([
    ('svmCV', tfidf_vect_ngram_chars),
    ('svm_clf', CalibratedClassifierCV(base_estimator=svm.SVC(C=1.0, kernel='linear', class_weight=None, max_iter=1000,
                                                              random_state=None,
                                                              tol=0.0001, verbose=0), cv=5))
])

req = svm_pipeline_ngram_char.fit(train_x, train_y)
predicted_svm = svm_pipeline_ngram_char.predict(valid_x)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Classification Report for SVC using Char Level Tf Idf Vectorizer Linear Kernel\n")
print(classification_report(valid_y, predicted_svm))
print("Train Accuracy ", metrics.accuracy_score(predicted_svm, valid_y))
print("Test Accuracy ", req.score(train_x, train_y))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


# using pipeline create NuSVC model with linear kernel
nusvc_pipeline_ngram_char = Pipeline([
    ('svmCV', tfidf_vect_ngram_chars),
    ('svm_clf', CalibratedClassifierCV(base_estimator=svm.NuSVC(kernel='linear', class_weight=None, max_iter=1000,
                                                                random_state=None,
                                                                tol=0.0001, verbose=0), cv=5))
])

req = nusvc_pipeline_ngram_char.fit(train_x, train_y)
predicted_svm = nusvc_pipeline_ngram_char.predict(valid_x)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Classification Report for NuSVC using Char Level Tf Idf Vectorizer Linear Kernel\n")
print(classification_report(valid_y, predicted_svm))
print("Train Accuracy ", metrics.accuracy_score(predicted_svm, valid_y))
print("Test Accuracy ", req.score(train_x, train_y))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")



