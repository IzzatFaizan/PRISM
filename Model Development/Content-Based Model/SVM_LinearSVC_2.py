import pandas as pd
import numpy as np
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, metrics, svm
from sklearn.metrics import confusion_matrix, classification_report

# database connection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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

# convert to numpy array
xtrain_tfidf_ngram_chars = xtrain_tfidf_ngram_chars.toarray()
xvalid_tfidf_ngram_chars = xvalid_tfidf_ngram_chars.toarray()

# flip content from left to right
xtrain_tfidf_ngram_chars = np.flipud(xtrain_tfidf_ngram_chars)
xvalid_tfidf_ngram_chars = np.flipud(xvalid_tfidf_ngram_chars)

# SVM on Character Level TF IDF Vectors
# svm_model = svm.SVC(kernel='linear', C=1.0, gamma='auto')
linear_model = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                             intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                             multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                             verbose=0)


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    req = classifier.fit(feature_vector_train, label)
    req.score(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    predictions2 = classifier.predict(feature_vector_train)
    print("Train Con_Matrix : ", confusion_matrix(label, predictions2))
    print("Test Con_Matrix : ", confusion_matrix(valid_y, predictions))
    print(classification_report(valid_y, predictions))

    return metrics.accuracy_score(predictions, valid_y), req.score(feature_vector_train, label)


def grid_search_cv(feature_vector_train, label):
    param_grid = {'C': [0.001, 0.01, 0.1, 10, 50, 100]}
    clf = GridSearchCV(linear_model, param_grid)
    clf.fit(feature_vector_train, label)
    return clf.fit(feature_vector_train, label)


# Print Accuracy
train_accuracy, test_accuracy = train_model(linear_model, xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("Train Accuracy : ", train_accuracy)
print("Test Accuracy : ", test_accuracy)


# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(train_y)

# Plot data points and color using their class
color = ['black' if c == 0 else 'lightgrey' for c in y]
plt.scatter(X_std[:, 0], X_std[:, 1], c=color)

# Create the hyperplane
w = linear_model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (linear_model.intercept_[0]) / w[1]

# Plot the hyperplane
plt.plot(xx, yy)
plt.axis("off"), plt.show()