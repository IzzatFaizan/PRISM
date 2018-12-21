import itertools
import pickle
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, metrics, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
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

# using pipeline
svm_pipeline = Pipeline([
    ('svmCV', tfidf_vect_ngram_chars),
    ('svm_clf', CalibratedClassifierCV(base_estimator=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                                                                intercept_scaling=1, loss='squared_hinge',
                                                                max_iter=1000,
                                                                multi_class='ovr', penalty='l2', random_state=None,
                                                                tol=0.0001, verbose=0), cv=5))
])

req = svm_pipeline.fit(train_x, train_y)
predicted_svm = svm_pipeline.predict(valid_x)
print(classification_report(valid_y, predicted_svm))
print("Train Accuracy ", metrics.accuracy_score(predicted_svm, valid_y))
print("Test Accuracy ", req.score(train_x, train_y))

final_model = "svm_linearSVC.sav"
pickle.dump(svm_pipeline, open(final_model, 'wb'))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


predicted_svm1 = svm_pipeline.predict(train_x)
print('training', confusion_matrix(train_y, predicted_svm1))

predicted_svm = svm_pipeline.predict(valid_x)
print('test', confusion_matrix(valid_y, predicted_svm))

# Compute confusion matrix
cnf_matrix = confusion_matrix(train_y, predicted_svm1)
np.set_printoptions(precision=2)
l = ['Berita Palsu', 'Berita Benar']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=l,
                      title='Confusion matrix Train, without normalization')

# Compute confusion matrix
cnf_matrix1 = confusion_matrix(valid_y, predicted_svm)
np.set_printoptions(precision=2)
l = ['Berita Palsu', 'Berita Benar']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix1, classes=l,
                      title='Confusion matrix Test, without normalization')

plt.show()
