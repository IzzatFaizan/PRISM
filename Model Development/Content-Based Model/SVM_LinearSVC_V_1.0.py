from random import shuffle

import mysql.connector
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
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
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.9)

train_x = np.array(train_x)
train_y = np.array(train_y)
valid_x = np.array(valid_x)
valid_y = np.array(valid_y)

train_y = train_y.reshape(90, 1)
valid_y = valid_y.reshape(10, 1)

train_f1 = train_x[:, 0]
train_f2 = train_x[:, 1]

train_f1 = train_f1.reshape(90, 1)
train_f2 = train_f2.reshape(90, 1)

w1 = np.zeros((90, 1))
w2 = np.zeros((90, 1))

epochs = 1
alpha = 0.0001

while epochs < 10000:
    y = w1 * train_f1 + w2 * train_f2
    prod = y * train_y
    print(epochs)
    count = 0
    for val in prod:
        if val >= 1:
            cost = 0
            w1 = w1 - alpha * (2 * 1 / epochs * w1)
            w2 = w2 - alpha * (2 * 1 / epochs * w2)

        else:
            cost = 1 - val
            w1 = w1 + alpha * (train_f1[count] * train_y[count] - 2 * 1 / epochs * w1)
            w2 = w2 + alpha * (train_f2[count] * train_y[count] - 2 * 1 / epochs * w2)
        count += 1
    epochs += 1

# Clip the weights
index = list(range(10, 90))
w1 = np.delete(w1, index)
w2 = np.delete(w2, index)

w1 = w1.reshape(10, 1)
w2 = w2.reshape(10, 1)
# Extract the test data features
test_f1 = valid_x[:, 0]
test_f2 = valid_x[:, 1]

test_f1 = test_f1.reshape(10, 1)
test_f2 = test_f2.reshape(10, 1)

# Predict
y_pred = w1 * test_f1 + w2 * test_f2
predictions = []
for val in y_pred:
    if val > 1:
        predictions.append(1)
    else:
        predictions.append(-1)

print(accuracy_score(valid_y, predictions))

clf = LinearSVC()
clf.fit(train_x, train_y)
y_pred = clf.predict(valid_x)
print(accuracy_score(valid_y, y_pred))