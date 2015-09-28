# Runs a random forest classifier on depth difference data to train a per pixel system

import numpy as np

import cPickle as pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# The percentage to be used for training, the remaining data will be used for testing
train_split = .9

# Get the data
data = pickle.load(open("/media/master/CORSAIR/depth_features/data/set_001.p", 'rb'))

# Get the labels
labels = pickle.load(open("/media/master/CORSAIR/depth_features/label/set_001.p", 'rb'))

# Split into training and testing
train_data = data[int(train_split * data.shape[0]):]
train_labels = labels[int(train_split * labels.shape[0]):]

test_data = data[:int(train_split * data.shape[0])]
test_labels = labels[:int(train_split * labels.shape[0])]

# Create the random forest with parameters similar to the microsoft paper
clf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)

# Train the forest
clf.fit(train_data, train_labels)

# Get the score
clf_predict = clf.predict(test_data)
score = accuracy_score(test_labels, clf_predict)

# Show score and other info
print "Random forest"
print "Number of training points: ", data.shape[0]
print "Accuracy: ", score

# Load an example image

