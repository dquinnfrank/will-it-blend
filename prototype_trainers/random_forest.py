# Runs a random forest classifier on depth difference data to train a per pixel system

import numpy as np

import cPickle as pickle

# Add the path to post_process
sys.path.insert(0, "../data_generation")

import post_process as pp
im_p = pp.Image_processing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# The percentage to be used for training, the remaining data will be used for testing
train_split = .9

# Get the data
data = pickle.load(open("/media/CORSAIR/depth_features/data/set_001.p", 'rb'))

# Get the labels
labels = pickle.load(open("/media/CORSAIR/depth_features/label/set_001.p", 'rb'))

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
#ex_image = pickle.load(open("/media/CORSAIR/ex_images/ex1.py", 'rb'))

# Predict all of the pixels
#ex_prediction = clf.predict(test_data)

# Reorder the axis for saving
#ex_prediction = np.expand_dims(ex_prediction, axis=0)
#ex_prediction = np.rollaxis(ex_prediction, 2, 1)

# Save the image
#im_p.save_image(ex_prediction, "/media/CORSAIR/ex_images/ex1_prediction.jpg")
