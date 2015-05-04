# Trains a neural network to find human poses
# This is a simple version that has llimited functionality

import os
import sys
import cPickle as pickle
import numpy as np
from guppy import hpy

# Start the memory monitor
hp = hpy()

# Need to import the post_processing module from data_generation
sys.path.insert(0, os.path.join('..', 'data_generation'))
import post_process as pp

# Keras is the framework for theano based neural nets
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad

# Get all of the input data
input_data = pickle.load(open("/media/master/DAVID_DRIVE/occulsion/set_001_small_all/data/0_4999.p", 'rb'))

# Get all of the labels
label_data = pickle.load(open("/media/master/DAVID_DRIVE/occulsion/set_001_small_all/label/0_4999.p", 'rb'))

print "Input data shape:"
print input_data.shape
print "Label data shape:"
print label_data.shape

# Add the stack dimension, needed for correct processing in the convolutional layers
input_data = np.expand_dims(input_data, axis=0)
#label_data = np.expand_dims(label_data, axis=0)

# Reshape to (n_images, stack, height, width)
input_data = np.rollaxis(input_data, 0, 2)

# Reshape to (n_images, height * width)
#label_data = np.rollaxis(label_data, 0, 2)
label_data = label_data.reshape(5000, 48*64)

print "Input data shape:"
print input_data.shape
print "Label data shape:"
print label_data.shape

# Separate into training and testing
train_data = input_data[:4500]
test_data = input_data[4500:]

train_label = label_data[:4500]
test_label = label_data[4500:]

# Make into GPU friendly float32
train_data = train_data.astype("float32")
test_data = test_data.astype("float32")

print "Training data shape:"
print train_data.shape
print "Training label shape:"
print train_label.shape
print "Testing data shape:"
print test_data.shape
print "Testing label shape:"
print test_label.shape

# Show the current memory usage
#print hp.heap()

# Network configuration
conv_inter = 6

image_height = 48
image_width = 64

# The network
model = Sequential()

model.add(Convolution2D(conv_inter, 1, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
# Show the current memory usage
#print hp.heap()
model.add(Convolution2D(conv_inter, conv_inter, 3, 3))
model.add(Activation('relu'))
# Show the current memory usage
#print hp.heap()
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
# Show the current memory usage
#print hp.heap()

model.add(Convolution2D(conv_inter*2, conv_inter, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
# Show the current memory usage
#print hp.heap()
model.add(Convolution2D(conv_inter*2, conv_inter*2, 3, 3))
model.add(Activation('relu'))
# Show the current memory usage
#print hp.heap()
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
# Show the current memory usage
#print hp.heap()

model.add(Flatten())
model.add(Dense((conv_inter * image_height * image_width)/8, 512, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Show the current memory usage
print (hp.heap())
model.add(Dense(512, image_height * image_width, init='normal'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)

model.fit(train_data, train_label, batch_size=32, nb_epoch=10)
score = model.evaluate(test_data, test_label, batch_size=32)

print('Test score:', score)
