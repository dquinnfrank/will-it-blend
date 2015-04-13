# Trains a neural network to find human poses

import os
import sys

# Need to import the post_processing module from data_generation
sys.path.insert(0, os.path.join('..', 'data_generation'))
import post_process as pp

# Keras is the framework for theano based neural nets
import keras
