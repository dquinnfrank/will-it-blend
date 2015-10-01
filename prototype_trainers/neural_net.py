# Runs a neural net to do per pixel classification

import numpy as np

import cPickle as pickle

import sys
import os

# Add the path to post_process
sys.path.insert(0, "../data_generation")

import post_process as pp
im_p = pp.Image_processing

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

class Neural_net:

	def __init__(self):

		pass

	def train_batch(self):

		pass

	def train(self):

		pass

	def save_model(self):

		pass

	def predict(self):

		pass

if __name__ == "__main__":

	pass
