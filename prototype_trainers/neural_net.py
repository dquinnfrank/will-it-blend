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

	# Creates the model
	#
	# load_name is the name of a model to be loaded
	# WARNING: not currently impemented
	def __init__(self, load_name=None, hidden_nodes=500):

		# If load name is sent, load the model
		# TODO: make this work
		#if load_name:
		if False:

			pass

		# Create a new network
		else:

			# Create the model container
			self.classifier = Sequential()

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
