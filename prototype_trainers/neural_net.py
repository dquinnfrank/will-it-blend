# Runs a neural net to do per pixel classification

import numpy as np

import cPickle as pickle

import sys
import os

# Add the path to post_process
sys.path.insert(0, "../data_generation")

import post_process as pp
im_p = pp.Image_processing
