will-it-blend
=============

Fast Detection of Partially Occluded Humans from Mobile Platforms

University of Nevada, Reno

David Frank, Dr. David Feil-Seifer, Dr. Richard Kelley

This repository contains files necessary to train a system to detect partially occluded humans. Additional software is required as stated below.

There are two main segements of this project: the data generation and the training. The data generation segment is contained in the data_generation folder and the training segment is contained in train_cnn.

============
Data Generation

This section will generate the training data for the model.

To Run:

To easily generate large data sets, generate_driver.py is used. This is a wrapper for generate_image.py that loads configurations from a file. Run generate_image.py with the following command:

python generate_driver.py config_name

config_name : The name of the configuration file to use


All configuration files must be contained in data_generation/configs

All of the files must have the following form, see defaut.txt for an example:

SAVE_PATH save_dir

GENERATE_NUMBER num_images

DEBUG_OUTPUT caps_bool

OFFSET start_at


save_dir : string : The directory to save all images to

num_images : int : The total number of images to generate

caps_bool : TRUE or FALSE : Sets debug output, TRUE will output information during the generation process

OFFSET : int : The number to start indexing at


The images will all be OpenEXR images, to use them for training there are more processing steps that are needed. To process the images use post_process.py .

Right now, post_process.py is best run in the interpreter. Navigate to the data_generation folder and start python.

Then enter: import post_process as pp

Now set the varible scale to the desired scale factor, to make half sized images: pp.scale = .5

Run the process_to_pickle function.

This will be changed in the future to be easier to use.


Setup:

Before any images can be generated, blender must be downloaded and configured by the user. There are also some additional steps to properly configure the image generation.


This project uses Blender version 2.72, which can be downloaded from http://download.blender.org/release/Blender2.72/ . Select the correct version for the system being used. This is the only version that is compatible with the project right now. Thus using a package manager such as apt-get is not recommended.


Some settings within Blender must be configured by the user. This only needs to be done once.
* From the top bar go to File > User Preferences
* Then go to the File tab and check the box: "Auto Run Python Scripts"
* Next go to the Addons tab
* Enable the Rigify addon by searching for rigify in the search bar and checking the box next to Rigging:Rigify
* Enable mhx importing by searching for mhx in the search bar and checking the box next to Import-Export: Import: MakeHuman (.mhx)
* Finally, save the changes by clicking Save User Settings. Exit blender.

The image generation code requires some configuration. Some of the items can be done by running setup.py or manually.
To run all setup items run: python setup.py -a [path_to_blender]


The items that need to be taken care of are:
* A file named path.txt must be in the folder data_generation. This file must contain the path to the blender executable. It will look like: [path_to_blender_folder]/[blender_version]/blender
setup.py can do this by running it with the argument -p, such as python setup.py -p [path_to_blender_executable] .
* Any custom modules imported by generate_image.py must be placed into [blender_version]/2.72/scripts/modules . This is due to the fact that blender runs its own python interpreter.
setup.py can do this when the -c option is used, such as: python setup.py -c . This will copy all files from the folder data_generation/blender_scripts to the modules folder.
* The OpenEXR bindings for python must be installed. The OpenEXR C++ Library is a prerequisite for the bindings and must be installed on the system before running, on Debian-based Linux this can be done by running sudo apt-get install libopenexr-dev. The Python bindings can be installed by running pip install openexr . The bindings and documentation can be found at: http://www.excamera.com/sphinx/articles-openexr.html . Running setup.py with the -e option will show instructions for Debian-based Linux, other system configurations will need to check the documentation.
* TODO: add keras additional layer

============
Train CNN

This section will train a neural net to detect the pose of a person

To Run:

Training the net requires that large amounts of data be generated using the data generation module. This data must be processed into pickles having the following form:

depth data: 

shape (n_images, height, width)

n_images can vary and does not need to be consistent between pickles

height and width cannot change

The data must be normalized depth data at each pixel

target labels:

shape (n_images, height * width)

n_images must be the same as the corresponding depth data

height and width cannot change

Each pixel must be labeled 0 - 12 with each number corresponding to a different class

0 - Not a person

1 - Head, left

2 - Head, right

3 - Torso, left

4 - Torso, right

5 - Upper arm, left

6 - Upper arm, right

7 - Lower arm, left

8 - Lower arm, right

9 - Upper leg, left

10 - Upper leg, right

11 - Lower leg, left

12 - Lower leg, right


To train a net, train_pickle.py is used. This will train networks and can be saved and loaded as pickles. In the future, a more portable format should be used. Run it with the following command:

python train_pickle.py train_source_dir test_source_dir [-l load_name -s save_name]

train_source_dir : the location of the training data

test_source_dir : the location of the testing data

load_name : the name to load the model from

save_name : the name to save the model to


To see the output predictions of a net, predict_image.py is used. This will take depth images and generate images that can be viewed, in order to understand how the net is doing. Run it with the following command:

python predict_image.py load_model_name save_image_dir image_pickle

load_model_name : the name, including path, of the model to use for predictions

save_image_dir : the name of the directory to save image predictions to, will be created if it doesn't exist

image_pickle : the name, including path, of the pickle of images to use


simple_train.py will be removed soon.

train_net.py is not currently functional. In the future, it will become the main method of training nets, replacing train_pickle.py .

Setup:

This project uses a Theano based deep learning library. Thus Theano must be setup; it is reccomended that a GPU be used. Setup changes by system, here are resources for setting it up on Ubuntu:

Basic Theano: http://deeplearning.net/software/theano/install_ubuntu.html

GPU with Theano: http://deeplearning.net/software/theano/tutorial/using_gpu.html

Optional cuDNN for faster CNNs: https://developer.nvidia.com/cudnn

For constructing networks, the Theano wrapper Keras is used.

Information on setup and use: https://github.com/fchollet/keras


============

Other Information:
The human model used by this project was generated with MakeHuman, an open source tool that can be found at http://www.makehuman.org/ . It is not necessary to install this software to run the data generator, but is needed to create or modify a human model for the data set. Much of the code is based around the characteristics of the MakeHuman model, using any other human model would require significant modifications.

Legal Notices:

OpenEXR:

Copyright (c) 2002-2011, Industrial Light & Magic, a division of Lucasfilm Entertainment Company Ltd. All rights reserved. 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:


Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of Industrial Light & Magic nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


MakeHuman:

The human models created with this program are licensed under CC0, so they are free to use. The MakeHuman source code is licensed under AGPL3, but it is not being distributed with this project. It may be in the future, once I figure out what the AGPL3 license would mean for the project.
