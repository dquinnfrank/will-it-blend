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

Setup:
Before any images can be generated, blender must be downloaded and configured by the user. There are also some additional steps to properly configure the image generation.

This project uses Blender version 2.72, which can be downloaded from http://download.blender.org/release/Blender2.72/ . Select the correct version for the system being used. This is the only version that is compatible with the project right now. Thus using a package manager such as apt-get is not recommended.

Some settings within Blender must be configured by the user. This only needs to be done once.
From the top bar go to File > User Preferences
Then go to the File tab and check the box: "Auto Run Python Scripts"
Next go to the Addons tab
Enable the Rigify addon by searching for rigify in the search bar and checking the box next to Rigging:Rigify
Enable mhx importing by searching for mhx in the search bar and checking the box next to Import-Export: Import: MakeHuman (.mhx)
Finally, save the changes by clicking Save User Settings. Exit blender.

The image generation code requires some configuration. This can be done by running setup.py or manually.
The items that need to be taken care of are:
A file named path.txt must be in the folder data_generation. This file must contain the path to the blender executable. It will look like: [path_to_blender_folder]/[blender_version]/blender
setup.py can do this by running it with the argument -p, such as python setup.py -p [path_to_blender_executable] .
Any custom modules imported by generate_image.py must be placed into [blender_version]/2.72/scripts/modules . This is due to the fact that blender runs its own python interpreter.
setup.py can do this when the -c option is used, such as: python setup.py -c . This will copy all files from the folder data_generation/blender_scripts to the modules folder.



Other Information:
The human model used by this project was generated with MakeHuman, an open source tool that can be found at http://www.makehuman.org/ . It is not necessary to install this software to run the data generator, but is needed to create or modify a human model for the data set. Much of the code is based around the characteristics of the MakeHuman model, using any other human model would require significant modifications.
