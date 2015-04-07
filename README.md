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
To run all setup items run: python setup.py -a [path_to_blender]

The items that need to be taken care of are:
A file named path.txt must be in the folder data_generation. This file must contain the path to the blender executable. It will look like: [path_to_blender_folder]/[blender_version]/blender
setup.py can do this by running it with the argument -p, such as python setup.py -p [path_to_blender_executable] .
Any custom modules imported by generate_image.py must be placed into [blender_version]/2.72/scripts/modules . This is due to the fact that blender runs its own python interpreter.
setup.py can do this when the -c option is used, such as: python setup.py -c . This will copy all files from the folder data_generation/blender_scripts to the modules folder.
The OpenEXR bindings for python must be installed. This can be done manually by navigating to the OpenEXR-1.2.0 folder and running that module's setup.py. The root setup.py for this project can do it by running it with the -e option, such as: python setup.py -e .


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

Lua Struct:

/******************************************************************************
* Copyright (C) 2010-2012 Lua.org, PUC-Rio.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

MakeHuman:

The human models created with this program are licensed under CC0, so they are free to use. The MakeHuman source code is licensed under AGPL3, but it is not being distributed with this project. It may be in the future, once I figure out what the AGPL3 license would mean for the project.
