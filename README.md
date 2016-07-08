#will-it-blend

##Fast Detection of Partially Occluded Humans from Mobile Platforms

University of Nevada, Reno

David Frank, Dr. David Feil-Seifer, Dr. Richard Kelley

This repository contains files necessary to train a system to detect partially occluded humans. Additional software is required as stated below. It is meant to run on Ubuntu 14.04

##Setup

Some actions muust be done manually by the user before installing.

An ssh key must be associated with the git hub account. Follow the instructions here to generate a new key if needed and add it to the github account: https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/#platform-linux

CUDA must be installed. Follow the instructions here to do this: http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu

Installing CUDA can sometimes lead to various video driver related errors, so perform the always prudent step of backing up anything important.

Before any images can be generated, Blender must be downloaded and configured by the user. There are also some additional steps to properly configure the image generation.

This project uses Blender version 2.72, which can be downloaded from http://download.blender.org/release/Blender2.72/ . Select the correct version for the system being used. This is the only version that is compatible with the project right now. Thus using a package manager such as apt-get is not recommended.


Some settings within Blender must be configured by the user. This only needs to be done once.
* From the top bar go to File > User Preferences
* Then go to the File tab and check the box: "Auto Run Python Scripts"
* Next go to the Addons tab
* Enable the Rigify addon by searching for rigify in the search bar and checking the box next to Rigging:Rigify
* Enable mhx importing by searching for mhx in the search bar and checking the box next to Import-Export: Import: MakeHuman (.mhx)
* Finally, save the changes by clicking Save User Settings. Exit blender.

Once this is complete, to do all other setup items run: 

```
python setup.py -u -a [path_to_blender]
```

##Generating and Processing Data

###Generation

Data generation can take a few days based on data set size.

To generate default datasets, navigate to `data_generation` and run:

```
python generate_driver.py [config_file]
```

Configuration files must be placed into the `data_generation/configs` folder. Check examples for how to make these.

Custom datasets can be made by using the functions in `scene_manager` module, source in `data_generation/blender_scripts`. Check `generate_image.py` for usage examples.

###Processing

The raw images must be processed before training can be done. This can also take a few days.

To process data, navigate to `data_generation` and run:

```
python post_process.py source_dir target_name [-s start_index -e end_index -c scale_factor -b batch_size]
```

##Other Information
The human model used by this project was generated with MakeHuman, an open source tool that can be found at http://www.makehuman.org/ . It is not necessary to install this software to run the data generator, but is needed to create or modify a human model for the data set. Much of the code is based around the characteristics of the MakeHuman model, using any other human model would require significant modifications.

##Legal Notices

###OpenEXR

Copyright (c) 2002-2011, Industrial Light & Magic, a division of Lucasfilm Entertainment Company Ltd. All rights reserved. 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:


Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of Industrial Light & Magic nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


###MakeHuman

The human models created with this program are licensed under CC0, so they are free to use. The MakeHuman source code is licensed under AGPL3, but it is not being distributed with this project.

###Torch7

Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Deepmind Technologies, NYU, NEC Laboratories America 
   and IDIAP Research Institute nor the names of its contributors may be 
   used to endorse or promote products derived from this software without 
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
