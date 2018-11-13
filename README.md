
Occiput - Suite for Tomographic Vision - version 0.5
===============================================================================

Tomographic reconstruction software for PET, PET-MRI and SPECT in 2D, 3D (volumetric) and 4D (spatio-temporal) in Python.

The software provides high-speed reconstruction using Graphics Processing Units (GPU). *Note*: an NVidia CUDA-compatible GPU is required.

***Occiput*** can be utilized with arbitrary scanner geometries. It can be utilized for abstract tomographic reconstruction experiments to develop new algorithms and explore new system geometries, or to connect to real-world scanners, providing production quality image reconstruction with standard algorithms (such as MLEM and OSEM).

***Occiput*** implements advanced algorithms for motion correction, kinetic imaging, multi-modal reconstruction, respiratory and cardiac gated imaging.

The source code contains Jupyter notebooks with examples.

A Python extension package ***Occiput_Interface_Biograph_mMR***, implementing the interface to the Siemens Biograph mMR PET-MRI scanner is available upon request and following authorization from Siemens. Notebooks containing Biograph_mMR in the title can only be executed after installing the extension package.

Please email us at occiput.reconstruction@gmail.com


Installation
============

Linux, Windows (not tested recently), MacOS
-------------------------------------------

Pre-requisites: Occiput requires *NVidia GPU Drivers*, *NVidia CUDA* and the *NiftyRec* GPU accelerated tomographic ray-tracing library.

1. [Install NVidia GPU Drivers and CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install NiftyRec libraries](https://github.com/mscipio/NiftyRec): build the latest version using CMake

3. Make sure that CUDA libraries and NiftyRec libraries are in the system path:

 - **Linux**:

    `export LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/path_to_cuda_libraries:/path_to_niftyrec_libraries`

 - **MacOS**:

    `export DYLD_LIBRARY_PATH:$DYLD_LIBRARY_PATH:/path_to_cuda_libraries:/path_to_niftyrec_libraries`

 - **Windows**:

    `setx path "%path%;c:/path_to_cuda_libraries:/path_to_niftyrec_libraries;"`

4. Install ``Occiput Suite`` from the source code available in this repo:

 -    `git clone https://github.com/mscipio/occiput-suite.git`

 -    `python setup.py build install`


Getting started
===============
Examples and demos of the features of Occiput are in the /occiput/notebooks folder.
To get started, install ***Python Jupyter*** and open the scripts in
[/occiput_suite/notebooks.](https://github.com/mscipio/occiput-suite/tree/master/occiput_suite/notebooks>)

The notebook [DOCUMENTATION.ipynb](https://github.com/mscipio/occiput-suite/blob/master/occiput_suite/notebooks/DOCUMENTATION.ipynb) contains an index and short description of the notebooks.


Website
=======
For more information see [occiput.io](http://occiput.mgh.harvard.edu/)


Changelog
=========
For more information see [CHANGES.md](https://github.com/mscipio/occiput-suite/blob/master/CHANGES.md)

