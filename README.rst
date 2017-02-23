Magnetron2
==========

Updated version of magnetron using DNest 4. 

Magnetron is a code to model one-dimensional data (e.g. time series) 
with a superposition of simple shapes. The number of components in the 
superposition is part of the model and need not be known beforehand.

Note that this version of magnetron2 uses a **Poisson Likelihood**, since it 
was originally written for X-ray data of magnetar bursts. For a version that 
works on data with normally distributed uncertainties (like optical or radio 
data of, say blazars), check out ongoing work in the `blazars` branch!

Dependencies
============

- DNest4
- python 3.5 
- numpy
- scipy

Installation
============

First, make sure DNest4 is installed and the `DNEST4_PATH` variable set correctly.
Also, please run `python setup.py install` in the python directory of DNest4.

In this case, installing magnetron2 should be as simple as typing `make` in the `/code/`
directory!

Running the Code
================

`magnetron2` unfortunately doesn't have a super convenient user interface. File names of 
data files to be run generally need to be changed directly in `main.cpp`. For automatization, 
you can use the `run_dnest.py` script, which automatically reads all files with a certain 
matching string in a given directory and will run magnetron2 on each file one after the other. 
Be aware that the script is a bit hacky and not the ideal version to do this, but this is 
what we have.

Documentation
=============

Coming soon!

Copyright
=========

All content © 2016 the authors. The code is distributed under the MIT license.

Pull requests are welcome! If you are interested in the further development of
this project, please `get in touch via the issues
<https://github.com/dhuppenkothen/magnetron2/issues>`_!

