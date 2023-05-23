# mercergp
A library for low-rank Mercer Gaussian Processes.

This is a PyTorch based library containing classes for representation of special
functions and Mercer kernel Gaussian processes for GP regression and other models.

This library is provided as a companion to our "Favard Kernels" paper.

This repository has as an informal 
dependency the library `ortho`, a link to which is also provided in the paper.
To install, first install the `ortho` library. 

To install this library, clone the repository, and ensure that  the location
you install `mercergp` is in your PYTHONPATH environment variable. 

Run `pytest` in the root folder to run unit tests.

A superset of the dependencies and necessary versions can be found in 
dependencies.txt, which contains a dump of pipdeptree on this project.
