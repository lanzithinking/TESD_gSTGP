# TESD_gSTGP

## Learning Temporal Evolution of Spatial Dependence with Generalized Spatiotemporal Gaussian Process Models
### Shiwei Lan @ lanzithinking@outlook.com

Spatiotemporal processes are ubiquitous in our life and have been a trending topic in the scientific community, e.g. the dynamic brain connectivity study in neuroscience.
There is usually complicated dependence among spatial locations and such relationship does not necessarily stay static over time.
Spatiotemporal Gaussian process (STGP) is a popular nonparametric method to model this type of data. 
However, the classic STGP has a covariance kernel with space and time separated, failed to characterize the temporal evolution of spatial dependence (TESD). 
Even for some recent work on non-separable STGP, location and time are treated with no difference, which is unnecessarily inefficient. 
This paper generalizes STGP by introducing the time-dependence to the spatial kernel and varying its eigenvalues over time. 
A novel STGP model with the covariance kernel having a Kronecker sum structure is proposed and proved to be superior to the popular kernel with a Kronecker product structure.
A simulation study on the spatiotemporal process and a longitudinal neuroimaging analysis of Alzheimer's patients demonstrate the advantage of the proposed methodology in effectively and efficiently characterizing TESD.

See the paper on [arXiv](https://arxiv.org/abs/1901.04030).

## Directories:

* AD 
contains codes for the longitudinal neuroimaging analysis of Alzheimer's patients.
Click [here](https://drive.google.com/open?id=1IyEIdQTtmzCt4mVlR0IhRsR7WOA1SO5p) for the post-processed data.

* sampler
contains codes of MCMC samplers.

* simulation
contains codes for the simulation study of a non-separable spatiotemporal process.

* util
contains several folders of utilities. +STGP*/ are folders for the classes of generalized STGP.

Copyright &copy;2019