##################################################
mepca: tensor MP-PCA denoising of multi-echo fMRI
##################################################

``mepca`` is a Python translation of the MATLAB Tensor-MP-PCA toolbox, which implements
the method described in `Olesen et al. (2022)`_.
Usage is free, but please cite the paper.

.. _Olesen et al. (2022): https://doi.org/10.1002/mrm.29478

The package treats multi-echo fMRI data as a tensor with space, time, and echo as its
indices and denoises it with a sliding window over the spatial axes.
Each patch is unfolded along one index at a time, noise components are identified with
the Marchenko-Pastur distribution and discarded, and the retained components are unfolded
along the next index, so that noise is removed from every index of the tensor.

************
Installation
************

.. code-block:: bash

   pip install git+https://github.com/SPiN-Lab/ME-PCA.git

*******************
License Information
*******************

``mepca`` is licensed under the GNU Lesser General Public License, version 2.1.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   algorithm
   api

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
