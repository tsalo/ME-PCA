.. _usage:

#####
Usage
#####

************
Command line
************

The ``mepca`` command denoises a set of single-echo 4D NIfTI files jointly.
The files must share the same shape and are given in echo order:

.. code-block:: bash

   mepca sub-01_echo-1_bold.nii.gz sub-01_echo-2_bold.nii.gz sub-01_echo-3_bold.nii.gz \
       --output-dir denoised --window 4 4 --mask sub-01_desc-brain_mask.nii.gz

The echoes are stacked into a five-dimensional array (x, y, z, time, echo).
All spatial axes form the first tensor index, time the second, and echo the third,
which reproduces the MATLAB ``run_TensorMPPCA.m`` script.

Outputs are written to the output directory:

- one denoised file per echo, with ``desc-tmppca`` inserted before the suffix
  (``sub-01_echo-1_desc-tmppca_bold.nii.gz``);
- ``*_desc-tmppca_sigma2.nii.gz``: the estimated noise variance;
- ``*_desc-tmppca_ncomponents.nii.gz``: the number of signal components retained for
  each tensor index (space, time, echo), stored along the last axis;
- ``*_desc-tmppca_snrgain.nii.gz``: the estimated gain in signal-to-noise ratio.

The maps have the shape of the window axes.
With a two-element window (the default), each patch spans all slices, so the window
slides over x and y only and the maps are 2D.
With a three-element window the window slides over all three spatial axes and the maps
are 3D.

Options:

``--window N [N ...]``
   Sliding window size (default ``4 4``).
``--mask FILE``
   3D brain mask.
   Window positions that contain no in-mask voxel are skipped.
   For a two-element window the mask is reduced to the x-y columns that contain any
   in-mask voxel.
``--sigma2 FLOAT``
   Known noise variance.
   If omitted, it is estimated per patch with the Marchenko-Pastur distribution.
``--no-opt-shrink``
   Disable optimal singular value shrinkage.
``--subtract-mean``
   Subtract the mean of each unfolding before the SVD.
``--center-assign``
   Assign the denoised value to the center voxel of each window only, instead of
   averaging all overlapping windows.
``--stride N [N ...]``
   Window step along each window axis (default 1).
``--desc``
   Value of the ``desc`` entity in the output filenames (default ``tmppca``).

******
Python
******

:func:`mepca.denoise_recursive_tensor` slides a window over the first axes of an array
and denoises each patch as a tensor:

.. code-block:: python

   import numpy as np
   from mepca import denoise_recursive_tensor

   # data: (x, y, z, time, echo)
   denoised, sigma2, n_components, snr_gain = denoise_recursive_tensor(
       data,
       window=(4, 4),
       indices=[[0, 1, 2], [3], [4]],  # space, time, echo
       mask=mask.any(axis=2),  # one entry per (x, y) column for a 2D window
   )

``indices`` describes the tensor structure of each patch as groups of zero-based axes of
the patch array, which has the window axes first, followed by the remaining axes of
``data``.
Axes in the same group are combined into one tensor index, and the groups are denoised
in the order given.
The equivalent of MATLAB's ``indices = {1:3 4 5}`` is ``[[0, 1, 2], [3], [4]]``.
By default the window axes form one index and every other axis is its own index.

:func:`mepca.denoise_array_recursive_tensor` denoises a single array:

.. code-block:: python

   from mepca import denoise_array_recursive_tensor

   denoised, sigma2, n_components = denoise_array_recursive_tensor(patch)

A 2D array is denoised with ordinary matrix MP-PCA.
Arrays with three or more dimensions are denoised recursively, one index at a time.
