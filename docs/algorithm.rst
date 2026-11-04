.. _algorithm:

#########
Algorithm
#########

This page summarizes the recursive tensor MP-PCA algorithm of `Olesen et al. (2022)`_
as implemented in ``mepca``, and lists the ways the Python code differs from the MATLAB
original.

.. _Olesen et al. (2022): https://doi.org/10.1002/mrm.29478

***************
Matrix MP-PCA
***************

For a matrix :math:`X` of size :math:`M \times N` with :math:`M \le N`, MP-PCA computes
the singular value decomposition :math:`X = U S V^T` and assumes that the smallest
singular values are due to noise with variance :math:`\sigma^2`.
For a candidate number of signal components :math:`P`, the noise variance is estimated
from the remaining singular values,

.. math::

   \hat{\sigma}^2(P) = \frac{\sum_{i > P} s_i^2}{(M - P)(N - P)},

and the upper edge of the Marchenko-Pastur distribution is
:math:`\hat{\sigma}^2(P) (\sqrt{M} + \sqrt{N})^2`.
The number of signal components is the smallest :math:`P` for which the
:math:`(P + 1)`-th squared singular value falls below that edge.
The components above the edge are kept and, if optimal shrinkage is enabled, their
squared singular values are shrunk with the Frobenius-norm optimal shrinker of
Gavish and Donoho (2017),

.. math::

   \tilde{s}^2 = s^2 - 2 (M + N) \sigma^2 + \frac{(N - M)^2 \sigma^4}{s^2}.

****************
Recursive tensor
****************

A tensor with :math:`K` indices is denoised one index at a time:

1. Unfold the tensor along its first index into a matrix whose rows are indexed by the
   first index and whose columns are indexed by all other indices (in column-major
   order).
2. Compute the SVD and discard the noise components with the MP cutoff.
3. Form :math:`V S` from the retained right singular vectors.
   Its rows are indexed by the remaining indices and its columns by the retained
   components, so it is itself a tensor with the first index moved to the end.
4. Unfold :math:`V S` along the next index and repeat from step 2.
5. After the last index, apply optimal shrinkage to the retained components, then
   reconstruct the denoised tensor by multiplying back with the stored left singular
   vectors in reverse order.

By default the noise variance is estimated jointly from the singular values of every
unfolding of the original tensor (``full_sigma2_pass=True``).
The combined estimate sums the noise singular values of all unfoldings and divides by the
sum of their :math:`(M - P)(N - P)` degrees of freedom.
If any index ends up with zero signal components, the recursion stops and the patch is
set to zero (or to its mean, if ``subtract_mean`` is enabled).

**************
Sliding window
**************

:func:`mepca.denoise_recursive_tensor` moves a window over the first axes of the data.
Every position where the window fits (and, if a mask is given, contains at least one
in-mask voxel) is denoised as a tensor whose structure is given by ``indices``.
Each voxel receives the average of the denoised values from every window that contains
it, unless ``center_assign`` is set, in which case only the center voxel of each window
is assigned.
Voxels that no window covers keep their original values and get NaN in the output maps.

The SNR gain is estimated from the number of parameters of the retained model:

.. math::

   \text{SNR gain} = \sqrt{\frac{\prod_k n_k}{\prod_k P_k + \sum_k (n_k - P_k) P_k}},

where :math:`n_k` is the size and :math:`P_k` the number of signal components of index
:math:`k`.
For matrix patches, :math:`\prod_k P_k` becomes :math:`P^2`.

*****************************
Differences from MATLAB
*****************************

The Python code follows the MATLAB code line by line and uses column-major (Fortran
order) reshapes so that every patch is unfolded exactly as in MATLAB.
The deliberate differences are:

- ``indices`` are zero-based.
  MATLAB's ``{1:3 4 5}`` becomes ``[[0, 1, 2], [3], [4]]``.
- The MATLAB docstring describes a ``sigma`` option, but the code only reads
  ``sigma2``.
  The Python functions expose ``sigma2`` (the noise variance).
- Voxels never covered by a window get NaN in every column of the component map.
  MATLAB's logical indexing only sets the first column.
- Output maps have the shape of the voxel axes as given.
  MATLAB pads a single voxel axis to a column vector, and its bounds check skips every
  window position for a one-element window; the Python code handles that case.
- When no noise components are found in any unfolding (only possible for all-zero
  data), the noise variance is set to zero instead of NaN.
- The denoised array keeps the exact shape of the input, including trailing singleton
  axes that MATLAB's ``size`` would drop.
  The values are unaffected, because singleton axes are removed from each patch before
  it is denoised.
- The default for ``opt_shrink`` is ``True``, as in the MATLAB code, even though the
  MATLAB docstring says ``false``.
- Optimal shrinkage is applied at the last iteration of the recursion only.
  As in MATLAB, this means it is not applied when ``num_inds`` is 1 for a tensor input.
