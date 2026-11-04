# mepca

Tensor MP-PCA denoising of multi-echo fMRI data.

[![License](https://img.shields.io/badge/License-LGPL%202.1-blue.svg)](https://opensource.org/licenses/LGPL-2.1)
[![Tests](https://github.com/SPiN-Lab/ME-PCA/actions/workflows/tests.yml/badge.svg)](https://github.com/SPiN-Lab/ME-PCA/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/me-pca/badge/?version=latest)](https://me-pca.readthedocs.io/en/latest/?badge=latest)

`mepca` is a Python translation of the MATLAB
[Tensor-MP-PCA](https://github.com/sunenj/Tensor-MP-PCA) toolbox, which implements
the method described in:

> Olesen, J. L., Ianus, A., Østergaard, L., Shemesh, N., & Jespersen, S. N. (2022).
> Tensor denoising of multidimensional MRI data. *Magnetic Resonance in Medicine*.
> https://doi.org/10.1002/mrm.29478

Usage is free, but please cite the paper above.

The package treats multi-echo fMRI data as a tensor with space, time, and echo as its
indices, and denoises it with a sliding window over the spatial axes. Each patch is
unfolded along one index at a time, noise components are identified with the
Marchenko-Pastur (MP) distribution and discarded, and the retained components are
unfolded along the next index, so that noise is removed from every index of the tensor.

## Installation

```bash
pip install git+https://github.com/SPiN-Lab/ME-PCA.git
```

For development, clone the repository and install it in editable mode with the test
dependencies:

```bash
pip install -e ".[tests]"
```

## Usage

### Command line

The `mepca` command denoises a set of single-echo 4D NIfTI files jointly:

```bash
mepca sub-01_echo-1_bold.nii.gz sub-01_echo-2_bold.nii.gz sub-01_echo-3_bold.nii.gz \
    --output-dir denoised --window 4 4 --mask sub-01_desc-brain_mask.nii.gz
```

This writes one denoised file per echo (`sub-01_echo-1_desc-tmppca_bold.nii.gz` and so
on) plus maps of the estimated noise variance, the number of signal components, and the
estimated SNR gain.

With a two-element window (the default), each patch spans all slices, so the window
slides over x and y only, as in the original MATLAB script. A three-element window
slides over all three spatial axes. Run `mepca --help` for the full list of options.

### Python

```python
import numpy as np
from mepca import denoise_recursive_tensor

# data: (x, y, z, time, echo)
denoised, sigma2, n_components, snr_gain = denoise_recursive_tensor(
    data,
    window=(4, 4),
    indices=[[0, 1, 2], [3], [4]],  # space, time, echo
    mask=mask.any(axis=2),  # one entry per (x, y) column for a 2D window
)
```

`indices` describes the tensor structure of each patch as groups of zero-based axes.
The equivalent of MATLAB's `indices = {1:3 4 5}` is `[[0, 1, 2], [3], [4]]`.

A single array can be denoised directly with `denoise_array_recursive_tensor`:

```python
from mepca import denoise_array_recursive_tensor

denoised, sigma2, n_components = denoise_array_recursive_tensor(patch)
```

## Differences from the MATLAB implementation

The Python code follows the MATLAB code line by line and uses column-major reshapes
so that every patch is unfolded exactly as in MATLAB. The deliberate differences are:

- `indices` are zero-based.
- The MATLAB docstring describes a `sigma` option, but the code only reads `sigma2`.
  The Python functions expose `sigma2` (the noise variance).
- Voxels never covered by a window get NaN in every column of the component map, not
  only in the first column.
- Output maps have the shape of the voxel axes as given, so a one-element window on
  2D data yields 1D maps instead of MATLAB's padded column vectors, and a one-element
  window works (MATLAB's bounds check skips every window position in that case).
- When no noise components are found in any unfolding (only possible for all-zero
  data), the noise variance is set to zero instead of NaN.
- The denoised array keeps the exact shape of the input, including trailing singleton
  axes that MATLAB's `size` would drop. The values are unaffected.

The default for `opt_shrink` is `True`, as in the MATLAB code (its docstring says
`false`).

## License

`mepca` is released under the GNU Lesser General Public License, version 2.1.
