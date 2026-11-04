"""mepca: tensor MP-PCA denoising of multi-echo fMRI data."""

from mepca.tensor_pca import (
    denoise_array_recursive_tensor,
    denoise_recursive_tensor,
    opt_shrink_frob,
)

try:
    from mepca._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "denoise_array_recursive_tensor",
    "denoise_recursive_tensor",
    "opt_shrink_frob",
]
