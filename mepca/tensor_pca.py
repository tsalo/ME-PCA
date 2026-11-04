"""Tensor MP-PCA denoising of multidimensional (tensor-structured) data.

This module is a Python translation of the MATLAB Tensor-MP-PCA toolbox that
accompanies Olesen et al. (2022). The MATLAB code is column-major and relies on
linear indexing, so every reshape in this module uses Fortran (column-major)
order to keep the index bookkeeping identical to the original.

References
----------
Olesen, J. L., Ianus, A., Østergaard, L., Shemesh, N., & Jespersen, S. N. (2022).
Tensor denoising of multidimensional MRI data. Magnetic Resonance in Medicine.
https://doi.org/10.1002/mrm.29478

Gavish, M., & Donoho, D. L. (2017). Optimal shrinkage of singular values.
IEEE Transactions on Information Theory, 63(4), 2137-2152.
https://doi.org/10.1109/TIT.2017.2653801
"""

import numpy as np

__all__ = [
    "denoise_array_recursive_tensor",
    "denoise_recursive_tensor",
    "opt_shrink_frob",
]


def _matlab_shape(shape):
    """Return a shape the way MATLAB's ``size`` would report it.

    Trailing singleton dimensions are dropped, but at least two dimensions are kept.
    """
    shape = [int(s) for s in shape]
    while len(shape) > 2 and shape[-1] == 1:
        shape.pop()

    while len(shape) < 2:
        shape.append(1)

    return tuple(shape)


def opt_shrink_frob(vals2, n_rows, n_cols, sigma2):
    """Apply optimal (Frobenius-norm) shrinkage to squared singular values.

    Parameters
    ----------
    vals2 : array_like
        Squared singular values of the signal components.
    n_rows, n_cols : int
        Dimensions of the matrix the singular values belong to.
    sigma2 : float
        Noise variance.

    Returns
    -------
    numpy.ndarray
        Shrunk squared singular values. Values can be negative; callers clip them at zero.

    Notes
    -----
    This is the closed form from Gavish and Donoho (2017) for the Frobenius loss,
    written in unnormalized units as in the MATLAB implementation.
    """
    vals2 = np.asarray(vals2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return vals2 - 2 * (n_cols + n_rows) * sigma2 + (n_cols - n_rows) ** 2 * sigma2**2 / vals2


def _estimate_noise(s, dims):
    """Estimate the noise variance from singular values with the MP distribution.

    Parameters
    ----------
    s : numpy.ndarray
        Singular values (1D, descending) of one unfolding of the data.
    dims : tuple of int
        Original dimensions of the data array.

    Returns
    -------
    sigma2 : float
        Estimated noise variance.
    n_components : int
        Estimated number of signal components.
    """
    m = s.size
    # Use the original dimensions of the array, as in the MATLAB code.
    n = np.prod(dims) / m
    vals2 = s**2
    # All possible values for the number of signal components.
    p = np.arange(m)
    # Noise variance estimate as a function of the number of signal components.
    sigma2_estimates = np.cumsum(vals2[::-1])[::-1] / (m - p) / (n - p)
    # Upper edge of the MP distribution as a function of the number of signal components.
    cutoff_estimates = sigma2_estimates * (np.sqrt(m) + np.sqrt(n)) ** 2
    # Every singular value below the cutoff is a noise component.
    below = np.flatnonzero(vals2 < cutoff_estimates)
    if below.size == 0:
        # Special case: no noise components found.
        n_components = m
        sigma2 = 0.0
    else:
        n_components = int(below[0])
        sigma2 = float(sigma2_estimates[n_components])

    if n_components == 0 and min(m, n) == 1:
        n_components = 1

    return sigma2, n_components


def _combined_noise_estimate(s_list, dims, n_components):
    """Combine the noise estimates of all unfoldings into a single estimate.

    Parameters
    ----------
    s_list : list of numpy.ndarray
        Singular values of each unfolding.
    dims : tuple of int
        Original dimensions of the data array.
    n_components : list of int
        Initial estimate of the number of signal components for each unfolding.

    Returns
    -------
    sigma2 : float
        Combined noise variance estimate.
    n_components : list of int
        Number of signal components of each unfolding under the combined estimate.
    """
    numerator = 0.0
    denominator = 0.0
    for s, p in zip(s_list, n_components):
        m = s.size
        n = np.prod(dims) / m
        vals2 = s**2
        numerator += vals2[p:].sum()
        denominator += (m - p) * (n - p)

    if denominator == 0:
        # No noise components were found in any unfolding. MATLAB would produce NaN here
        # and discard every component; treating the data as noise-free is more useful.
        sigma2 = 0.0
    else:
        sigma2 = float(numerator / denominator)

    new_components = []
    for s in s_list:
        m = s.size
        n = np.prod(dims) / m
        cutoff = sigma2 * (np.sqrt(m) + np.sqrt(n)) ** 2
        new_components.append(int(np.count_nonzero(s**2 > cutoff)))

    return sigma2, new_components


def _discard_noise_components(u, s, vh, sigma2):
    """Keep only the singular components above the MP cutoff."""
    m = u.shape[0]
    n = vh.shape[1]
    cutoff = sigma2 * (np.sqrt(m) + np.sqrt(n)) ** 2
    p = int(np.count_nonzero(s**2 > cutoff))
    return u[:, :p], s[:p], vh[:p, :], p


def _apply_optimal_shrinkage(u, s, vh, sigma2):
    """Shrink the retained singular values."""
    if s.size == 0:
        # No signal components were found.
        return s

    m = u.shape[0]
    n = vh.shape[1]
    p = s.size
    vals2 = opt_shrink_frob(s**2, max(m - p, 1), max(n - p, 1), sigma2)
    # MATLAB takes real(sqrt(vals2)), which is zero for negative values.
    return np.sqrt(np.maximum(vals2, 0))


def _subtract_mean(x):
    """Subtract the mean along the longer dimension of a matrix."""
    m, n = x.shape
    if m < n:
        x_mean = x.mean(axis=1, keepdims=True)
    else:
        x_mean = x.mean(axis=0, keepdims=True)

    return x - x_mean, x_mean


def _svd(x):
    """Economy-size SVD, returning ``u``, singular values, and ``vh``."""
    return np.linalg.svd(x, full_matrices=False)


def _unfold_and_svd(x, dim, subtract_mean):
    """Flatten ``x`` along its first index (``dim`` rows) and decompose it."""
    x = x.reshape((dim, -1), order="F")
    if subtract_mean:
        x, x_mean = _subtract_mean(x)
    else:
        x_mean = 0

    u, s, vh = _svd(x)
    return u, s, vh, x_mean


def denoise_array_recursive_tensor(
    x,
    num_inds=None,
    opt_shrink=True,
    subtract_mean=False,
    sigma2=None,
    full_sigma2_pass=True,
):
    """Denoise a single array with recursive (tensor) MP-PCA.

    Parameters
    ----------
    x : array_like
        Array to be denoised. A 2D array is denoised with ordinary matrix MP-PCA.
        Arrays with three or more dimensions are denoised recursively: the array is
        unfolded along its first index, noise components are discarded, and the retained
        right singular vectors (scaled by their singular values) are unfolded along the
        next index, and so on.
    num_inds : int, optional
        Number of indices to iterate through. Defaults to the number of dimensions of
        ``x``. A smaller value leaves the trailing indices undenoised.
    opt_shrink : bool, optional
        Apply optimal singular value shrinkage to the components retained at the last
        iteration. Default is True.
    subtract_mean : bool, optional
        Subtract the mean (along the longer dimension of each unfolding) before the SVD
        and add it back afterward. Default is False.
    sigma2 : float, optional
        Known noise variance. If not provided, it is estimated with the Marchenko-Pastur
        distribution.
    full_sigma2_pass : bool, optional
        If True (default), estimate the noise variance from the singular values of every
        unfolding. If False, estimate it from the first unfolding only.

    Returns
    -------
    denoised : numpy.ndarray
        Denoised array, with the same shape as ``x`` (trailing singleton dimensions are
        dropped, as MATLAB's ``size`` would).
    sigma2 : float
        Noise variance used (estimated or provided).
    n_components : numpy.ndarray of int
        Number of signal components retained at each index. For 2D input it has a
        single element. For tensors it has ``num_inds`` elements, zero-padded if the
        recursion stopped early.

    Notes
    -----
    This is a translation of ``denoise_array_recursive_tensor.m``. Every reshape uses
    column-major (Fortran) order so that the unfoldings match the MATLAB code.

    Deviations from the MATLAB code are limited to degenerate cases:

    * When no noise components are found in any unfolding, MATLAB produces a NaN noise
      variance and discards every component. This implementation uses a noise
      variance of zero instead, which keeps all components.
    """
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.inexact):
        x = x.astype(float)

    dims = _matlab_shape(x.shape)
    x = x.reshape(dims, order="F")
    if num_inds is None:
        num_inds = len(dims)

    num_inds = int(num_inds)
    if not 1 <= num_inds <= len(dims):
        raise ValueError(
            f"num_inds must be between 1 and the number of dimensions ({len(dims)}), "
            f"got {num_inds}"
        )

    use_mppca = sigma2 is None

    # Special case: matrix input.
    if len(dims) == 2:
        if min(dims) == 1:
            # Special case: nothing to denoise.
            return x, 0.0, np.array([1])

        if subtract_mean:
            x, x_mean = _subtract_mean(x)
        else:
            x_mean = 0

        u, s, vh = _svd(x)
        if use_mppca:
            sigma2, _ = _estimate_noise(s, dims)

        u, s, vh, p = _discard_noise_components(u, s, vh, sigma2)
        if opt_shrink:
            s = _apply_optimal_shrinkage(u, s, vh, sigma2)

        x = (u * s) @ vh + x_mean
        return x, float(sigma2), np.array([p])

    u_list, s_list, vh_list, mean_list, p_list = [], [], [], [], []

    # Estimate sigma2 from the first SVD, or make a full HOSVD pass to get all singular
    # values for a combined sigma2 estimate.
    if use_mppca:
        num_svds = num_inds if full_sigma2_pass else 1
        for n in range(num_svds):
            u, s, vh, x_mean = _unfold_and_svd(x, dims[n], subtract_mean)
            u_list.append(u)
            s_list.append(s)
            vh_list.append(vh)
            mean_list.append(x_mean)
            # Initial individual estimates of the number of signal components.
            p_list.append(_estimate_noise(s, dims)[1])
            # Prepare x for the next iteration: make the current first index the last
            # index without changing the ordering of the other indices (123 -> 231).
            x = vh.conj().T * s

        sigma2, p_list = _combined_noise_estimate(s_list, dims, p_list)
    else:
        u, s, vh, x_mean = _unfold_and_svd(x, dims[0], subtract_mean)
        u_list.append(u)
        s_list.append(s)
        vh_list.append(vh)
        mean_list.append(x_mean)
        p_list.append(0)

    # Recursive SVD, reusing the calculations for the first index.
    u, s, vh, p = _discard_noise_components(u_list[0], s_list[0], vh_list[0], sigma2)
    u_list[0], s_list[0], vh_list[0], p_list[0] = u, s, vh, p
    x = vh.conj().T * s
    n_levels = 1

    # Continue for the remaining indices.
    for n in range(1, num_inds):
        if p_list[n - 1] == 0:
            break

        u, s, vh, x_mean = _unfold_and_svd(x, dims[n], subtract_mean)
        u, s, vh, p = _discard_noise_components(u, s, vh, sigma2)
        if opt_shrink and n == num_inds - 1:
            # Apply optimal shrinkage at the last iteration.
            s = _apply_optimal_shrinkage(u, s, vh, sigma2)

        if n < len(u_list):
            u_list[n], s_list[n], vh_list[n], mean_list[n], p_list[n] = u, s, vh, x_mean, p
        else:
            u_list.append(u)
            s_list.append(s)
            vh_list.append(vh)
            mean_list.append(x_mean)
            p_list.append(p)

        n_levels = n + 1
        x = vh.conj().T * s

    # Reconstruct the denoised array with a backward pass.
    # Initially x = V * S from the last iteration.
    for n in reversed(range(n_levels)):
        if p_list[n] == 0:
            x = np.zeros((u_list[n].shape[0], x.shape[0]), dtype=x.dtype) + mean_list[n]
        else:
            x = u_list[n] @ x.reshape((-1, p_list[n]), order="F").conj().T + mean_list[n]

    x = x.reshape(dims, order="F")
    n_components = np.array(p_list[:n_levels] + [0] * (num_inds - n_levels))
    return x, float(sigma2), n_components


def denoise_recursive_tensor(
    data,
    window,
    indices=None,
    mask=None,
    center_assign=False,
    stride=None,
    opt_shrink=True,
    subtract_mean=False,
    sigma2=None,
    full_sigma2_pass=True,
):
    """Denoise multidimensional data with a sliding window and tensor MP-PCA.

    Usage is free, but please cite Olesen et al. (2022), doi:10.1002/mrm.29478.

    Parameters
    ----------
    data : array_like
        Noisy data. The first ``len(window)`` axes are the spatial (voxel) axes the
        window slides over. The remaining axes are measurement axes (for example time
        and echo).
    window : sequence of int
        Size of the sliding window along each voxel axis. ``window = (5, 5)`` processes
        patches of size 5 x 5 x (all remaining axes).
    indices : sequence of sequences of int, optional
        Tensor structure of each patch, as groups of zero-based axes of the patch array.
        The patch array has the window axes first, followed by the remaining axes of
        ``data``. Axes in the same group are combined into one tensor index. For data
        with five axes, where the first three are voxels and the window has two
        elements, ``indices=[[0, 1, 2], [3], [4]]`` combines the window axes and the
        third voxel axis into one index and denoises each patch as a three-index
        tensor, whereas ``indices=[[0, 1], [2], [3], [4]]`` denoises each patch as a
        four-index tensor. Defaults to combining the window axes into one index and
        keeping every other axis as its own index. The order of the groups determines
        the order in which the indices are denoised.
    mask : array_like of bool, optional
        Logical mask with the shape of the voxel axes. Window positions that contain no
        True voxel are skipped. Default is to process every window position.
    center_assign : bool, optional
        If True, only the center voxel of each window receives the denoised value, and
        window positions whose center voxel is outside the mask are skipped. Default is
        False, where every voxel in the window receives the denoised value and
        overlapping windows are averaged.
    stride : int or sequence of int, optional
        Step of the sliding window along each voxel axis. Default is 1.
    opt_shrink : bool, optional
        Apply optimal singular value shrinkage. Default is True.
    subtract_mean : bool, optional
        Subtract the mean of each unfolding before the SVD. Default is False.
    sigma2 : float, optional
        Known noise variance. If not provided, it is estimated with the Marchenko-Pastur
        distribution for each patch.
    full_sigma2_pass : bool, optional
        Estimate the noise variance from every unfolding of each patch (default) rather
        than from the first unfolding only.

    Returns
    -------
    denoised : numpy.ndarray
        Denoised data with the same shape as ``data``. Voxels that no window covered
        keep their original values.
    sigma2 : numpy.ndarray
        Estimated noise variance, with the shape of the voxel axes. NaN for voxels that
        no window covered.
    n_components : numpy.ndarray
        Estimated number of signal components per tensor index, with the shape of the
        voxel axes plus one trailing axis. NaN for voxels that no window covered.
    snr_gain : numpy.ndarray
        Estimated gain in signal-to-noise ratio, with the shape of the voxel axes.

    Notes
    -----
    This is a translation of ``denoise_recursive_tensor.m``. The window is moved over
    every position where it fits inside the data, and each voxel receives the average
    of the denoised values from all windows that contain it.

    Deviations from the MATLAB code:

    * ``indices`` are zero-based (MATLAB's ``{1:3 4 5}`` becomes ``[[0, 1, 2], [3], [4]]``).
    * The MATLAB docstring documents a ``sigma`` option, but the code only reads
      ``sigma2``; this implementation exposes ``sigma2`` (the noise variance).
    * MATLAB only marks the first column of ``n_components`` as NaN for skipped voxels;
      here every column is NaN.
    * Output maps have the shape of the voxel axes as given, rather than MATLAB's
      padded ``size`` (a one-dimensional window yields a 1D map instead of a column).
    * A one-element window works on data with a single voxel axis; in MATLAB the bounds
      check skips every window position in that case.
    * ``denoised`` keeps the exact shape of ``data``, including trailing singleton axes
      that MATLAB's ``size`` would drop. The values are unaffected, because singleton
      axes are removed from each patch before it is denoised.
    """
    data = np.asarray(data)
    if not np.issubdtype(data.dtype, np.inexact):
        data = data.astype(float)

    dims = data.shape
    ndim = len(dims)

    window = np.atleast_1d(np.asarray(window)).astype(int).ravel()
    n_win = window.size
    if n_win == 0 or n_win > ndim:
        raise ValueError(
            f"window must have between 1 and {ndim} elements for data with {ndim} dimensions"
        )

    if np.any(window < 1):
        raise ValueError("window sizes must be positive")

    vox_shape = tuple(dims[:n_win])
    if np.any(window > vox_shape):
        raise ValueError(f"window {tuple(window)} does not fit in the voxel axes {vox_shape}")

    # MATLAB's size always has at least two elements.
    dims_vox = _matlab_shape(vox_shape)
    num_vox = int(np.prod(dims_vox))
    # Reshape data to have all voxels along the first index and all measurements along
    # the second.
    data2d = data.reshape((num_vox, -1), order="F")

    # Default index ordering: same order as the input data, with the window axes
    # combined into one index.
    if indices is None:
        indices = [list(range(n_win))] + [[k] for k in range(n_win, ndim)]

    indices = [np.atleast_1d(np.asarray(group, dtype=int)).ravel() for group in indices]
    if len(indices) == 0 or any(group.size == 0 for group in indices):
        raise ValueError("indices must contain at least one non-empty group of axes")

    all_axes = np.concatenate(indices)
    if np.any(all_axes < 0):
        raise ValueError("indices must be non-negative, zero-based axes")

    if np.unique(all_axes).size != all_axes.size:
        raise ValueError("indices must not list the same axis twice")

    if mask is None:
        mask = np.ones(vox_shape, dtype=bool)
    else:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != vox_shape:
            raise ValueError(
                f"mask dimensions {mask.shape} do not match the voxel dimensions {vox_shape}"
            )

    mask_flat = mask.reshape(-1, order="F")

    if stride is None:
        stride = np.ones(n_win, dtype=int)
    else:
        stride = np.atleast_1d(np.asarray(stride)).astype(int).ravel()
        if stride.size == 1:
            stride = np.repeat(stride, n_win)

        if stride.size != n_win:
            raise ValueError("stride must have one element per window dimension")

        if np.any(stride < 1):
            raise ValueError("stride must be positive")

    # Dimensions of the patch array: the window followed by the measurement axes, padded
    # with singleton axes if indices refer to axes beyond the data.
    array_size = list(window) + list(dims[n_win:])
    array_size += [1] * max(0, int(all_axes.max()) + 1 - len(array_size))
    array_size = np.array(array_size)

    # Linear (column-major) offsets of the voxels within the window relative to the
    # corner voxel.
    n_window_vox = int(np.prod(window))
    window_subs = list(np.unravel_index(np.arange(n_window_vox), tuple(window), order="F"))
    window_subs += [np.zeros(n_window_vox, dtype=int)] * (len(dims_vox) - n_win)
    index_increments = np.ravel_multi_index(window_subs, dims_vox, order="F")

    # Permutation order in accordance with the provided indices.
    permute_order = np.concatenate([all_axes, np.setdiff1d(np.arange(array_size.size), all_axes)])
    inverse_order = np.argsort(permute_order)

    # Size after reshaping in accordance with the provided indices.
    new_size = [int(np.prod(array_size[group])) for group in indices]
    new_size.append(int(np.prod(array_size)) // int(np.prod(new_size)))
    new_size = [size for size in new_size if size != 1]
    new_size.append(1)
    new_size = np.array(new_size)
    patch_shape = _matlab_shape(new_size)
    num_inds = min(len(patch_shape), len(indices))
    # Matrix patches yield a single component count; tensors yield one per index.
    n_p_cols = 1 if len(patch_shape) == 2 else num_inds
    n_tensor_inds = new_size.size - 1
    if n_p_cols != 1 and n_p_cols != n_tensor_inds:
        raise ValueError(
            "indices must cover every non-window axis of the data (or group the remaining "
            "axes explicitly) so that the SNR gain can be estimated"
        )

    # Window corner positions (zero-based) that fit in the data and match the stride.
    corners = np.array(np.unravel_index(np.arange(num_vox), dims_vox, order="F")).T
    window_full = np.concatenate([window, np.ones(len(dims_vox) - n_win, dtype=int)])
    stride_full = np.concatenate([stride, np.ones(len(dims_vox) - n_win, dtype=int)])
    in_bounds = np.all(corners + window_full <= np.array(dims_vox), axis=1)
    on_stride = np.all(corners % stride_full == 0, axis=1)
    positions = np.flatnonzero(in_bounds & on_stride)

    if center_assign:
        center_sub = tuple(np.ceil(window / 2).astype(int) - 1)
        center_ind = int(np.ravel_multi_index(center_sub, tuple(window), order="F"))

    # Pre-allocate.
    denoised = np.zeros(data2d.shape, dtype=data2d.dtype)
    count = np.zeros(num_vox)
    sigma2_map = np.zeros(num_vox)
    p_map = np.zeros((num_vox, n_p_cols))

    # Loop over window positions and denoise.
    for i in positions:
        # Indices of the voxels within the window.
        vox_indices = i + index_increments

        # Skip if no voxels in the mask are included.
        mask_x = mask_flat[vox_indices]
        if not mask_x.any():
            continue

        # Skip if the center voxel is not included.
        if center_assign and not mask_x[center_ind]:
            continue

        # Create the data array for this patch.
        x = data2d[vox_indices, :].reshape(array_size, order="F")
        x = np.transpose(x, permute_order)
        x = x.reshape(patch_shape, order="F")

        # Denoise the patch.
        x, patch_sigma2, patch_p = denoise_array_recursive_tensor(
            x,
            num_inds=num_inds,
            opt_shrink=opt_shrink,
            subtract_mean=subtract_mean,
            sigma2=sigma2,
            full_sigma2_pass=full_sigma2_pass,
        )

        x = x.reshape(array_size[permute_order], order="F")
        x = np.transpose(x, inverse_order)
        x = x.reshape((vox_indices.size, -1), order="F")

        # Assign.
        if center_assign:
            # Only use the center voxel.
            x = x[center_ind : center_ind + 1, :]
            vox_indices = vox_indices[center_ind : center_ind + 1]

        denoised[vox_indices, :] += x
        count[vox_indices] += 1
        sigma2_map[vox_indices] += patch_sigma2
        p_map[vox_indices, : patch_p.size] += patch_p

    # Assign original data to skipped voxels.
    skipped = count == 0
    denoised[skipped, :] = data2d[skipped, :]
    sigma2_map[skipped] = np.nan
    p_map[skipped, :] = np.nan

    # Divide by the number of times each voxel has been visited to get the average.
    count[skipped] = 1
    denoised /= count[:, np.newaxis]
    sigma2_map /= count
    p_map /= count[:, np.newaxis]

    # Estimate the SNR gain according to the removed variance.
    sizes = new_size[:-1].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if p_map.shape[1] == 1:
            denominator = p_map[:, 0] ** 2 + np.sum((sizes - p_map) * p_map, axis=1)
        else:
            denominator = np.prod(p_map, axis=1) + np.sum((sizes - p_map) * p_map, axis=1)

        snr_gain = np.sqrt(np.prod(new_size) / denominator)

    # Adjust the outputs to match the input dimensions.
    denoised = denoised.reshape(dims, order="F")
    sigma2_map = sigma2_map.reshape(vox_shape, order="F")
    p_map = p_map.reshape(vox_shape + (n_p_cols,), order="F")
    snr_gain = snr_gain.reshape(vox_shape, order="F")
    return denoised, sigma2_map, p_map, snr_gain
