"""Denoising workflows for multi-echo fMRI NIfTI files."""

import os
import re

import nibabel as nb
import numpy as np

from mepca.tensor_pca import denoise_recursive_tensor

#: Tensor structure used for multi-echo data: all spatial axes combined into one index,
#: time as the second index, and echo as the third index.
MULTI_ECHO_INDICES = [[0, 1, 2], [3], [4]]


def _split_extension(filename):
    """Split a NIfTI filename into its stem and extension."""
    basename = os.path.basename(filename)
    for ext in (".nii.gz", ".nii"):
        if basename.endswith(ext):
            return basename[: -len(ext)], ext

    stem, ext = os.path.splitext(basename)
    return stem, ext


def output_filename(filename, output_dir, desc, suffix=None):
    """Build an output filename with a ``desc`` entity inserted before the suffix.

    Parameters
    ----------
    filename : str
        Input filename, for example ``sub-01_echo-1_bold.nii.gz``.
    output_dir : str
        Directory the output file goes in.
    desc : str
        Value of the ``desc`` entity to insert, for example ``tmppca``.
    suffix : str, optional
        Replacement for the suffix of the input filename (the part after the last
        underscore). If not provided, the input suffix is kept.

    Returns
    -------
    str
        Output path, for example ``<output_dir>/sub-01_echo-1_desc-tmppca_bold.nii.gz``.
    """
    stem, ext = _split_extension(filename)
    if ext not in (".nii", ".nii.gz"):
        ext = ".nii.gz"

    if "_" in stem:
        entities, in_suffix = stem.rsplit("_", 1)
    else:
        entities, in_suffix = stem, None

    if suffix is None:
        suffix = in_suffix

    parts = [entities, f"desc-{desc}"]
    if suffix:
        parts.append(suffix)

    return os.path.join(output_dir, "_".join(parts) + ext)


def denoise_multi_echo(
    echo_files,
    output_dir,
    window=(4, 4),
    mask_file=None,
    opt_shrink=True,
    subtract_mean=False,
    sigma2=None,
    center_assign=False,
    stride=None,
    desc="tmppca",
):
    """Denoise a set of single-echo 4D NIfTI files jointly with tensor MP-PCA.

    The echoes are stacked into a five-dimensional array (x, y, z, time, echo) and
    denoised with :func:`mepca.denoise_recursive_tensor`, treating all spatial axes as
    one tensor index, time as the second, and echo as the third. This reproduces the
    MATLAB ``run_TensorMPPCA.m`` script.

    Parameters
    ----------
    echo_files : sequence of str
        Paths to the 4D magnitude NIfTI files, one per echo, in echo order.
    output_dir : str
        Directory to write the denoised files and maps to. It is created if needed.
    window : sequence of int, optional
        Sliding window size over the first ``len(window)`` spatial axes. With the default
        of two elements, each patch spans all slices, so the window slides over x and y
        only. Use three elements to slide over all three spatial axes.
    mask_file : str, optional
        Path to a 3D brain mask. Window positions that contain no in-mask voxel are
        skipped. For a two-element window, the mask is reduced to the x-y columns that
        contain any in-mask voxel.
    opt_shrink, subtract_mean, sigma2, center_assign, stride
        Passed to :func:`mepca.denoise_recursive_tensor`.
    desc : str, optional
        Value of the ``desc`` entity in the output filenames. Default is ``tmppca``.

    Returns
    -------
    dict
        Paths of the written files: ``"echoes"`` (list of denoised echo files),
        ``"sigma2"``, ``"n_components"``, and ``"snr_gain"``.
    """
    echo_files = list(echo_files)
    if len(echo_files) < 2:
        raise ValueError("at least two echo files are required for multi-echo denoising")

    window = tuple(int(w) for w in np.atleast_1d(window))
    if len(window) not in (2, 3):
        raise ValueError("window must have two or three elements for 4D echo files")

    imgs = [nb.load(f) for f in echo_files]
    ref_shape = imgs[0].shape
    if len(ref_shape) != 4:
        raise ValueError(f"echo files must be 4D, got shape {ref_shape} for {echo_files[0]}")

    ref_affine = imgs[0].affine
    for f, img in zip(echo_files, imgs):
        if img.shape != ref_shape:
            raise ValueError(
                f"all echo files must have the same shape: {f} has {img.shape}, "
                f"expected {ref_shape}"
            )

        if not np.allclose(img.affine, ref_affine, atol=1e-4):
            raise ValueError(
                f"all echo files must be on the same grid: the affine of {f} differs from "
                f"that of {echo_files[0]}"
            )

    # MATLAB: abs(single(niftiread(...)))
    data = np.stack([np.abs(img.get_fdata(dtype=np.float32)) for img in imgs], axis=-1)

    mask = None
    if mask_file is not None:
        mask = np.asarray(nb.load(mask_file).dataobj) > 0
        if mask.shape != ref_shape[:3]:
            raise ValueError(
                f"mask shape {mask.shape} does not match the spatial shape {ref_shape[:3]}"
            )

        if len(window) == 2:
            mask = mask.any(axis=2)

    denoised, sigma2_map, p_map, snr_gain = denoise_recursive_tensor(
        data,
        window,
        indices=MULTI_ECHO_INDICES,
        mask=mask,
        center_assign=center_assign,
        stride=stride,
        opt_shrink=opt_shrink,
        subtract_mean=subtract_mean,
        sigma2=sigma2,
    )

    os.makedirs(output_dir, exist_ok=True)
    outputs = {"echoes": []}
    for i_echo, (f, img) in enumerate(zip(echo_files, imgs)):
        out_file = output_filename(f, output_dir, desc)
        out_img = nb.Nifti1Image(denoised[..., i_echo].astype(np.float32), img.affine, img.header)
        out_img.set_data_dtype(np.float32)
        out_img.to_filename(out_file)
        outputs["echoes"].append(out_file)

    # Maps are named after the first echo, with the echo entity removed.
    map_source = re.sub(r"_echo-[^_]+", "", os.path.basename(echo_files[0]))
    ref = imgs[0]
    for name, array in (
        ("sigma2", sigma2_map),
        ("n_components", p_map),
        ("snr_gain", snr_gain),
    ):
        out_file = output_filename(map_source, output_dir, desc, suffix=name.replace("_", ""))
        map_img = nb.Nifti1Image(array.astype(np.float32), ref.affine)
        map_img.to_filename(out_file)
        outputs[name] = out_file

    return outputs
