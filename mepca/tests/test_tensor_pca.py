"""Tests for the tensor MP-PCA implementation.

MATLAB is not available in the test environment, so these tests check the mathematical
properties the MATLAB code has (noise estimation, rank recovery, exact reconstruction of
low-rank tensors) and the consistency between the sliding-window function and direct
patch denoising, which exercises the column-major index bookkeeping.
"""

import numpy as np
import pytest

from mepca import tensor_pca
from mepca.tensor_pca import (
    denoise_array_recursive_tensor,
    denoise_recursive_tensor,
    opt_shrink_frob,
)


def _low_rank_matrix(rng, shape, rank, scale=1.0):
    """Random matrix with the given rank."""
    left = rng.standard_normal((shape[0], rank))
    right = rng.standard_normal((rank, shape[1]))
    return scale * (left @ right)


def _low_rank_tensor(rng, shape, ranks, scale=1.0):
    """Random tensor in Tucker form with the given multilinear ranks."""
    x = rng.standard_normal(ranks)
    for n, r in zip(shape, ranks):
        factor = rng.standard_normal((n, r))
        # Contract the leading axis and move the result to the end, so that after one
        # pass over every axis the axes are back in their original order.
        x = np.tensordot(x, factor, axes=([0], [1]))

    return scale * x


def _fro(x):
    return np.linalg.norm(np.asarray(x).ravel())


def _assert_components_near(n_components, ranks):
    """Check MP-estimated component counts against the true ranks.

    The Marchenko-Pastur cutoff can let the largest noise component through for some
    random draws, so one extra component per index is accepted; underestimation is not.
    """
    n_components = np.asarray(n_components)
    ranks = np.asarray(ranks)
    assert n_components.shape == ranks.shape
    assert np.all(n_components >= ranks), (n_components, ranks)
    assert np.all(n_components <= ranks + 1), (n_components, ranks)


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((5,), (5, 1)),
        ((5, 1), (5, 1)),
        ((5, 1, 1), (5, 1)),
        ((2, 3, 1), (2, 3)),
        ((2, 3, 1, 4), (2, 3, 1, 4)),
        ((1, 1), (1, 1)),
    ],
)
def test_matlab_shape(shape, expected):
    assert tensor_pca._matlab_shape(shape) == expected


def test_opt_shrink_frob_matches_closed_form():
    vals2 = np.array([400.0, 150.0, 90.0])
    m, n, sigma2 = 4, 9, 1.5
    expected = vals2 - 2 * (n + m) * sigma2 + (n - m) ** 2 * sigma2**2 / vals2
    np.testing.assert_allclose(opt_shrink_frob(vals2, m, n, sigma2), expected)


def test_opt_shrink_frob_never_inflates_retained_components():
    # Retained components lie above the MP edge, and for those the shrinkage is a
    # reduction.
    m, n, sigma2 = 20, 200, 2.0
    edge = sigma2 * (np.sqrt(m) + np.sqrt(n)) ** 2
    vals2 = edge * np.array([1.01, 2.0, 10.0, 100.0])
    shrunk = opt_shrink_frob(vals2, m, n, sigma2)
    assert np.all(shrunk <= vals2)


def test_estimate_noise_on_pure_noise():
    rng = np.random.default_rng(0)
    sigma = 2.0
    x = sigma * rng.standard_normal((40, 400))
    s = np.linalg.svd(x, compute_uv=False)
    sigma2, n_components = tensor_pca._estimate_noise(s, x.shape)
    assert n_components <= 1
    assert sigma2 == pytest.approx(sigma**2, rel=0.1)


def test_estimate_noise_handles_no_noise_components():
    # The smallest singular value always falls below its own cutoff unless every
    # singular value is zero, which is the only way to reach the "no noise" branch.
    s = np.zeros(3)
    sigma2, n_components = tensor_pca._estimate_noise(s, (3, 10))
    assert n_components == 3
    assert sigma2 == 0.0


def test_subtract_mean_follows_longer_dimension():
    rng = np.random.default_rng(1)
    wide = rng.standard_normal((3, 10))
    centered, mean = tensor_pca._subtract_mean(wide)
    assert mean.shape == (3, 1)
    np.testing.assert_allclose(centered.mean(axis=1), 0, atol=1e-12)

    tall = rng.standard_normal((10, 3))
    centered, mean = tensor_pca._subtract_mean(tall)
    assert mean.shape == (1, 3)
    np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-12)


# ----------------------------------------------------------------------------------
# denoise_array_recursive_tensor: matrices
# ----------------------------------------------------------------------------------


def test_matrix_vector_special_case():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((10, 1))
    denoised, sigma2, n_components = denoise_array_recursive_tensor(x)
    np.testing.assert_array_equal(denoised, x)
    assert sigma2 == 0.0
    np.testing.assert_array_equal(n_components, [1])

    # 1D input is treated as a column vector.
    denoised, _, _ = denoise_array_recursive_tensor(x.ravel())
    np.testing.assert_array_equal(denoised, x)


def test_matrix_denoising_recovers_low_rank_signal():
    rng = np.random.default_rng(3)
    m, n, rank = 30, 300, 3
    signal = _low_rank_matrix(rng, (m, n), rank, scale=3.0)
    noise = rng.standard_normal((m, n))
    denoised, sigma2, n_components = denoise_array_recursive_tensor(signal + noise)

    assert denoised.shape == (m, n)
    _assert_components_near(n_components, [rank])
    assert sigma2 == pytest.approx(1.0, rel=0.15)
    assert _fro(denoised - signal) < 0.5 * _fro(noise)


def test_matrix_denoising_without_shrinkage_projects_onto_signal_subspace():
    rng = np.random.default_rng(4)
    m, n, rank = 20, 100, 2
    signal = _low_rank_matrix(rng, (m, n), rank, scale=5.0)
    noisy = signal + rng.standard_normal((m, n))
    denoised, _, n_components = denoise_array_recursive_tensor(noisy, opt_shrink=False)

    # Without shrinkage, the result is the rank-P truncated SVD of the input.
    u, s, vh = np.linalg.svd(noisy, full_matrices=False)
    p = n_components[0]
    expected = (u[:, :p] * s[:p]) @ vh[:p]
    np.testing.assert_allclose(denoised, expected, atol=1e-10)


def test_matrix_shrinkage_reduces_singular_values():
    rng = np.random.default_rng(5)
    m, n, rank = 20, 100, 2
    noisy = _low_rank_matrix(rng, (m, n), rank, scale=5.0) + rng.standard_normal((m, n))
    without, _, _ = denoise_array_recursive_tensor(noisy, opt_shrink=False)
    with_shrink, _, _ = denoise_array_recursive_tensor(noisy, opt_shrink=True)
    s_without = np.linalg.svd(without, compute_uv=False)[:rank]
    s_with = np.linalg.svd(with_shrink, compute_uv=False)[:rank]
    assert np.all(s_with < s_without)


def test_matrix_known_sigma2():
    rng = np.random.default_rng(6)
    m, n, rank = 30, 200, 2
    signal = _low_rank_matrix(rng, (m, n), rank, scale=3.0)
    noisy = signal + rng.standard_normal((m, n))
    denoised, sigma2, n_components = denoise_array_recursive_tensor(noisy, sigma2=1.0)
    assert sigma2 == 1.0
    _assert_components_near(n_components, [rank])
    assert _fro(denoised - signal) < _fro(noisy - signal)


def test_matrix_subtract_mean():
    rng = np.random.default_rng(7)
    m, n, rank = 30, 300, 2
    signal = _low_rank_matrix(rng, (m, n), rank, scale=3.0)
    # A large per-row offset would otherwise count as an extra signal component.
    offset = 50 * rng.standard_normal((m, 1))
    noise = rng.standard_normal((m, n))
    denoised, _, n_components = denoise_array_recursive_tensor(
        signal + offset + noise,
        subtract_mean=True,
    )
    _assert_components_near(n_components, [rank])
    assert _fro(denoised - (signal + offset)) < 0.5 * _fro(noise)


def test_matrix_pure_noise_is_mostly_removed():
    rng = np.random.default_rng(8)
    noise = rng.standard_normal((40, 400))
    denoised, sigma2, n_components = denoise_array_recursive_tensor(noise)
    assert n_components[0] <= 1
    assert sigma2 == pytest.approx(1.0, rel=0.1)
    assert _fro(denoised) < 0.3 * _fro(noise)


# ----------------------------------------------------------------------------------
# denoise_array_recursive_tensor: tensors
# ----------------------------------------------------------------------------------


def test_tensor_exact_low_rank_is_reconstructed():
    # With a negligible noise variance every singular value above numerical noise is
    # kept, so the recursion must reproduce the input and the multilinear ranks exactly.
    # This exercises the column-major unfolding and the backward reconstruction pass.
    rng = np.random.default_rng(9)
    shape, ranks = (6, 7, 8), (2, 3, 4)
    x = _low_rank_tensor(rng, shape, ranks)
    denoised, sigma2, n_components = denoise_array_recursive_tensor(x, sigma2=1e-12)
    assert denoised.shape == shape
    assert sigma2 == 1e-12
    np.testing.assert_array_equal(n_components, ranks)
    np.testing.assert_allclose(denoised, x, atol=1e-8)


def test_tensor_exact_low_rank_four_way():
    rng = np.random.default_rng(10)
    shape, ranks = (5, 6, 4, 7), (2, 3, 2, 3)
    x = _low_rank_tensor(rng, shape, ranks)
    denoised, _, n_components = denoise_array_recursive_tensor(x, sigma2=1e-12)
    np.testing.assert_array_equal(n_components, ranks)
    np.testing.assert_allclose(denoised, x, atol=1e-8)


def test_tensor_denoising_recovers_low_rank_signal():
    rng = np.random.default_rng(11)
    shape, ranks = (12, 15, 20), (2, 3, 2)
    signal = _low_rank_tensor(rng, shape, ranks, scale=3.0)
    noise = rng.standard_normal(shape)
    denoised, sigma2, n_components = denoise_array_recursive_tensor(signal + noise)

    assert denoised.shape == shape
    _assert_components_near(n_components, ranks)
    assert sigma2 == pytest.approx(1.0, rel=0.2)
    assert _fro(denoised - signal) < 0.5 * _fro(noise)


def test_tensor_single_sigma2_pass():
    rng = np.random.default_rng(12)
    shape, ranks = (12, 15, 20), (2, 3, 2)
    signal = _low_rank_tensor(rng, shape, ranks, scale=3.0)
    noise = rng.standard_normal(shape)
    denoised, sigma2, n_components = denoise_array_recursive_tensor(
        signal + noise,
        full_sigma2_pass=False,
    )
    _assert_components_near(n_components, ranks)
    assert sigma2 == pytest.approx(1.0, rel=0.2)
    assert _fro(denoised - signal) < 0.5 * _fro(noise)


def test_tensor_subtract_mean():
    rng = np.random.default_rng(13)
    shape, ranks = (12, 15, 20), (2, 3, 2)
    signal = _low_rank_tensor(rng, shape, ranks, scale=3.0)
    noise = rng.standard_normal(shape)
    denoised, sigma2, n_components = denoise_array_recursive_tensor(
        signal + noise,
        subtract_mean=True,
    )
    assert denoised.shape == shape
    assert np.all(np.isfinite(denoised))
    assert n_components.shape == (3,)
    assert sigma2 == pytest.approx(1.0, rel=0.2)
    assert _fro(denoised - signal) < 0.6 * _fro(noise)


def test_tensor_num_inds_limits_recursion():
    rng = np.random.default_rng(14)
    shape, ranks = (6, 7, 8), (2, 3, 4)
    x = _low_rank_tensor(rng, shape, ranks)
    denoised, _, n_components = denoise_array_recursive_tensor(x, num_inds=2, sigma2=1e-12)
    np.testing.assert_array_equal(n_components, ranks[:2])
    np.testing.assert_allclose(denoised, x, atol=1e-8)


def test_tensor_pure_noise_stops_recursion():
    rng = np.random.default_rng(15)
    shape = (20, 25, 30)
    noise = rng.standard_normal(shape)
    denoised, sigma2, n_components = denoise_array_recursive_tensor(noise)
    assert denoised.shape == shape
    assert n_components.shape == (3,)
    assert sigma2 == pytest.approx(1.0, rel=0.1)
    # Once an index has no signal components, the remaining ones are reported as zero.
    first_zero = np.flatnonzero(n_components == 0)
    if first_zero.size:
        assert np.all(n_components[first_zero[0] :] == 0)

    assert _fro(denoised) < 0.5 * _fro(noise)


def test_tensor_known_sigma2_with_zero_components_gives_zeros():
    rng = np.random.default_rng(16)
    noise = rng.standard_normal((10, 12, 14))
    # A huge noise variance discards every component.
    denoised, _, n_components = denoise_array_recursive_tensor(noise, sigma2=1e6)
    np.testing.assert_array_equal(n_components, [0, 0, 0])
    np.testing.assert_array_equal(denoised, 0)


def test_tensor_trailing_singleton_axis_is_treated_as_matrix():
    rng = np.random.default_rng(17)
    noisy = _low_rank_matrix(rng, (20, 100), 2, scale=5.0) + rng.standard_normal((20, 100))
    as_matrix = denoise_array_recursive_tensor(noisy)
    as_tensor = denoise_array_recursive_tensor(noisy[:, :, np.newaxis])
    np.testing.assert_allclose(as_tensor[0], as_matrix[0])
    assert as_tensor[1] == as_matrix[1]
    np.testing.assert_array_equal(as_tensor[2], as_matrix[2])


def test_invalid_num_inds():
    with pytest.raises(ValueError, match="num_inds"):
        denoise_array_recursive_tensor(np.zeros((3, 4, 5)), num_inds=4)

    with pytest.raises(ValueError, match="num_inds"):
        denoise_array_recursive_tensor(np.zeros((3, 4, 5)), num_inds=0)


# ----------------------------------------------------------------------------------
# denoise_recursive_tensor: sliding window
# ----------------------------------------------------------------------------------


def _noisy_data(rng, shape, ranks, scale=3.0):
    signal = _low_rank_tensor(rng, shape, ranks, scale=scale)
    return signal, signal + rng.standard_normal(shape)


def test_sliding_window_output_shapes_and_dtypes():
    rng = np.random.default_rng(20)
    shape = (8, 8, 3, 10, 2)
    _, noisy = _noisy_data(rng, shape, (2, 2, 2, 3, 2))
    denoised, sigma2, n_components, snr_gain = denoise_recursive_tensor(noisy, (4, 4))

    assert denoised.shape == shape
    assert sigma2.shape == (8, 8)
    # Default indices group the window axes: (16, 3, 10, 2) -> four tensor indices.
    assert n_components.shape == (8, 8, 4)
    assert snr_gain.shape == (8, 8)
    # Every voxel is covered by at least one window position.
    assert np.all(np.isfinite(denoised))
    assert np.all(np.isfinite(sigma2))
    assert np.all(np.isfinite(n_components))
    assert np.all(snr_gain >= 1)


def test_sliding_window_reduces_noise():
    rng = np.random.default_rng(21)
    shape = (8, 8, 3, 30, 3)
    signal, noisy = _noisy_data(rng, shape, (2, 2, 2, 3, 2))
    denoised, sigma2, _, _ = denoise_recursive_tensor(noisy, (4, 4))
    assert _fro(denoised - signal) < 0.6 * _fro(noisy - signal)
    assert np.nanmedian(sigma2) == pytest.approx(1.0, rel=0.3)


def test_nonoverlapping_windows_match_direct_patch_denoising():
    # With the stride equal to the window, every voxel is visited exactly once, so the
    # sliding-window result must equal the array function applied to each patch. This
    # checks the linear-index offsets, the permutation, and the reshapes.
    rng = np.random.default_rng(22)
    shape = (8, 6, 5, 7)
    _, noisy = _noisy_data(rng, shape, (2, 2, 2, 3))
    window = (4, 3)
    denoised, sigma2, n_components, _ = denoise_recursive_tensor(noisy, window, stride=window)

    for i0 in range(0, 8, 4):
        for j0 in range(0, 6, 3):
            patch = noisy[i0 : i0 + 4, j0 : j0 + 3]
            x = patch.reshape((12, 5, 7), order="F")
            x_denoised, patch_sigma2, patch_p = denoise_array_recursive_tensor(x)
            expected = x_denoised.reshape((4, 3, 5, 7), order="F")
            np.testing.assert_allclose(denoised[i0 : i0 + 4, j0 : j0 + 3], expected, atol=1e-10)
            np.testing.assert_allclose(sigma2[i0 : i0 + 4, j0 : j0 + 3], patch_sigma2)
            np.testing.assert_array_equal(
                n_components[i0 : i0 + 4, j0 : j0 + 3],
                np.broadcast_to(patch_p, (4, 3, 3)),
            )


def test_overlapping_windows_average_patches():
    rng = np.random.default_rng(23)
    shape = (5, 4, 6)
    _, noisy = _noisy_data(rng, shape, (2, 2, 2))
    # Two window positions along the first axis (corners 0 and 1) that overlap on
    # rows 1-3.
    denoised, _, _, _ = denoise_recursive_tensor(noisy, (4, 4))

    expected = np.zeros(shape)
    count = np.zeros(shape[:2])
    for i0 in (0, 1):
        x = noisy[i0 : i0 + 4].reshape((16, 6), order="F")
        x_denoised = denoise_array_recursive_tensor(x)[0].reshape((4, 4, 6), order="F")
        expected[i0 : i0 + 4] += x_denoised
        count[i0 : i0 + 4] += 1

    expected /= count[..., np.newaxis]
    np.testing.assert_allclose(denoised, expected, atol=1e-10)


def test_indices_control_tensor_structure():
    rng = np.random.default_rng(24)
    shape = (4, 4, 6, 5)
    _, noisy = _noisy_data(rng, shape, (2, 2, 3, 2))
    window = (4, 4)

    # One group for all spatial axes: each patch is a (96, 5) matrix.
    denoised, _, n_components, _ = denoise_recursive_tensor(
        noisy,
        window,
        indices=[[0, 1, 2], [3]],
    )
    x = noisy.reshape((96, 5), order="F")
    expected = denoise_array_recursive_tensor(x)[0].reshape(shape, order="F")
    np.testing.assert_allclose(denoised, expected, atol=1e-10)
    assert n_components.shape == (4, 4, 1)

    # Default grouping: each patch is a (16, 6, 5) tensor.
    denoised, _, n_components, _ = denoise_recursive_tensor(noisy, window)
    x = noisy.reshape((16, 6, 5), order="F")
    expected = denoise_array_recursive_tensor(x)[0].reshape(shape, order="F")
    np.testing.assert_allclose(denoised, expected, atol=1e-10)
    assert n_components.shape == (4, 4, 3)

    # Reordered groups: the slice axis is denoised first.
    denoised, _, _, _ = denoise_recursive_tensor(noisy, window, indices=[[2], [0, 1], [3]])
    x = np.transpose(noisy, (2, 0, 1, 3)).reshape((6, 16, 5), order="F")
    expected = denoise_array_recursive_tensor(x)[0].reshape((6, 4, 4, 5), order="F")
    expected = np.transpose(expected, (1, 2, 0, 3))
    np.testing.assert_allclose(denoised, expected, atol=1e-10)


def test_indices_beyond_data_add_singleton_axes():
    rng = np.random.default_rng(25)
    shape = (4, 4, 6, 5)
    _, noisy = _noisy_data(rng, shape, (2, 2, 3, 2))
    # Referring to a fifth axis that does not exist pads the patch with a singleton
    # axis, which is then removed again when the patch is reshaped.
    with_pad = denoise_recursive_tensor(noisy, (4, 4), indices=[[0, 1], [2], [3], [4]])
    without = denoise_recursive_tensor(noisy, (4, 4))
    np.testing.assert_allclose(with_pad[0], without[0], atol=1e-10)


def test_trailing_singleton_axis_does_not_change_values():
    rng = np.random.default_rng(36)
    shape = (6, 6, 4, 5)
    _, noisy = _noisy_data(rng, shape, (2, 2, 2, 2))
    plain = denoise_recursive_tensor(noisy, (3, 3))
    padded = denoise_recursive_tensor(noisy[..., np.newaxis], (3, 3))
    assert padded[0].shape == shape + (1,)
    np.testing.assert_allclose(padded[0][..., 0], plain[0], atol=1e-10)
    np.testing.assert_allclose(padded[1], plain[1])
    np.testing.assert_array_equal(padded[2], plain[2])


def test_mask_skips_windows_without_in_mask_voxels():
    rng = np.random.default_rng(26)
    shape = (8, 8, 4, 6)
    _, noisy = _noisy_data(rng, shape, (2, 2, 2, 2))
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4, :4] = True
    denoised, sigma2, n_components, snr_gain = denoise_recursive_tensor(
        noisy,
        (4, 4),
        mask=mask,
    )

    # Windows whose corner is at index 3 or less still touch the masked block, so
    # voxels up to index 6 are assigned. Row and column 7 are never covered.
    assert np.all(np.isfinite(sigma2[:7, :7]))
    assert np.all(np.isnan(sigma2[7, :]))
    assert np.all(np.isnan(sigma2[:, 7]))
    assert np.all(np.isnan(n_components[7, :, :]))
    assert np.all(np.isnan(snr_gain[7, :]))
    np.testing.assert_array_equal(denoised[7, :], noisy[7, :])
    np.testing.assert_array_equal(denoised[:, 7], noisy[:, 7])
    assert not np.allclose(denoised[:4, :4], noisy[:4, :4])


def test_mask_shape_is_validated():
    noisy = np.zeros((8, 8, 4, 6))
    with pytest.raises(ValueError, match="mask dimensions"):
        denoise_recursive_tensor(noisy, (4, 4), mask=np.ones((8, 8, 4), dtype=bool))


def test_center_assign_uses_center_voxel_only():
    rng = np.random.default_rng(27)
    shape = (9, 9, 4, 6)
    _, noisy = _noisy_data(rng, shape, (2, 2, 2, 2))
    window = (3, 3)
    denoised, sigma2, _, _ = denoise_recursive_tensor(noisy, window, center_assign=True)

    # Only the centers of windows that fit (indices 1-7) are assigned.
    assert np.all(np.isfinite(sigma2[1:8, 1:8]))
    assert np.all(np.isnan(sigma2[0, :]))
    assert np.all(np.isnan(sigma2[8, :]))
    assert np.all(np.isnan(sigma2[:, 0]))
    assert np.all(np.isnan(sigma2[:, 8]))
    np.testing.assert_array_equal(denoised[0], noisy[0])

    # Each assigned voxel equals the center of its own window, denoised directly.
    i0, j0 = 2, 5
    x = noisy[i0 : i0 + 3, j0 : j0 + 3].reshape((9, 4, 6), order="F")
    x_denoised = denoise_array_recursive_tensor(x)[0].reshape((3, 3, 4, 6), order="F")
    np.testing.assert_allclose(denoised[i0 + 1, j0 + 1], x_denoised[1, 1], atol=1e-10)


def test_center_assign_respects_mask():
    rng = np.random.default_rng(28)
    shape = (9, 9, 4, 6)
    _, noisy = _noisy_data(rng, shape, (2, 2, 2, 2))
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True
    _, sigma2, _, _ = denoise_recursive_tensor(noisy, (3, 3), mask=mask, center_assign=True)
    assert np.isfinite(sigma2[4, 4])
    assert np.count_nonzero(np.isfinite(sigma2)) == 1


def test_one_dimensional_window():
    # Data with a single voxel axis, for example masked voxels by time by echo.
    rng = np.random.default_rng(29)
    shape = (20, 10, 3)
    _, noisy = _noisy_data(rng, shape, (3, 2, 2))
    denoised, sigma2, n_components, snr_gain = denoise_recursive_tensor(noisy, (5,), stride=5)

    assert denoised.shape == shape
    assert sigma2.shape == (20,)
    assert n_components.shape == (20, 3)
    assert snr_gain.shape == (20,)
    assert np.all(np.isfinite(sigma2))
    for i0 in range(0, 20, 5):
        x_denoised, patch_sigma2, _ = denoise_array_recursive_tensor(noisy[i0 : i0 + 5])
        np.testing.assert_allclose(denoised[i0 : i0 + 5], x_denoised, atol=1e-10)
        np.testing.assert_allclose(sigma2[i0 : i0 + 5], patch_sigma2)


def test_stride_skips_positions():
    rng = np.random.default_rng(30)
    shape = (8, 8, 4, 6)
    _, noisy = _noisy_data(rng, shape, (2, 2, 2, 2))
    _, sigma2, _, _ = denoise_recursive_tensor(noisy, (3, 3), stride=(3, 3))
    # Corners at 0 and 3 cover indices 0-5 along each axis; corner 6 does not fit.
    assert np.all(np.isfinite(sigma2[:6, :6]))
    assert np.all(np.isnan(sigma2[6:, :]))
    assert np.all(np.isnan(sigma2[:, 6:]))

    # A scalar stride applies to every window axis.
    _, sigma2_scalar, _, _ = denoise_recursive_tensor(noisy, (3, 3), stride=3)
    np.testing.assert_array_equal(np.isnan(sigma2_scalar), np.isnan(sigma2))


def test_known_sigma2_is_used_everywhere():
    rng = np.random.default_rng(31)
    shape = (6, 6, 4, 6)
    _, noisy = _noisy_data(rng, shape, (2, 2, 2, 2))
    _, sigma2, _, _ = denoise_recursive_tensor(noisy, (3, 3), sigma2=1.0)
    np.testing.assert_array_equal(sigma2, 1.0)


def test_matrix_patches_give_single_component_count_and_snr_gain():
    rng = np.random.default_rng(32)
    shape = (8, 8, 10)
    _, noisy = _noisy_data(rng, shape, (2, 2, 3))
    # With only one measurement axis, each patch is a (16, 10) matrix.
    _, _, n_components, snr_gain = denoise_recursive_tensor(noisy, (4, 4))
    assert n_components.shape == (8, 8, 1)
    assert np.all(np.isfinite(snr_gain))

    # SNR gain for a matrix: sqrt(M N / (P^2 + (M - P) P + (N - P) P)).
    p = n_components[0, 0, 0]
    expected = np.sqrt(160 / (p**2 + (16 - p) * p + (10 - p) * p))
    assert snr_gain[0, 0] == pytest.approx(expected)


def test_tensor_snr_gain_formula():
    rng = np.random.default_rng(33)
    shape = (4, 4, 6, 5)
    _, noisy = _noisy_data(rng, shape, (2, 2, 3, 2))
    _, _, n_components, snr_gain = denoise_recursive_tensor(noisy, (4, 4))
    p = n_components[0, 0]
    sizes = np.array([16, 6, 5])
    expected = np.sqrt(np.prod(sizes) / (np.prod(p) + np.sum((sizes - p) * p)))
    assert snr_gain[0, 0] == pytest.approx(expected)


def test_integer_input_is_converted_to_float():
    rng = np.random.default_rng(34)
    data = rng.integers(0, 100, size=(6, 6, 4, 5))
    denoised, _, _, _ = denoise_recursive_tensor(data, (3, 3))
    assert np.issubdtype(denoised.dtype, np.floating)


def test_float32_input_stays_float32():
    rng = np.random.default_rng(35)
    data = rng.standard_normal((6, 6, 4, 5)).astype(np.float32)
    denoised, _, _, _ = denoise_recursive_tensor(data, (3, 3))
    assert denoised.dtype == np.float32


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window": (4, 4, 4, 4, 4)}, "window must have"),
        ({"window": (0, 4)}, "positive"),
        ({"window": (9, 4)}, "does not fit"),
        ({"window": (4, 4), "indices": [[0, 1], [1]]}, "same axis twice"),
        ({"window": (4, 4), "indices": [[0, 1], [-1]]}, "non-negative"),
        ({"window": (4, 4), "indices": []}, "at least one"),
        ({"window": (4, 4), "stride": (1, 1, 1)}, "one element per window"),
        ({"window": (4, 4), "stride": 0}, "stride must be positive"),
        ({"window": (4, 4), "indices": [[0, 1], [2]]}, "cover every"),
    ],
)
def test_invalid_sliding_window_arguments(kwargs, match):
    noisy = np.zeros((8, 8, 4, 6))
    with pytest.raises(ValueError, match=match):
        denoise_recursive_tensor(noisy, **kwargs)
