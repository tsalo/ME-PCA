"""Tests for the multi-echo workflow and command-line interface."""

import os

import nibabel as nb
import numpy as np
import pytest

from mepca.cli import main
from mepca.workflows import denoise_multi_echo, output_filename


@pytest.mark.parametrize(
    ("filename", "desc", "suffix", "expected"),
    [
        ("sub-01_echo-1_bold.nii.gz", "tmppca", None, "sub-01_echo-1_desc-tmppca_bold.nii.gz"),
        ("sub-01_echo-1_bold.nii", "tmppca", None, "sub-01_echo-1_desc-tmppca_bold.nii"),
        ("sub-01_bold.nii.gz", "tmppca", "sigma2", "sub-01_desc-tmppca_sigma2.nii.gz"),
        ("data.nii.gz", "x", None, "data_desc-x.nii.gz"),
        ("data.nii.gz", "x", "snrgain", "data_desc-x_snrgain.nii.gz"),
    ],
)
def test_output_filename(filename, desc, suffix, expected):
    out = output_filename(os.path.join("in", filename), "out", desc, suffix=suffix)
    assert out == os.path.join("out", expected)


@pytest.fixture
def echo_files(tmp_path):
    rng = np.random.default_rng(0)
    shape = (6, 6, 2, 8)
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    files = []
    for i_echo in range(3):
        data = 100 * np.exp(-0.3 * (i_echo + 1)) + rng.standard_normal(shape)
        img = nb.Nifti1Image(data.astype(np.float32), affine)
        fname = tmp_path / f"sub-01_task-rest_echo-{i_echo + 1}_bold.nii.gz"
        img.to_filename(fname)
        files.append(str(fname))

    return files


def test_denoise_multi_echo_writes_outputs(echo_files, tmp_path):
    out_dir = tmp_path / "out"
    outputs = denoise_multi_echo(echo_files, str(out_dir), window=(3, 3))

    assert len(outputs["echoes"]) == 3
    for i_echo, out_file in enumerate(outputs["echoes"]):
        assert os.path.basename(out_file) == (
            f"sub-01_task-rest_echo-{i_echo + 1}_desc-tmppca_bold.nii.gz"
        )
        img = nb.load(out_file)
        assert img.shape == (6, 6, 2, 8)
        assert np.all(np.isfinite(img.get_fdata()))

    sigma2 = nb.load(outputs["sigma2"])
    assert os.path.basename(outputs["sigma2"]) == "sub-01_task-rest_desc-tmppca_sigma2.nii.gz"
    assert sigma2.shape == (6, 6)
    n_components = nb.load(outputs["n_components"])
    assert n_components.shape == (6, 6, 3)
    snr_gain = nb.load(outputs["snr_gain"])
    assert snr_gain.shape == (6, 6)


def test_denoise_multi_echo_three_dimensional_window_and_mask(echo_files, tmp_path):
    mask = np.zeros((6, 6, 2), dtype=bool)
    mask[:3, :3, :] = True
    mask_file = tmp_path / "mask.nii.gz"
    nb.Nifti1Image(mask.astype(np.uint8), np.diag([3.0, 3.0, 3.0, 1.0])).to_filename(mask_file)

    outputs = denoise_multi_echo(
        echo_files,
        str(tmp_path / "out3d"),
        window=(3, 3, 2),
        mask_file=str(mask_file),
    )
    sigma2 = nb.load(outputs["sigma2"]).get_fdata()
    assert sigma2.shape == (6, 6, 2)
    assert np.all(np.isfinite(sigma2[:5, :5]))
    assert np.all(np.isnan(sigma2[5, :]))


def test_denoise_multi_echo_validates_inputs(echo_files, tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        denoise_multi_echo(echo_files[:1], str(tmp_path))

    with pytest.raises(ValueError, match="two or three elements"):
        denoise_multi_echo(echo_files, str(tmp_path), window=(3,))

    other = tmp_path / "other.nii.gz"
    nb.Nifti1Image(np.zeros((6, 6, 2, 9), dtype=np.float32), np.eye(4)).to_filename(other)
    with pytest.raises(ValueError, match="same shape"):
        denoise_multi_echo(echo_files + [str(other)], str(tmp_path))

    shifted = tmp_path / "shifted.nii.gz"
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    affine[0, 3] = 3.0
    nb.Nifti1Image(np.zeros((6, 6, 2, 8), dtype=np.float32), affine).to_filename(shifted)
    with pytest.raises(ValueError, match="same grid"):
        denoise_multi_echo(echo_files + [str(shifted)], str(tmp_path))


def test_map_names_ignore_echo_entity_in_directory(echo_files, tmp_path):
    # An "_echo-" substring in the directory name must not affect the map names.
    in_dir = tmp_path / "my_echo-data"
    in_dir.mkdir()
    moved = []
    for f in echo_files:
        target = in_dir / os.path.basename(f)
        os.replace(f, target)
        moved.append(str(target))

    outputs = denoise_multi_echo(moved, str(tmp_path / "out"), window=(3, 3))
    assert os.path.basename(outputs["sigma2"]) == "sub-01_task-rest_desc-tmppca_sigma2.nii.gz"


def test_cli(echo_files, tmp_path, capsys):
    out_dir = tmp_path / "cli"
    exit_code = main(
        echo_files
        + [
            "--output-dir",
            str(out_dir),
            "--window",
            "3",
            "3",
            "--no-opt-shrink",
            "--desc",
            "test",
        ]
    )
    assert exit_code == 0
    written = sorted(os.listdir(out_dir))
    assert written == [
        "sub-01_task-rest_desc-test_ncomponents.nii.gz",
        "sub-01_task-rest_desc-test_sigma2.nii.gz",
        "sub-01_task-rest_desc-test_snrgain.nii.gz",
        "sub-01_task-rest_echo-1_desc-test_bold.nii.gz",
        "sub-01_task-rest_echo-2_desc-test_bold.nii.gz",
        "sub-01_task-rest_echo-3_desc-test_bold.nii.gz",
    ]
    assert capsys.readouterr().out.count("Wrote ") == 6
