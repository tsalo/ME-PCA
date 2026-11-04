"""Command-line interface for mepca."""

import argparse

from mepca import __version__
from mepca.workflows import denoise_multi_echo


def _get_parser():
    parser = argparse.ArgumentParser(
        prog="mepca",
        description=(
            "Denoise multi-echo fMRI data with tensor MP-PCA (Olesen et al., 2022). "
            "The echoes are stacked and denoised jointly, treating space, time, and echo "
            "as the three indices of a tensor."
        ),
    )
    parser.add_argument(
        "echo_files",
        nargs="+",
        help="4D magnitude NIfTI files, one per echo, in echo order.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory to write the denoised files and maps to.",
    )
    parser.add_argument(
        "--window",
        nargs="+",
        type=int,
        default=[4, 4],
        metavar="N",
        help=(
            "Sliding window size. Two values slide a window over x and y with patches that "
            "span all slices (default: 4 4). Three values slide over x, y, and z."
        ),
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="3D brain mask. Window positions with no in-mask voxel are skipped.",
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        default=None,
        help="Known noise variance. If omitted, it is estimated per patch.",
    )
    parser.add_argument(
        "--no-opt-shrink",
        action="store_false",
        dest="opt_shrink",
        help="Disable optimal singular value shrinkage.",
    )
    parser.add_argument(
        "--subtract-mean",
        action="store_true",
        help="Subtract the mean of each unfolding before the SVD.",
    )
    parser.add_argument(
        "--center-assign",
        action="store_true",
        help="Assign the denoised value to the center voxel of each window only.",
    )
    parser.add_argument(
        "--stride",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Window step along each window axis (default: 1).",
    )
    parser.add_argument(
        "--desc",
        default="tmppca",
        help="Value of the desc entity in the output filenames (default: tmppca).",
    )
    parser.add_argument("--version", action="version", version=f"mepca {__version__}")
    return parser


def main(argv=None):
    """Run the multi-echo denoising workflow from the command line."""
    parser = _get_parser()
    args = parser.parse_args(argv)
    outputs = denoise_multi_echo(
        args.echo_files,
        args.output_dir,
        window=args.window,
        mask_file=args.mask,
        opt_shrink=args.opt_shrink,
        subtract_mean=args.subtract_mean,
        sigma2=args.sigma2,
        center_assign=args.center_assign,
        stride=args.stride,
        desc=args.desc,
    )
    for out_file in outputs["echoes"]:
        print(f"Wrote {out_file}")

    for name in ("sigma2", "n_components", "snr_gain"):
        print(f"Wrote {outputs[name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
