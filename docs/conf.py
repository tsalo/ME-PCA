"""Sphinx configuration for the mepca documentation."""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.pardir))

import mepca  # noqa: E402

# -- General configuration ------------------------------------------------

autosummary_generate = True
add_module_names = False

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "mepca"
copyright = "2024-" + datetime.today().strftime("%Y") + ", mepca developers"
author = "mepca developers"

version = mepca.__version__
release = mepca.__version__

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

# -- Napoleon settings ----------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_admonition_for_notes = True

# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {"includehidden": False}
html_static_path = ["_static"]


def setup(app):
    """Add the CSS overrides."""
    app.add_css_file("theme_overrides.css")


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
}
