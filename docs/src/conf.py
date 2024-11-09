# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath("../../mlx_optimizers"))

# -- Project information -----------------------------------------------------

project = "mlx-optimizers"
copyright = "2024, Jason Stock"
author = "Jason Stock"
version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

python_use_unqualified_type_names = True
autosummary_generate = True
autosummary_filename_map = {"mlx.core.Stream": "stream_class"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

breathe_projects = {"mlx_optimizers": "../build/xml"}
breathe_default_project = "mlx_optimizers"

templates_path = ["_templates"]
html_static_path = ["_static"]
source_suffix = ".rst"
main_doc = "index"
highlight_language = "python"
pygments_style = "sphinx"
add_module_names = False

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/stockeh/mlx-optimizers",
    "use_repository_button": True,
    "navigation_with_keys": False,
    "logo": {
        "image_light": "_static/light-mode-logo.svg",
        "image_dark": "_static/dark-mode-logo.svg",
    },
}

html_favicon = html_theme_options["logo"]["image_light"]  # type: ignore

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "mlx_doc"


def setup(app):
    from sphinx.util import inspect

    wrapped_isfunc = inspect.isfunction

    def isfunc(obj):
        type_name = str(type(obj))
        if "nanobind.nb_method" in type_name or "nanobind.nb_func" in type_name:
            return True
        return wrapped_isfunc(obj)

    inspect.isfunction = isfunc


# -- Options for LaTeX output ------------------------------------------------

latex_documents = [(main_doc, "MLX.tex", "MLX Optimization Documentation", author, "manual")]
latex_elements = {
    "preamble": r"""
    \usepackage{enumitem}
    \setlistdepth{5}
    \setlist[itemize,1]{label=$\bullet$}
    \setlist[itemize,2]{label=$\bullet$}
    \setlist[itemize,3]{label=$\bullet$}
    \setlist[itemize,4]{label=$\bullet$}
    \setlist[itemize,5]{label=$\bullet$}
    \renewlist{itemize}{itemize}{5}
""",
}
