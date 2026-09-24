# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

from packaging.version import parse as parse_version

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent)]

import scgenome  # noqa

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# -- General configuration ------------------------------------------------


nitpicky = True  # Warn about broken links. This is here for a reason: Do not change.
needs_sphinx = '2.0'  # Nicer param docs
suppress_warnings = [
    'ref.citation',
    'myst.header',  # https://github.com/executablebooks/MyST-Parser/issues/262
]

project = 'scgenome'
copyright = '2018, McPherson'
author = 'Andrew McPherson'

# Derived from the installed package rather than hardcoded, which had drifted to
# 0.0.7 while the package was at 0.0.20. versioneer produces things like
# '0.0.20+36.g6d32118' for non-tagged builds; keep the full string as the release
# and the short X.Y.Z as the version.
release = scgenome.__version__

if release.startswith('0+'):
    # versioneer reports '0+unknown' when there is no git metadata and no
    # baked-in _version.py, e.g. a bare source export. Say so rather than
    # rendering a misleading '0.0.0'.
    release = version = 'unknown'

else:
    _parsed = parse_version(release)
    version = f'{_parsed.major}.{_parsed.minor}.{_parsed.micro}'

    if _parsed.is_devrelease:
        version += '.dev'

# default settings
templates_path = ['_templates']
master_doc = 'index'
default_role = 'literal'
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    # The jupytext .md files under notebooks/ are the source of truth. Any
    # paired .ipynb on disk would otherwise collide with them for the same
    # docname ("multiple files found for the document ...").
    'notebooks/*.ipynb',
]
pygments_style = 'sphinx'
source_suffix = [".rst", ".md"]

extensions = [
    # myst_nb supersedes myst_parser (it enables it internally) and additionally
    # executes the gallery pages so their figures appear in the built docs
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',
    'nbsphinx',
    'sphinx_gallery.load_style',
    'sphinx_autodoc_typehints',  # needs to be after napoleon
    'scanpydoc.rtd_github_links',
]


# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = 'bysource'
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
todo_include_todos = False

typehints_defaults = 'braces'

# The gallery pages under notebooks/ are jupytext markdown, not MyST-NB
# markdown, so their ```python fences are only recognised as code cells when
# jupytext parses them. Without this they render as inert code blocks with no
# figures.
nb_custom_formats = {'.md': ['jupytext.reads', {'fmt': 'md'}]}

# 'auto' executes notebooks that have no stored outputs. Jupytext markdown never
# stores outputs, so in practice every gallery page is executed on each build.
nb_execution_mode = 'auto'
nb_execution_timeout = 300

# A gallery page that stops working should fail the build rather than publish
# a traceback where a figure should be
nb_execution_raise_on_error = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
}

# -- Options for HTML output ----------------------------------------------

html_theme = 'scanpydoc'
html_context = {
    'display_github': True,
    "github_user": "shahcompbio",
    "github_repo": "scgenome",
    "github_version": "master",
    "conf_py_path": "/docs/",
}


def setup(app):
    app.warningiserror = on_rtd

# -- Options for other output formats ------------------------------------------

htmlhelp_basename = f'{project}doc'
doc_title = f'{project} Documentation'
latex_documents = [(master_doc, f'{project}.tex', doc_title, author, 'manual')]
man_pages = [(master_doc, project, doc_title, [author], 1)]
texinfo_documents = [
    (
        master_doc,
        project,
        doc_title,
        author,
        project,
        'One line description of project.',
        'Miscellaneous',
    )
]


# Options for plot examples
nitpick_ignore = [
    ('py:class', 'type'),
    ('py:class', 'AnnData'),
    ('py:class', 'Axes'),
    ('py:class', 'Figure'),
    ('py:class', 'optional'),
    ('py:class', 'Bio.Phylo.BaseTree.Tree'),
    ('py:class', 'matplotlib.colors.ListedColormap'),
    ('py:class', 'anndata.AnnData'),
    ('py:class', 'ad.AnnData'),
    ('py:class', 'anndata._core.anndata.AnnData'),
    ('py:class', 'DataFrame'),
    ('py:class', 'Sequence'),
    # Docstring type names that refer to scgenome's own helpers or to loose
    # duck-typed conventions, neither of which resolve as classes
    ('py:class', 'RefGenomeInfo'),
    ('py:class', 'RegionMapper'),
    ('py:class', 'Legend'),
    ('py:class', 'color'),
    ('py:class', 'function'),
    ('py:class', 'callable'),
    ('py:class', 'pd.Series'),
    ('py:class', 'matplotlib.colors.Colormap'),
    ('py:class', 'PyRanges'),
    ('py:class', 'pyranges.PyRanges'),
    ('py:class', 'pyrange.PyRanges'),
    ('py:class', 'pyranges.pyranges.PyRanges'),
    ('py:class', 'ndarray'),
    ('py:class', 'matplotlib.axes.Axes'),
    ('py:class', 'matplotlib.figure.Figure'),
    # Currently undocumented: https://github.com/mwaskom/seaborn/issues/1810
    ('py:class', 'seaborn.ClusterGrid'),
    ('py:class', 'numpy.random.mtrand.RandomState'),
    # Will work once scipy 1.8 is released
    ('py:class', 'scipy.sparse.base.spmatrix'),
    ('py:class', 'scipy.sparse.csr.csr_matrix'),
]

plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_working_directory = HERE.parent  # Project root
