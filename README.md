# scgenome: single cell whole genome analysis in python

![codebuild](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiSTVXZ3NyZWVSejMrd2JRd25GUTJnUmMwclFqd0t0ajNzbWk4dWlld3N4UHVYZzUzUzJkVUVPQmloZUZzbWNGM2lwWUlVN1hDMnBmandJZWdENU5GUjlrPSIsIml2UGFyYW1ldGVyU3BlYyI6IjM2UTFWbjRyUmx0VXlMWUgiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master)

![Documentation Status](https://readthedocs.org/projects/scgenome/badge/?version=latest)

scgenome is scalable python toolkit for analyzing single-cell whole genome
data built on [anndata](https://anndata.readthedocs.io) and inspired by
[scanpy](https://scanpy.readthedocs.io).  scgenome includes preprocessing,
visualization, and clustering functionality and can be used to analyze
copy number, allele specific copy number, SNV and breakpoint features.

## Quick start

```python
import scgenome

# Load a small bundled example dataset (25 cells x 6206 bins)
adata = scgenome.datasets.OV2295_HMMCopy_reduced()

# Tell scgenome which reference genome the bins are on
adata.uns['genome'] = 'hg19'

# Annotate and apply per-cell quality filters
adata = scgenome.pp.calculate_filter_metrics(adata)
adata = scgenome.pp.filter_cells(adata)

# Cluster cells and order them for display
adata = scgenome.tl.cluster_cells(adata, layer_name='copy', max_k=5)
adata = scgenome.tl.sort_cells(adata, layer_name='copy')

# Plot a copy number heatmap with cluster and quality annotations
scgenome.pl.plot_cell_tcn_matrix_fig(
    adata,
    layer_name='state',
    cell_order_fields=['cluster_id', 'cell_order'],
    annotation_fields=['cluster_id', 'quality'],
)
```

Each `pp.*` and `tl.*` call annotates the `AnnData` in place and returns it, so
the workflow reads as a pipeline. `tl.cluster_cells` adds `obs['cluster_id']`,
`tl.sort_cells` adds `obs['cell_order']`, and the plotting call consumes both.

## Learning path

If you are new to scgenome, work through these in order:

1. **[Quickstart](docs/quickstart.rst)** — the walkthrough above, explained step by step.
2. **[Concepts](docs/concepts.rst)** — the AnnData data model, namespaces, and conventions every function follows.
3. **[Gallery](docs/examples.rst)** — notebook-backed plotting examples.
4. **[API reference](docs/api.rst)** — the full function reference.

## Installation

Installation from pypi is recommended:

```
pip install scgenome
```

We also recommend installing scgenome into a virtual environment using
virtualenv.  To create a fresh environment with scgenome use the following
steps:

```
virtualenv venv
source venv/bin/activate
pip install scgenome
```

To install from source, clone this repo and use the following steps in
the repo directory:

```
virtualenv venv
source venv/bin/activate
pip install -e .
```

## Documentation

Documentation is built with Sphinx from the `docs/` directory and published at
[scgenome.readthedocs.io](https://scgenome.readthedocs.io).

| Page | Contents |
|------|----------|
| [docs/install.rst](docs/install.rst) | Installation, including Docker images |
| [docs/quickstart.rst](docs/quickstart.rst) | End-to-end walkthrough on an example dataset |
| [docs/concepts.rst](docs/concepts.rst) | Data model, namespaces, genome versions, conventions |
| [docs/examples.rst](docs/examples.rst) | Plotting gallery backed by notebooks |
| [docs/api.rst](docs/api.rst) | Full `pp` / `tl` / `pl` function reference |

## Pip build

To build a source distribution and upload it to pypi:

```
pip install build twine
python -m build --sdist
twine upload --repository pypi dist/*
```

## Build documentation

```
pip install -r docs/requirements/requirements.txt
pip install scgenome
cd docs
make html
```
