"""Single-Cell Whole Genome Sequencing Analysis in Python.

All data is stored in AnnData objects with the following conventions:

    adata.X              — primary matrix (typically raw reads)
    adata.layers['copy'] — continuous copy number values
    adata.layers['state']— integer copy number states
    adata.obs            — per-cell metadata (quality, cluster_id, cell_order)
    adata.var            — per-bin metadata with columns chr, start, end
    adata.uns['genome']  — genome version ('hg19', 'grch38', 'mm10')

API modules:
    scgenome.pp          — preprocessing: load, filter, transform
    scgenome.tl          — tools: cluster, sort, PCA, UMAP, rebin, phylo
    scgenome.pl          — plotting: heatmaps, profiles, trees
    scgenome.datasets    — example datasets

All tl/pp functions take AnnData as first argument, mutate in-place,
and return the same AnnData. Layer selection via layer_name parameter.

Quick start::

    import scgenome
    adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    adata = scgenome.pp.calculate_filter_metrics(adata)
    adata = scgenome.pp.filter_cells(adata)
    adata = scgenome.tl.cluster_cells(adata, layer_name='copy')
    adata = scgenome.tl.sort_cells(adata, layer_name='copy')
    scgenome.pl.plot_cell_cn_matrix(adata, layer_name='state',
                                    cell_order_fields=['cell_order'])
"""

from . import tools as tl
from . import preprocessing as pp
from . import plotting as pl
from . import datasets

from . import _version
__version__ = _version.get_versions()['version']

# has to be done at the end, after everything has been imported
import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'pp', 'pl']})

del sys
