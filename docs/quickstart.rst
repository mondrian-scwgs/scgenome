QUICKSTART
===================================

This page walks through a complete copy-number analysis on a small bundled
dataset: load, quality filter, cluster, order, embed, and plot. Every code
block runs as written and the figures below are generated from it during the
documentation build.

If you have not installed scgenome yet, see :doc:`install`. For the reasoning
behind the data model and naming conventions used here, see :doc:`concepts`.

Load an example dataset
-----------------------

scgenome ships two small datasets for experimentation. ``OV2295_HMMCopy_reduced``
is HMMCopy output from the OV2295 ovarian cell lines, reduced to 25 cells and
6206 bins so it loads instantly.

.. plot::
    :context: close-figs
    :nofigs:

    import scgenome

    adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    print(adata.shape)
    print(list(adata.layers))

The result is an :class:`~anndata.AnnData` object. ``adata.X`` holds raw read
counts, ``adata.layers['copy']`` holds continuous copy number, and
``adata.layers['state']`` holds integer copy number states. Cells are rows
(``adata.obs``) and genomic bins are columns (``adata.var``, with ``chr``,
``start`` and ``end`` columns).

Set the genome version
----------------------

Functions that need to know chromosome names and lengths — anything that plots
against genomic coordinates, rebins, or maps genes — resolve the reference
genome from ``adata.uns['genome']``. The bundled dataset is aligned to hg19:

.. plot::
    :context: close-figs
    :nofigs:

    adata.uns['genome'] = 'hg19'

Setting this once on the object is the recommended approach; the alternatives
are a per-call ``genome=`` argument or a global default via
:func:`scgenome.refgenome.set_genome_version`. See :ref:`genome-version` for the
full resolution order.

Quality control
---------------

:func:`~scgenome.pp.calculate_filter_metrics` computes per-cell QC annotations,
and :func:`~scgenome.pp.filter_cells` subsets the object to the cells that pass.
Splitting these into two steps lets you inspect the metrics, adjust thresholds,
or apply your own filter before discarding anything.

.. plot::
    :context: close-figs
    :nofigs:

    adata = scgenome.pp.calculate_filter_metrics(adata)

    # Inspect what each filter would remove before applying it
    filter_cols = [c for c in adata.obs.columns if c.startswith('filter_')]
    print(adata.obs[filter_cols].sum())

    adata = scgenome.pp.filter_cells(adata)
    print(adata.shape)

``calculate_filter_metrics`` skips any filter whose input column is absent and
logs a warning rather than failing. This dataset has no ``is_s_phase`` column,
so ``filter_is_s_phase`` is not computed and is skipped by ``filter_cells``.

Cluster cells
-------------

:func:`~scgenome.tl.cluster_cells` groups cells by copy-number profile. By
default it sweeps k from ``min_k`` to ``max_k`` and selects the value with the
best BIC. With only 25 cells a small ``max_k`` keeps this fast.

.. plot::
    :context: close-figs
    :nofigs:

    adata = scgenome.tl.cluster_cells(adata, layer_name='copy', max_k=5)
    print(adata.obs['cluster_id'].value_counts())

The chosen k and the parameters used are recorded in
``adata.uns['clustering']['params']``, so the object carries a record of how its
own annotations were produced.

Order cells for display
-----------------------

A copy-number heatmap is only readable if similar cells sit next to each other.
:func:`~scgenome.tl.sort_cells` runs hierarchical clustering and writes an
integer ordering to ``adata.obs['cell_order']``.

.. plot::
    :context: close-figs
    :nofigs:

    adata = scgenome.tl.sort_cells(adata, layer_name='copy')
    print(adata.obs['cell_order'].head())

Nothing is reordered on disk — ``cell_order`` is just a column that plotting
functions sort on, which means you can combine it with other columns to get
a nested ordering (cluster first, then within-cluster similarity).

Dimensionality reduction
------------------------

:func:`~scgenome.tl.pca_loadings` and :func:`~scgenome.tl.compute_umap` provide
cell embeddings. PCA results land in ``obsm``/``varm``/``uns`` following scanpy
conventions; UMAP coordinates land in ``obs`` as plain columns.

.. plot::
    :context: close-figs
    :nofigs:

    adata = scgenome.tl.pca_loadings(adata, layer='copy', n_components=5)
    print(adata.obsm['X_pca'].shape)
    print(adata.uns['pca']['variance_ratio'])

    adata = scgenome.tl.compute_umap(adata, layer_name='copy', n_neighbors=5)
    print(adata.obs[['UMAP1', 'UMAP2']].head())

Plot a copy number heatmap
--------------------------

:func:`~scgenome.pl.plot_cell_tcn_matrix` draws the copy-number matrix into a
single axis. ``cell_order_fields`` takes a list of ``obs`` columns to sort rows
by, applied left to right, so the call below groups cells by cluster and then
orders within each cluster by hierarchical similarity.

.. plot::
    :context: close-figs

    scgenome.pl.plot_cell_tcn_matrix(
        adata,
        layer_name='state',
        cell_order_fields=['cluster_id', 'cell_order'],
    )

:func:`~scgenome.pl.plot_cell_tcn_matrix_fig` builds a full figure instead:
heatmap, a legend, and annotation bars for any ``obs`` columns you name.
Categorical and continuous columns are detected automatically and get
appropriate colour scales.

.. plot::
    :context: close-figs

    g = scgenome.pl.plot_cell_tcn_matrix_fig(
        adata,
        layer_name='state',
        cell_order_fields=['cluster_id', 'cell_order'],
        annotation_fields=['cluster_id', 'quality'],
    )

Both functions return a dict of the matplotlib objects they created — ``ax``,
``im`` and the reordered ``adata`` for the single-axis version; ``fig``,
``axes``, ``heatmap_ax`` and more for the figure version. Use these to
customize the result rather than rebuilding the plot.

Plot a single cell profile
--------------------------

:func:`~scgenome.pl.plot_cn_profile` plots one cell along genomic coordinates,
with points positioned by a value layer and coloured by a state layer.

.. plot::
    :context: close-figs

    cell_id = adata.obs.index[0]

    scgenome.pl.plot_cn_profile(
        adata,
        cell_id,
        value_layer_name='copy',
        state_layer_name='state',
    )

Pass ``chromosome='1'`` (optionally with ``start`` and ``end``) to zoom into a
single region instead of plotting the whole genome.

Where to go next
----------------

- :doc:`concepts` — the data model, namespace layout, and the table of what
  each function reads and modifies.
- :doc:`examples` — the plotting gallery, including allele-specific copy number
  and phylogenetic trees.
- :doc:`api` — the complete function reference.
