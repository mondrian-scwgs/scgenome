CONCEPTS
===================================

scgenome is a toolkit for single-cell whole-genome data built on
`anndata <https://anndata.readthedocs.io>`_ and modelled on
`scanpy <https://scanpy.readthedocs.io>`_. If you have used scanpy, the layout
will be familiar: one annotated matrix carries the data, and functions
progressively annotate it in place.

This page explains the ideas you need to hold in your head while using the
library. For a runnable walkthrough, see :doc:`quickstart`.

The data model
--------------

Everything is an :class:`~anndata.AnnData` object in which **rows are cells and
columns are genomic bins**. A copy-number dataset is a matrix of cells by bins
plus several parallel matrices of the same shape and metadata tables along each
axis.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Slot
     - Contents
   * - ``adata.X``
     - The primary matrix, typically raw read counts per bin.
   * - ``adata.layers['copy']``
     - Continuous copy number values, same shape as ``X``.
   * - ``adata.layers['state']``
     - Integer copy number states, same shape as ``X``.
   * - ``adata.obs``
     - Per-cell metadata: quality scores, read counts, ``cluster_id``, ``cell_order``.
   * - ``adata.var``
     - Per-bin metadata. Always has ``chr``, ``start`` and ``end``.
   * - ``adata.uns``
     - Unstructured annotations: genome version, clustering parameters, PCA metadata.
   * - ``adata.obsm``
     - Multi-column per-cell arrays: ``X_pca``, ``copy_state_diff``.
   * - ``adata.varm``
     - Multi-column per-bin arrays: ``PCs``.

The important consequence of this layout is that **the matrix is never reordered
or overwritten by an analysis step**. Clustering does not group the rows; it
writes a ``cluster_id`` column. Sorting does not permute the rows; it writes a
``cell_order`` column. Reordering happens only at plot time, driven by whichever
``obs`` columns you name. This is why you can apply several orderings to the
same object, and why you can plot different layers of the same cells side by
side without re-running any analysis.

The ``layers`` distinction matters in practice. ``copy`` is continuous and is
what you cluster, sort and embed on. ``state`` is integer and is what you plot,
because the discrete copy-number colour palette is defined on integer states.
Most functions therefore default to ``layer_name='copy'`` for analysis and
``layer_name='state'`` for plotting.

Namespaces
----------

The API is organized into four namespaces, following scanpy::

    import scgenome

    scgenome.pp.*        # preprocessing: read data, compute QC metrics, filter
    scgenome.tl.*        # tools: cluster, sort, embed, rebin, phylogenetics
    scgenome.pl.*        # plotting: heatmaps, profiles, trees
    scgenome.datasets.*  # bundled example datasets

The split between ``pp`` and ``tl`` is about intent, not mechanism.
Preprocessing brings data into the object and removes what you do not want to
analyse; tools derive interpretable annotations from data that is already
loaded. Plotting functions consume annotations produced by tools and never
modify the object.

In practice, ``pp`` functions are the ones you run once at the start, and ``tl``
functions are the ones you re-run with different parameters while exploring.

A typical workflow
------------------

Analysis reads as a pipeline, because each step returns the object it
annotated::

    import scgenome

    adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    adata.uns['genome'] = 'hg19'

    # Preprocessing: annotate QC metrics, then drop failing cells
    adata = scgenome.pp.calculate_filter_metrics(adata)
    adata = scgenome.pp.filter_cells(adata)

    # Tools: derive annotations
    adata = scgenome.tl.cluster_cells(adata, layer_name='copy', max_k=5)
    adata = scgenome.tl.sort_cells(adata, layer_name='copy')
    adata = scgenome.tl.pca_loadings(adata, layer='copy')
    adata = scgenome.tl.compute_umap(adata, layer_name='copy')

    # Plotting: consume annotations
    scgenome.pl.plot_cell_tcn_matrix_fig(
        adata,
        layer_name='state',
        cell_order_fields=['cluster_id', 'cell_order'],
        annotation_fields=['cluster_id', 'quality'],
    )

Your own data enters the same pipeline through a ``pp`` reader instead of
``datasets``:

- :func:`~scgenome.pp.read_dlp_hmmcopy` — DLP HMMCopy reads and metrics files.
- :func:`~scgenome.pp.read_medicc2_cn` — MEDICC2 copy number profiles.
- :func:`~scgenome.pp.read_snv_genotyping` — SNV genotyping output.
- :func:`~scgenome.pp.create_cn_anndata` — build an object from your own frames.

.. _genome-version:

Genome versions
---------------

Any function that works in genomic coordinates — plotting against position,
rebinning, gene mapping — needs to know chromosome names and lengths. The
reference genome is resolved in this order:

1. An explicit ``genome=`` argument on functions that accept one.
2. ``adata.uns['genome']``, e.g. ``'hg19'``, ``'grch38'`` or ``'mm10'``.
3. The global default set by :func:`scgenome.refgenome.set_genome_version`,
   which is ``'hg19'`` on import.

Because a global default is always in place, coordinate-based functions will not
complain about a missing genome — they will quietly use hg19. Set the version
explicitly on any object that is not hg19. Setting it on the object is usually
the right choice, because the genome is a property of the data rather than of a
particular call::

    adata.uns['genome'] = 'hg19'

Getting this wrong produces a clear error rather than a wrong plot: if
``adata.var['chr']`` contains chromosome names the selected genome does not
know about, plotting raises a ``ValueError`` listing the mismatch.

Function conventions
--------------------

These hold across the library, so you can predict how an unfamiliar function
behaves:

**Mutation and return.** Every ``pp.*`` and ``tl.*`` function that operates on
an object takes the ``AnnData`` as its first argument and returns the same
object, annotated in place. The ``adata = f(adata)`` idiom is a convention for
readability, not a copy. The exceptions are functions that change the shape of
the data and therefore must return something new: :func:`~scgenome.tl.rebin`
and :func:`~scgenome.tl.aggregate_clusters` return new objects, and
:func:`~scgenome.pp.filter_cells` returns a subset view.

**Layer selection.** Functions that read a matrix take a ``layer_name`` (or
``layer``) parameter, where ``None`` means use ``adata.X``. Several analysis
functions also accept a list of layer names, which concatenates those layers
along the bin axis before computing — useful for clustering on allele-specific
copy number, where the signal is split across two layers.

**Documented effects.** Every function's docstring carries ``Reads`` and
``Modifies`` sections naming the exact slots it touches. When in doubt about
what a call will change, check there first.

**Plotting returns handles.** Plotting functions never modify the object. They
return either a matplotlib ``Axes`` or a dict of the objects they created, so
you can adjust the result instead of reimplementing the plot.

**Validation.** Functions check their inputs up front: a ``TypeError`` if the
first argument is not an ``AnnData``, and a ``ValueError`` naming the missing
layers or columns — and the available alternatives — if a required slot is
absent. Optional inputs behave differently: ``calculate_filter_metrics`` logs a
warning and skips a filter whose source column is missing rather than failing.

What each function reads and modifies
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 32 38

   * - Function
     - Reads
     - Modifies
   * - :func:`~scgenome.pp.calculate_filter_metrics`
     - ``layers['copy']``, ``layers['state']``, optional ``obs['quality']``, ``obs['total_mapped_reads_hmmcopy']``, ``obs['is_s_phase']``
     - ``obs['filter_*']``, ``obsm['copy_state_diff']``, ``obsm['copy_state_diff_mean']``
   * - :func:`~scgenome.pp.filter_cells`
     - ``obs`` filter columns
     - Subsets cells
   * - :func:`~scgenome.tl.cluster_cells`
     - ``layers[layer_name]``
     - ``obs['cluster_id']``, ``obs['cluster_size']``, ``uns['clustering']``
   * - :func:`~scgenome.tl.sort_cells`
     - ``layers[layer_name]``
     - ``obs['cell_order']``
   * - :func:`~scgenome.tl.sort_clusters`
     - ``layers[layer_name]``, ``obs[cluster_col]``
     - ``obs['cluster_order']``
   * - :func:`~scgenome.tl.detect_outliers`
     - ``layers[layer_name]``
     - ``obs['is_outlier']``, ``uns['outliers']``
   * - :func:`~scgenome.tl.pca_loadings`
     - ``layers[layer]`` or ``X``
     - ``obsm['X_pca']``, ``varm['PCs']``, ``uns['pca']``
   * - :func:`~scgenome.tl.compute_umap`
     - ``layers[layer_name]``
     - ``obs['UMAP1']``, ``obs['UMAP2']``
   * - :func:`~scgenome.tl.rebin`
     - ``X``, ``layers``, ``var``
     - Returns a new rebinned object
   * - :func:`~scgenome.tl.aggregate_clusters`
     - ``obs[cluster_col]``, ``layers``
     - Returns a new cluster-level object

The object as an analysis log
-----------------------------

Because tools record their parameters alongside their results, an object carries
a record of how it was produced. After clustering,
``adata.uns['clustering']['params']`` holds the method, the selected k, and the
range that was searched; after PCA, ``adata.uns['pca']['params']`` holds the
layer and component count, and ``adata.uns['pca']['variance_ratio']`` the
explained variance.

This makes ``adata.write('analysis.h5ad')`` a reasonably complete checkpoint:
reloading it restores both the annotations and the parameters that generated
them. It is worth inspecting ``adata.uns`` when returning to an object you have
not looked at in a while.

Troubleshooting
---------------

**"mismatching chromosomes" on a plot.** ``adata.var['chr']`` does not match the
active genome. Check ``adata.uns['genome']``, and check whether your chromosome
names carry a ``chr`` prefix that the reference does not.

**A filter silently did nothing.** ``calculate_filter_metrics`` skips filters
whose input column is absent, and ``filter_cells`` skips filters that are not
present in ``obs``; both log warnings. Enable logging
(``logging.basicConfig(level=logging.WARNING)``) to see them, and check which
``filter_*`` columns actually exist.

**``ValueError`` naming a missing layer.** The function needs a layer this
object does not have. The error lists the layers that are present. A common
cause is passing a MEDICC2 or SIGNALS object to a function expecting HMMCopy's
``copy`` and ``state`` layers.

**``TypeError`` on the first argument.** ``pp`` and ``tl`` functions take the
``AnnData`` first. Some readers take filenames instead — check the signature.

**An ``ImplicitModificationWarning`` about views.** You are annotating a view
produced by subsetting, such as the return of ``filter_cells``. It is harmless —
anndata materializes the view — but call ``.copy()`` after subsetting if you
want to avoid it.
