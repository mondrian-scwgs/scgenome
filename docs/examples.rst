GALLERY
==================

Worked examples built on the bundled example datasets, so each page runs as
written with no external data. For a guided introduction, start with the
:doc:`quickstart` instead.

Plotting
--------

:doc:`notebooks/heatmap`
    Copy number heatmaps: single-axis matrices, full figures with cluster and
    quality annotation bars, cytoband annotation of bins, and rebinning to a
    coarser resolution for display.

:doc:`notebooks/cell_plotting`
    Copy number profiles for individual cells along the genome: colouring by
    integer state, zooming to a chromosome or region, non-linear y-axis
    compression for amplifications, and plotting per-bin covariates such as GC.

:doc:`notebooks/explore_pca`
    PCA over a copy number matrix: cells in component space, and per-bin
    loadings plotted along the genome to identify the regions driving each
    component.

.. toctree::
   :hidden:
   :maxdepth: 3

   notebooks/heatmap
   notebooks/cell_plotting
   notebooks/explore_pca
