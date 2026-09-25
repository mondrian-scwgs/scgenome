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
    compression for amplifications, segment plots, and plotting per-bin
    covariates such as GC.

:doc:`notebooks/allele_specific_plotting`
    B allele frequency and allele specific states, distinguishing balanced
    changes from loss of heterozygosity, and aggregating cells into pseudobulk
    total and allele specific profiles.

:doc:`notebooks/multi_region_plotting`
    Laying several disjoint genomic regions out on one axis with
    ``RegionMapper``, so that plots can focus on the chromosomes and intervals
    that matter.

:doc:`notebooks/rearrangements`
    Structural variant breakpoints drawn as arcs over a copy number profile,
    within one chromosome and between several regions.

:doc:`notebooks/explore_pca`
    PCA over a copy number matrix: cells in component space, and per-bin
    loadings plotted along the genome to identify the regions driving each
    component.

.. toctree::
   :hidden:
   :maxdepth: 3

   notebooks/heatmap
   notebooks/cell_plotting
   notebooks/allele_specific_plotting
   notebooks/multi_region_plotting
   notebooks/rearrangements
   notebooks/explore_pca
