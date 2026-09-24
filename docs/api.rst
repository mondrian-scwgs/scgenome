.. module:: scgenome
.. automodule:: scgenome
   :noindex:

API
===================================

Import scgenome as::

   import scgenome

Preprocessing: `pp`
-------------------

.. module:: scgenome.pp
.. currentmodule:: scgenome

Data loading and pre-processing functionality.

Data loading
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pp.read_dlp_hmmcopy
   pp.read_medicc2_cn
   pp.read_snv_genotyping
   pp.convert_dlp_hmmcopy
   pp.convert_dlp_signals
   pp.create_cn_anndata

Filtering
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pp.calculate_filter_metrics
   pp.filter_cells


Tools: `tl`
-----------

.. module:: scgenome.tl
.. currentmodule:: scgenome

Any transformation of the data matrix that is not *preprocessing*. In contrast to a *preprocessing* function, a *tool* usually adds an easily interpretable annotation to the data matrix, which can then be visualized with a corresponding plotting function.

Clustering and ordering
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.cluster_cells
   tl.detect_outliers
   tl.sort_cells
   tl.sort_clusters
   tl.aggregate_clusters
   tl.aggregate_clusters_hmmcopy

Embeddings
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.compute_umap
   tl.pca_loadings

Generating and transforming binned data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.create_bins
   tl.rebin
   tl.rebin_regular
   tl.weighted_mean
   tl.bin_width_weighted_mean
   tl.count_gc
   tl.mean_from_bigwig
   tl.add_cyto_giemsa_stain

Gene regions
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.read_ensemble_genes_gtf
   tl.aggregate_genes
   tl.get_gene_cn

Phylogenetics
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.prune_leaves
   tl.aggregate_tree_branches
   tl.align_cn_tree

Anndata manipulation
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tl.ad_concat_cells
   tl.get_obs_data
   tl.get_var_data


Plotting: `pl`
--------------

.. module:: scgenome.pl
.. currentmodule:: scgenome

The plotting module :mod:`scgenome.pl` largely parallels the ``tl.*`` and a few of the ``pp.*`` functions.
For most tools and for some preprocessing functions, you'll find a plotting function with the same name.

Plotting functions never modify the object they are given. They return either a
matplotlib ``Axes`` or a dict of the plot elements they created, so the result
can be customized rather than rebuilt.

Copy number heatmaps
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.plot_cell_cn_matrix
   pl.plot_cell_cn_matrix_fig
   pl.plot_cn_rect

Copy number profiles
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.plot_cn_profile
   pl.plot_profile
   pl.plot_cell_tcn
   pl.plot_cell_ascn
   pl.plot_pseudobulk_tcn
   pl.plot_pseudobulk_ascn

Rearrangements
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.plot_rearrangement_arcs

Quality control
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.plot_gc_reads

Phylogenetics
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.plot_tree_cn

Colors and legends
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pl.cn_legend
   pl.add_allele_state_layer

Genomic regions
~~~~~~~~~~~~~~~

Profile plotting functions accept a ``region_mapper`` argument for laying out
several disjoint genomic regions on one axis. The helper classes
``scgenome.pl.GenomicRegion`` and ``scgenome.pl.RegionMapper`` construct these;
see their docstrings for usage.


Datasets: `datasets`
--------------------

.. module:: scgenome.datasets
.. currentmodule:: scgenome

Small bundled datasets for experimentation and for the examples in these docs.

.. autosummary::
   :toctree: generated/

   datasets.OV2295_HMMCopy_reduced
   datasets.OV_051_Medicc2_reduced


Reference genome: `refgenome`
-----------------------------

.. currentmodule:: scgenome

Chromosome names and lengths for coordinate-based functions. See
:ref:`genome-version` for how the active genome is resolved.

.. autosummary::
   :toctree: generated/

   refgenome.set_genome_version
   refgenome.get_genome_info
