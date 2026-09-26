# scgenome — Single-Cell Whole Genome Sequencing Analysis

## Data Model

All data is stored in `AnnData` objects from the `anndata` library.

| Slot | Contents |
|------|----------|
| `adata.X` | Primary matrix (typically raw reads) |
| `adata.layers['copy']` | Continuous copy number values |
| `adata.layers['state']` | Integer copy number states |
| `adata.obs` | Per-cell metadata (quality, cluster_id, cell_order, etc.) |
| `adata.var` | Per-bin metadata with columns `chr`, `start`, `end` |
| `adata.uns` | Unstructured data (clustering params, genome version, PCA results) |
| `adata.obsm` | Multi-dimensional annotations (X_pca, copy_state_diff) |
| `adata.varm` | Variable-dimensional annotations (PCs) |

## Genome Version

Genome version is resolved in this priority:
1. Explicit `genome=` parameter on functions that need it
2. `adata.uns['genome']` (e.g., `'hg19'`, `'grch38'`, `'mm10'`)
3. Global default via `scgenome.refgenome.set_genome_version()`

Set genome on your adata: `adata.uns['genome'] = 'hg19'`

## API Structure

```python
import scgenome

scgenome.pp.*   # preprocessing: load, filter, transform
scgenome.tl.*   # tools: cluster, sort, PCA, UMAP, rebin, phylo
scgenome.pl.*   # plotting: heatmaps, profiles, trees
scgenome.datasets.*  # example datasets
```

## Common Workflow

```python
import scgenome

# Load data
adata = scgenome.datasets.OV2295_HMMCopy_reduced()
# Or: adata = scgenome.pp.read_dlp_hmmcopy(reads_file, metrics_file)
# Or: adata = scgenome.pp.read_medicc2_cn(cn_profiles_file)

# QC and filter
adata = scgenome.pp.calculate_filter_metrics(adata)
adata = scgenome.pp.filter_cells(adata)

# Cluster
adata = scgenome.tl.cluster_cells(adata, layer_name='copy')
# Result: adata.obs['cluster_id'], adata.obs['cluster_size']

# Sort for visualization
adata = scgenome.tl.sort_cells(adata, layer_name='copy')
# Result: adata.obs['cell_order']

# Dimensionality reduction
adata = scgenome.tl.pca_loadings(adata, layer='copy')
# Result: adata.obsm['X_pca'], adata.varm['PCs'], adata.uns['pca']

adata = scgenome.tl.compute_umap(adata, layer_name='copy')
# Result: adata.obs['UMAP1'], adata.obs['UMAP2']

# Plot
scgenome.pl.plot_cell_tcn_matrix(adata, layer_name='state', cell_order_fields=['cell_order'])
scgenome.pl.plot_cn_profile(adata, cell_id, value_layer_name='copy', state_layer_name='state')
```

## Heatmaps

Three functions, chosen by what the values mean rather than by an argument.
Each has a `_fig` variant adding annotation bars and a legend.

| Function | For | Colors |
|----------|-----|--------|
| `pl.plot_cell_tcn_matrix` | Integer total CN states (`layer_name='state'`) | CN palette |
| `pl.plot_cell_ascn_matrix` | Allele specific states, derived from layers `A`/`B` | Allele state palette |
| `pl.plot_cell_matrix` | Anything else, including continuous layers like `copy` | Continuous `cmap` |

Palettes map values to colors by equality, so colors never depend on which
values happen to be present. Never use the CN palette on a continuous layer —
it matches by equality and renders almost everything white;
`plot_cell_tcn_matrix` warns if you try.

`pl.plot_cell_cn_matrix`/`_fig` and the `raw=` argument are deprecated aliases
kept for compatibility; use the table above instead.

## Composing panels

`pl.CellGrid` lays out several panels against one shared row order. It owns
only two things: allocating axes and collecting legends. Everything drawn comes
from `pl.panels.*`, each of which takes an `ax` and draws one thing.

```python
g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'], figsize=(14, 5))
     .add_dendrogram()
     .add_heatmap('state', palette='cn', name='Total CN')
     .add_heatmap('copy', cmap='viridis', vmin=0, vmax=4, name='Copy')
     .add_obs_annotation(['cluster_id', 'sample_id'])
     .add_var_annotation('gc')
     .plot())

g.fig, g.axes['Total CN'], g.panels['Copy'], g.legends, g.cell_order
```

- The row order is resolved once and handed to every panel, so a tree, a
  dendrogram and any number of heatmaps are guaranteed to agree.
- Panel widths are declared and summed once, so adding an annotation bar cannot
  resize the matrices beside it.
- Panels *describe* their legend (`LegendSpec`) rather than drawing it, so
  panels showing the same values collapse to one legend. `legend_title=`
  overrides a heatmap's, `name=` only addresses the panel.
- `add_heatmap(adata=other)` draws a different AnnData against the same order,
  blanking rows for cells it does not have — that is how two samples are
  compared side by side.

`add_dendrogram()` reads the linkage `tl.sort_cells` stored. It refuses an order
that would cross its brackets, by the same contiguity rule trees use.

`plot_cell_*_matrix_fig` are presets over `CellGrid` and return what they always
did, plus a `'grid'` key. `pl.plot_tree_cn` is deprecated in favour of
`CellGrid` with `.add_tree()`.

## Ordering

Row and column order is a value, not a side effect of plotting.
`tl.resolve_cell_order` reads an adata and returns a `pd.Index`; it mutates
nothing. Pass the result to several panels via `cell_order=` and their rows are
guaranteed to line up:

```python
order = scgenome.tl.resolve_cell_order(adata, fields=['cluster_id', 'cell_order'])
scgenome.pl.plot_cell_tcn_matrix(adata, cell_order=order, ax=axes[0])
scgenome.pl.plot_cell_matrix(adata, layer_name='copy', cell_order=order, ax=axes[1])
```

`cell_order_fields=` remains as sugar for `resolve_cell_order(adata, fields=...)`.
The two are mutually exclusive.

A tree **constrains** the order rather than competing with it, so sort fields
are allowed alongside one and order cells *within* clades. An order is drawable
against a tree iff every clade's leaves occupy a contiguous block of rows; if
not, `OrderConflict` is raised rather than rendering a plot that implies
groupings which do not exist. Pass `on_conflict='reorder'` to let the tree drive
row order, with the fields tie-breaking within clades.

`tl.sort_cells` keeps the linkage in `uns['cell_order'][column]`, so the
dendrogram behind an ordering can be drawn without reclustering.

## Function Conventions

- All `tl.*` and `pp.*` functions that operate on adata take `AnnData` as first argument and return the same `AnnData` (mutated in-place).
- Layer selection is via `layer_name` (or `layer`) parameter. `None` means use `.X`.
- Each function's docstring has `Reads` and `Modifies` sections listing exact adata slots.
- Plotting functions return matplotlib axes or dicts of plot elements; they do not mutate adata.

## Key Functions Reference

| Function | Reads | Modifies |
|----------|-------|----------|
| `pp.calculate_filter_metrics` | layers['copy','state'] | obs['filter_*'], obsm['copy_state_diff*'] |
| `pp.filter_cells` | obs[filter columns] | subsets cells |
| `tl.cluster_cells` | layers[layer_name] | obs['cluster_id','cluster_size'], uns['clustering'] |
| `tl.sort_cells` | layers[layer_name] | obs['cell_order'], uns['cell_order']['cell_order'] |
| `tl.sort_clusters` | layers[layer_name], obs[cluster_col] | obs['cluster_order'], uns['cell_order']['cluster_order'] |
| `tl.resolve_cell_order` | obs[fields], tree | nothing, returns a `pd.Index` |
| `tl.resolve_bin_order` | var['chr','start'] | nothing, returns a `pd.Index` |
| `tl.align_tree_to_order` | tree | nothing, returns a rotated copy |
| `tl.linkage_order_conflict` | linkage | nothing, returns the split merge or None |
| `tl.detect_outliers` | layers[layer_name] | obs['is_outlier'], uns['outliers'] |
| `tl.pca_loadings` | layers[layer] or X | obsm['X_pca'], varm['PCs'], uns['pca'] |
| `tl.compute_umap` | layers[layer_name] | obs['UMAP1','UMAP2'] |
| `tl.rebin` | X, layers, var | returns new rebinned AnnData |
| `tl.aggregate_clusters` | obs[cluster_col], layers | returns new cluster-level AnnData |

## Available Datasets

- `scgenome.datasets.OV2295_HMMCopy_reduced()` — HMMCopy CN data with layers['copy','state'] and QC metrics
- `scgenome.datasets.OV_051_Medicc2_reduced()` — Medicc2 CN data
- `scgenome.datasets.OV081_Signals_reduced()` — signals allele specific CN, adds layers['A','B','BAF','alleleA','alleleB','totalcounts']; the only bundled dataset supporting `pl.plot_cell_ascn` / `pl.plot_pseudobulk_ascn`
- `scgenome.datasets.OV081_breakpoints()` — DataFrame of somatic rearrangements matching OV081, for `pl.plot_rearrangement_arcs`

Regenerate the OV081 files with `scripts/make_OV081_signals_reduced.py` (needs the
full source data, which is not in the repo).

## Error Handling

Functions validate inputs and raise clear errors:
- `TypeError` if first argument is not AnnData
- `ValueError` listing missing layers/obs columns with available alternatives
