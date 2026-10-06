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
scgenome.pl.plot_tcn_heatmap(adata, layer_name='state', obs_order_fields=['cell_order'])
scgenome.pl.plot_tcn_profile(adata, cell_id, value_layer_name='copy', state_layer_name='state')
```

## Heatmaps

Three functions, chosen by what the values mean rather than by an argument.
Each has a `_fig` variant adding annotation bars and a legend.

| Function | For | Colors |
|----------|-----|--------|
| `pl.plot_tcn_heatmap` | Integer total CN states (`layer_name='state'`) | CN palette |
| `pl.plot_ascn_heatmap` | Allele specific states, derived from layers `A`/`B` | Allele state palette |
| `pl.plot_heatmap` | Anything else, including continuous layers like `copy` | Continuous `cmap` |

Palettes map values to colors by equality, so colors never depend on which
values happen to be present. Never use the CN palette on a continuous layer —
it matches by equality and renders almost everything white;
`plot_tcn_heatmap` warns if you try.

These are named for the values they draw rather than for what the rows are,
because the rows of an AnnData are not always cells.

`pl.plot_cell_cn_matrix`/`_fig` and the `raw=` argument are deprecated aliases
kept for compatibility; use the table above instead.

## Drawing primitives

Each of these fills one axes and nothing else, with no figure created and no
layout decided. They are ordinary plotting functions: `pl.CellGrid` arranges
them, but none of them needs it.

| Function | Draws |
|----------|-------|
| `pl.plot_heatmap` | a cell by bin matrix |
| `pl.plot_obs_annotation` | one obs column as a bar down the side |
| `pl.plot_var_annotation` | one var column as a bar along the top |
| `pl.plot_tree` | a phylogeny, as given |
| `pl.plot_dendrogram` | the linkage behind an ordering |

Two conventions make them compose: rows are drawn in `cell_order` with row `i`
at `y == i`, so one order shared between them lines their rows up; and each
*describes* its legend as a `LegendSpec` instead of drawing it, so a caller can
collect legends and drop duplicates. All return a `PanelResult`, which supports
`result['ax']` as well as `result.ax`.

## Composing panels

`pl.CellGrid` lays out several panels against one shared row order. It owns
only two things: allocating axes and collecting legends. Everything drawn comes
from the drawing primitives above.

`scgenome/plotting` is layered, and imports only ever point down:

```
cn_colors, results   palettes; what a drawing function returns
heatmap, phylo       drawing primitives, one axes each
grid                 CellGrid, which arranges them
presets              the *_fig figures, built on CellGrid
```

Keep a new drawing function in `heatmap.py` or `phylo.py` by topic, and
anything that builds a whole figure in `presets.py`. Putting a preset beside a
primitive is what previously forced them apart into a module of their own.

```python
g = (scgenome.pl.CellGrid(adata, obs_order_fields=['cell_order'], figsize=(14, 5))
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

`plot_*_heatmap_fig` are presets over `CellGrid` and return what they always
did, plus a `'grid'` key. `pl.plot_tree_cn` is deprecated in favour of
`CellGrid` with `.add_tree()`.

## Ordering

Row and column order is a value, not a side effect of plotting.
`tl.resolve_cell_order` reads an adata and returns a `pd.Index`; it mutates
nothing. Pass the result to several panels via `obs_order=` and their rows are
guaranteed to line up:

```python
order = scgenome.tl.resolve_cell_order(adata, fields=['cluster_id', 'cell_order'])
scgenome.pl.plot_tcn_heatmap(adata, obs_order=order, ax=axes[0])
scgenome.pl.plot_heatmap(adata, layer_name='copy', obs_order=order, ax=axes[1])
```

`obs_order_fields=` remains as sugar for `resolve_cell_order(adata, fields=...)`.
The two are mutually exclusive.

A tree **is** an order, so plotting functions take a tree or a cell order,
never both. Making a tree agree with some other ordering is an explicit step:

```python
order = scgenome.tl.resolve_cell_order(adata, fields=['quality'])
tree = scgenome.tl.align_tree_to_order(tree, order)   # rotated copy
```

`align_tree_to_order` rotates internal nodes, which never changes the topology
or a branch length, so it picks among the orders the tree already admits. An
order is realizable iff every clade's leaves occupy a contiguous block of rows;
if not it raises `OrderConflict` rather than returning a tree that would imply
groupings which do not exist. Pass `on_conflict='reorder'` to get the closest
tree-realizable order instead, with the requested order tie-breaking within
clades.

A clustering admits many leaf orders: swapping the two children of any merge
leaves the clustering untouched, so a dendrogram over n cells can be drawn
2**(n-1) ways and `sort_cells` picks one arbitrarily. To pick the one that
gathers a label instead:

```python
order = scgenome.tl.order_cells_by_groups(adata, 'cluster_id')
```

It reorders whole merges only, so the result is always drawable against the
dendrogram. `tl.align_tree_to_groups` does the same for a `Bio.Phylo` tree and
returns a rotated copy; `tl.linkage_to_tree` builds such a tree from the stored
linkage. Both find the fewest blocks exactly, by dynamic programming, for a
modest number of groups. What speckle survives is where the clustering and the
label genuinely disagree, since reordering cannot unmix a merge.

The dendrogram panel keeps the same contiguity check, since there is no
pre-rotated linkage to hand it, but it has no reconciliation mode: it either
draws or tells you the order would cross its brackets.

`tl.sort_cells` keeps the linkage in `uns['cell_order'][column]`, so the
dendrogram behind an ordering can be drawn without reclustering.

## Naming

Two rules decide what a function or argument is called.

**Axis plumbing is `obs`/`var`; operations keep their domain noun.** Selecting,
ordering or annotating a row is axis machinery, so it reads `obs_id`,
`obs_ids`, `obs_order`, `obs_order_fields` — and `var_ids`, `var_order` for
bins. But `cluster_cells`, `filter_cells`, `sort_cells` and
`resolve_cell_order` genuinely operate on cells and say so, the same split
scanpy draws between `sc.pp.filter_cells` and `adata.obs_names`. A plot is
named for the values it draws, never for the rows: `plot_tcn_profile` and
`plot_tcn_heatmap` stay true whether the rows are cells, clusters or samples.

**Python names may change; names inside the object may not.** Functions and
arguments carry deprecating aliases (see `scgenome/_deprecate.py`). Keys stored
in an AnnData do not change at all, because they are written into every saved
`.h5ad`: `obs['cell_order']`, `obs['cluster_id']`, `uns['cell_order']`,
`uns['clustering']['params']['cell_ids']`, `uns['cell_order'][k]['layer']` and
`uns['cell_order']['cluster_order']['cluster_col']` all keep their names even
where the matching argument was renamed. `tests/test_renames.py` guards this.

Renaming something that has callers is additive and has no removal date: use
`renamed_arguments` for an argument or a thin warning wrapper for a function,
and pin the old defaults in the wrapper if they differed.

A name with no realistic callers is renamed in place instead, with no alias —
an alias nobody will use is just a second name to maintain. Decide which case
you are in from release history rather than by feel, since a name can look old
in the source and still be days old to anyone installing the package:

```sh
git log --all --reverse --format='%h %ad %s' --date=short \
    -S'def the_name\(' --pickaxe-regex -- '*.py'    # when it was written
git for-each-ref --sort=creatordate --format='%(refname:short)' refs/tags
git show <tag>:path/to/file.py | grep 'def the_name('   # when it first shipped
```

The heatmap and profile names renamed here were all written on 2026-09-25 and
first released in v0.0.23, so they went in place. `plot_cn_profile` and
`plot_cell_cn_matrix`/`_fig` date to 2022 and keep their aliases.
`test_renames.py` asserts both halves: the surviving aliases still work, and
the never-adopted names are gone.

## Function Conventions

- All `tl.*` and `pp.*` functions that operate on adata take `AnnData` as first argument and return the same `AnnData` (mutated in-place).
- Layer selection is via `layer_name` everywhere. `None` means use `.X`. The
  clustering and sorting functions also accept an iterable of layer names.
  `tl.get_obs_data`/`get_var_data` take `layer_names`, a list of layers to
  return side by side.
- Each function's docstring has `Reads` and `Modifies` sections listing exact adata slots.
- Plotting functions return matplotlib axes or dicts of plot elements; they do not mutate adata.

## Key Functions Reference

| Function | Reads | Modifies |
|----------|-------|----------|
| `pp.calculate_filter_metrics` | layers['copy','state'] | obs['filter_*'], obsm['copy_state_diff*'] |
| `pp.filter_cells` | obs[filter columns] | subsets cells |
| `tl.cluster_cells` | layers[layer_name] | obs['cluster_id','cluster_size'], uns['clustering'] |
| `tl.sort_cells` | layers[layer_name] | obs['cell_order'], uns['cell_order']['cell_order'] |
| `tl.sort_clusters` | layers[layer_name], obs[cluster_field] | obs['cluster_order'], uns['cell_order']['cluster_order'] |
| `tl.resolve_cell_order` | obs[fields], tree | nothing, returns a `pd.Index` |
| `tl.resolve_bin_order` | var['chr','start'] | nothing, returns a `pd.Index` |
| `tl.align_tree_to_order` | tree | nothing, returns a rotated copy |
| `tl.align_tree_to_groups` | tree, group labels | nothing, returns a rotated copy |
| `tl.order_cells_by_groups` | uns['cell_order'], obs[groups] | nothing, returns a `pd.Index` |
| `tl.linkage_to_tree` | uns['cell_order'] | nothing, returns a `Bio.Phylo` tree |
| `tl.linkage_order_conflict` | linkage | nothing, returns the split merge or None |
| `tl.detect_outliers` | layers[layer_name] | obs['is_outlier'], uns['outliers'] |
| `tl.pca_loadings` | layers[layer_name] or X | obsm['X_pca'], varm['PCs'], uns['pca'] |
| `tl.compute_umap` | layers[layer_name] | obs['UMAP1','UMAP2'] |
| `tl.rebin` | X, layers, var | returns new rebinned AnnData |
| `tl.aggregate_clusters` | obs[cluster_field], layers | returns new cluster-level AnnData |

## Available Datasets

- `scgenome.datasets.OV2295_HMMCopy_reduced()` — HMMCopy CN data with layers['copy','state'] and QC metrics
- `scgenome.datasets.OV_051_Medicc2_reduced()` — Medicc2 CN data
- `scgenome.datasets.OV081_Signals_reduced()` — signals allele specific CN, adds layers['A','B','BAF','alleleA','alleleB','totalcounts']; the only bundled dataset supporting `pl.plot_ascn_profile` / `pl.plot_pseudobulk_ascn`
- `scgenome.datasets.OV081_breakpoints()` — DataFrame of somatic rearrangements matching OV081, for `pl.plot_rearrangement_arcs`

Regenerate the OV081 files with `scripts/make_OV081_signals_reduced.py` (needs the
full source data, which is not in the repo).

## Error Handling

Functions validate inputs and raise clear errors:
- `TypeError` if first argument is not AnnData
- `ValueError` listing missing layers/obs columns with available alternatives
