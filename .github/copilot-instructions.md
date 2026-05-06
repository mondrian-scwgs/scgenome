---
applyTo: "**"
---

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
scgenome.pl.plot_cell_cn_matrix(adata, layer_name='state', cell_order_fields=['cell_order'])
scgenome.pl.plot_cn_profile(adata, cell_id, value_layer_name='copy', state_layer_name='state')
```

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
| `tl.sort_cells` | layers[layer_name] | obs['cell_order'] |
| `tl.sort_clusters` | layers[layer_name], obs[cluster_col] | obs['cluster_order'] |
| `tl.detect_outliers` | layers[layer_name] | obs['is_outlier'], uns['outliers'] |
| `tl.pca_loadings` | layers[layer] or X | obsm['X_pca'], varm['PCs'], uns['pca'] |
| `tl.compute_umap` | layers[layer_name] | obs['UMAP1','UMAP2'] |
| `tl.rebin` | X, layers, var | returns new rebinned AnnData |
| `tl.aggregate_clusters` | obs[cluster_col], layers | returns new cluster-level AnnData |

## Available Datasets

- `scgenome.datasets.OV2295_HMMCopy_reduced()` — HMMCopy CN data with layers['copy','state'] and QC metrics
- `scgenome.datasets.OV_051_Medicc2_reduced()` — Medicc2 CN data

## Error Handling

Functions validate inputs and raise clear errors:
- `TypeError` if first argument is not AnnData
- `ValueError` listing missing layers/obs columns with available alternatives
