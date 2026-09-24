---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: venv (3.9.19)
    language: python
    name: python3
---

# Explore PCA

`scgenome.tl.pca_loadings` runs PCA over a cell-by-bin matrix and annotates the
`AnnData` in place. Cell coordinates go to `obsm['X_pca']`, the per-bin loadings
go to `varm['PCs']`, and the fit metadata goes to `uns['pca']`.

Plotting the loadings along the genome shows *which regions* drive each
component, which is often more informative than the cell scatter.

```python

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

import scgenome

adata = scgenome.datasets.OV2295_HMMCopy_reduced()
adata.uns['genome'] = 'hg19'

```

## Run PCA

We decompose the continuous `copy` layer. With 25 cells in this reduced dataset
a handful of components captures nearly all the structure.

```python

adata = scgenome.tl.pca_loadings(adata, layer='copy', n_components=4)

print(adata.obsm['X_pca'].shape)
print(adata.varm['PCs'].shape)
print(adata.uns['pca']['variance_ratio'])

```

## Cells in PCA space

`obsm['X_pca']` holds one row per cell. Colouring by `cluster_id` shows how well
the components separate the clusters.

```python

pca_df = pd.DataFrame(
    adata.obsm['X_pca'][:, :2],
    columns=['PC1', 'PC2'],
    index=adata.obs.index)
pca_df['cluster_id'] = adata.obs['cluster_id'].values

fig, ax = plt.subplots(figsize=(4, 4))
for cluster_id, group in pca_df.groupby('cluster_id', observed=True):
    ax.scatter(group['PC1'], group['PC2'], label=cluster_id, s=20)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(title='cluster_id', fontsize='small', bbox_to_anchor=(1, 1))

```

## Loadings along the genome

`varm['PCs']` is bins by components, so transposing it gives a matrix with the
same bins as `adata` but one row per component. Wrapping that in an `AnnData`
lets the standard profile plotting functions treat each component as if it were
a cell.

```python

n_components = adata.varm['PCs'].shape[1]

loadings = ad.AnnData(
    np.ascontiguousarray(adata.varm['PCs'].T),
    obs=pd.DataFrame(index=[f'PC{i + 1}' for i in range(n_components)]),
    var=adata.var)
loadings.uns['genome'] = adata.uns['genome']

loadings

```

Each component is now an `obs` entry that `scgenome.pl.plot_cn_profile` can
plot. Passing `value_layer_name=None` uses `X`, which here holds the loadings.

```python

variance_ratio = adata.uns['pca']['variance_ratio']

for i in range(n_components):
    pc = f'PC{i + 1}'

    plt.figure(figsize=(20, 2))
    ax = scgenome.pl.plot_cn_profile(
        loadings, pc,
        value_layer_name=None)
    ax.set_ylabel(pc)
    ax.set_title(f'{pc} ({variance_ratio[i]:.1%} of variance)')

```

Contiguous runs of same-signed loading mark regions whose copy number covaries
across cells — typically the large events that distinguish the clusters seen in
the PCA scatter above.
