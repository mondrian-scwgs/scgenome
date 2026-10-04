---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---


# Composing Panels

A heatmap is rarely the whole picture. You usually want a second matrix beside
it, a tree or dendrogram relating the cells, and a few bars of per cell
metadata, all describing the same rows.

`CellGrid` lays those out against one shared row order. It owns only two
things, allocating axes and collecting legends; everything drawn comes from
`scgenome.pl.panels`, each of which takes an `ax` and draws one thing.


```python

import matplotlib.pyplot as plt
import scgenome

adata = scgenome.datasets.OV2295_HMMCopy_reduced()
adata = scgenome.tl.sort_cells(adata, layer_name='copy')
adata = adata[:, adata.var['gc'] > 0].copy()

```


## Two layers side by side

Panels are added as columns and drawn when you call `plot`. Here the same cells
are shown twice, as integer states on the left and as continuous copy values on
the right, with a dendrogram relating them and annotation bars on the far side.


```python

g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'], figsize=(14, 5))
     .add_dendrogram()
     .add_heatmap('state', palette='cn', name='Total CN')
     .add_heatmap('copy', cmap='viridis', vmin=0, vmax=4, name='Copy')
     .add_obs_annotation(['cluster_id', 'sample_id', 'quality'])
     .plot())

```


The rows line up because the order is resolved once, on the grid, and handed to
every panel. Nothing in the figure recomputes it:


```python

rows = {name: list(g.panels[name].extras['adata'].obs.index)
        for name in ('Total CN', 'Copy')}

print(rows['Total CN'] == rows['Copy'] == list(g.cell_order))

```


The two matrices are also exactly the same width, which they stay no matter how
many annotation bars you add, because the widths are declared and summed once:


```python

widths = [round(float(g.axes[name].get_position().bounds[2]), 6)
          for name in ('Total CN', 'Copy')]

print(widths, '| equal:', widths[0] == widths[1])

```


`plot` returns a `GridResult`. Panels are addressable by name, so a figure can
be adjusted after the fact without rebuilding it.


```python

g.axes['Total CN'].set_title('integer states', fontsize=9)
g.axes['Copy'].set_title('continuous copy', fontsize=9)
g.fig

```


## Annotating bins

`add_obs_annotation` draws a bar per cell, down the side. `add_var_annotation`
draws a bar per bin, along the top of a heatmap. Because each heatmap has its
own bins, a bin annotation belongs to one panel; `on` says which, defaulting to
the most recently added.


```python

g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'], figsize=(12, 5))
     .add_heatmap('state', palette='cn', name='Total CN')
     .add_var_annotation('gc', on='Total CN')
     .add_obs_annotation(['cluster_id', 'quality'])
     .plot())

```


## Comparing two samples

A panel can carry its own `AnnData`. It is reindexed to the grid's row order
and rows for cells it does not have are left blank, so two samples can be put
beside each other without either one moving.


```python

samples = adata.obs['sample_id'].unique()[:2]
first = adata[adata.obs['sample_id'] == samples[0]].copy()
second = adata[adata.obs['sample_id'] == samples[1]].copy()

g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'], figsize=(12, 5))
     .add_heatmap('state', adata=first, palette='cn', name=str(samples[0]))
     .add_heatmap('state', adata=second, palette='cn', name=str(samples[1]))
     .add_obs_annotation('sample_id')
     .plot())

```


Both panels describe the same legend, so the grid draws it once rather than
twice. `legend_title` overrides what a heatmap's legend is called; `name` only
addresses the panel.


```python

print(list(g.legends))

```


## Trees and dendrograms

`add_dendrogram` reads the linkage `tl.sort_cells` stored in
`uns['cell_order']`, so it describes the clustering the ordering actually came
from rather than a fresh one. `add_tree` draws a phylogeny instead.

A tree constrains the row order rather than competing with it, so sort fields
are allowed alongside one and order cells *within* clades.


```python

import io
import Bio.Phylo

leaves = list(adata.obs.sort_values('cell_order').index)
tree = Bio.Phylo.read(io.StringIO('(' + ','.join(f'{c}:1' for c in leaves) + ');'), 'newick')

g = (scgenome.pl.CellGrid(adata, tree=tree, figsize=(12, 5))
     .add_tree()
     .add_heatmap('state', palette='cn', name='Total CN')
     .add_obs_annotation('cluster_id')
     .plot())

```


An order is drawable against a tree or a dendrogram only if every clade's
leaves occupy a contiguous block of rows. If they do not, the brackets would
cross, and drawing anyway would render cleanly while implying groupings that do
not exist. So it raises instead.

This is not a corner case: grouping by cluster before ordering within clusters
is the idiom `tl.sort_cells` itself recommends, and it generally does split the
clades of the dendrogram built from that same clustering.


```python

order = scgenome.tl.resolve_cell_order(adata, fields=['cluster_id', 'cell_order'])

try:
    (scgenome.pl.CellGrid(adata, cell_order=order)
     .add_dendrogram()
     .add_heatmap('state', palette='cn')
     .plot())
except scgenome.tl.OrderConflict as error:
    print(str(error).splitlines()[0])

```


Pass `on_conflict='reorder'` to let the tree drive the rows, with the fields
tie-breaking within clades, instead of refusing:


```python

g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cluster_id', 'cell_order'],
                          tree=tree, on_conflict='reorder', figsize=(12, 5))
     .add_tree()
     .add_heatmap('state', palette='cn', name='Total CN')
     .add_obs_annotation('cluster_id')
     .plot())

```


## Driving the panels yourself

`CellGrid` is a convenience over `scgenome.pl.panels`, not a requirement. Each
panel takes an axes and draws into it, so you can lay them out with plain
matplotlib when you want a shape the grid does not offer. Pass the same order
to each and the rows still agree.


```python

from scgenome.plotting import panels

order = scgenome.tl.resolve_cell_order(adata, fields=['cell_order'])

fig, axes = plt.subplots(
    ncols=2, figsize=(12, 4), width_ratios=[0.3, 1],
    gridspec_kw=dict(wspace=0.02))

panels.dendrogram(adata, axes[0], cell_order=order)
panels.heatmap(adata, axes[1], layer='state', palette='cn', cell_order=order)

```


Panels describe their legend rather than drawing it, which is what lets a
layout collect legends from several panels and drop duplicates. If you are
driving the layout yourself, draw them where you like:


```python

result = panels.heatmap(adata, plt.subplots(figsize=(8, 3))[1],
                        layer='state', palette='cn')

print(result.legend.kind, '|', result.legend.title,
      '|', len(result.legend.levels), 'levels')

```


## The existing figure functions

`plot_cell_tcn_matrix_fig` and its siblings are presets over `CellGrid`. They
return what they always did, plus a `grid` key holding the `GridResult`, so an
existing call can be adjusted rather than rewritten.


```python

g = scgenome.pl.plot_cell_tcn_matrix_fig(
    adata,
    cell_order_fields=['cell_order'],
    annotation_fields=['cluster_id', 'sample_id'])

print(sorted(g.keys()))
print(list(g['grid'].panels))

```
