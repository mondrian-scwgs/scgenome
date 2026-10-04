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
ordinary plotting functions, each of which fills one axes and none of which
needs a grid.


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


Legends are collected from the panels into a single strip rather than drawn by
each one. Panels describing the *same* values contribute one legend between
them, so two heatmaps of one palette would appear here once; these two draw
different things, so both appear. `legend_title` overrides what a heatmap's
legend is called, while `name` only addresses the panel.


```python

print(list(g.legends))

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


## Trees and dendrograms

`add_dendrogram` reads the linkage `tl.sort_cells` stored in
`uns['cell_order']`, so it describes the clustering the ordering actually came
from rather than a fresh one. `add_tree` draws a phylogeny instead.

Handed a tree, a grid takes that tree's leaf order as its row order.
`tl.linkage_to_tree` turns the clustering `sort_cells` recorded into a
`Bio.Phylo` tree, so you do not need an external phylogeny to try this.


```python

tree = scgenome.tl.linkage_to_tree(adata)

g = (scgenome.pl.CellGrid(adata, tree=tree, figsize=(12, 5))
     .add_tree()
     .add_heatmap('state', palette='cn', name='Total CN')
     .add_obs_annotation('cluster_id')
     .plot())

```


A tree is already an ordering, so a grid takes a tree or a cell order, never
both. When you want a tree *and* a particular sort, reconcile them yourself and
pass the result. `align_tree_to_order` only rotates internal nodes, which never
changes the topology or a branch length, so it is choosing among the orders the
tree already admits.

Not every order is one of them. A clade whose leaves do not land in one
contiguous block cannot be drawn without its brackets crossing, and a drawing
like that renders cleanly while implying groupings that are not there. So it
raises rather than returning such a tree.

This is not a corner case. Grouping by cluster before ordering within clusters
is the idiom `tl.sort_cells` itself recommends, and it generally does split the
clades of the tree built from that same clustering.


```python

grouped = scgenome.tl.resolve_cell_order(adata, fields=['cluster_id', 'cell_order'])

try:
    scgenome.tl.align_tree_to_order(tree, grouped)
except scgenome.tl.OrderConflict as error:
    print(error)

```


Pass `on_conflict='reorder'` when you would rather have the tree win, with the
requested order tie-breaking within clades, than be refused. The result is a
tree you can hand straight to a grid.


```python

rescued = scgenome.tl.align_tree_to_order(tree, grouped, on_conflict='reorder')

g = (scgenome.pl.CellGrid(adata, tree=rescued, figsize=(12, 5))
     .add_tree()
     .add_heatmap('state', palette='cn', name='Total CN')
     .add_obs_annotation('cluster_id')
     .plot())

```


The dendrogram panel applies the same contiguity rule, since there is no
pre-rotated linkage to hand it. It has no reconciliation mode: it draws, or it
tells you the order would cross its brackets, and says what to do instead.


```python

try:
    (scgenome.pl.CellGrid(adata, cell_order=grouped)
     .add_dendrogram()
     .add_heatmap('state', palette='cn')
     .plot())
except scgenome.tl.OrderConflict as error:
    print(error)
finally:
    plt.close('all')

```


## Gathering a label in the ordering

A clustering admits many leaf orders: swapping the two children of any merge
leaves the clustering itself untouched, so a dendrogram over n cells can be
drawn 2**(n-1) ways. `sort_cells` picks one of them arbitrarily, which is why a
label drawn beside it tends to speckle even though nothing is wrong.

`tl.order_cells_by_groups` picks the member of that family which gathers a
label into as few blocks as the clustering allows. Because it only ever
reorders whole merges, the dendrogram is unchanged and the result is always
drawable against it.


```python

signals = scgenome.datasets.OV081_Signals_reduced()
signals = scgenome.tl.sort_cells(signals, layer_name='copy')

as_sorted = scgenome.tl.resolve_cell_order(signals, fields=['cell_order'])
gathered = scgenome.tl.order_cells_by_groups(signals, 'cluster_id')

```


Drawn side by side, the dendrogram is the same tree in both panels. Only the
order of each merge's two children differs, and with it where each cluster
lands.


```python

fig, axes = plt.subplots(
    ncols=4, figsize=(11, 5), width_ratios=[0.5, 0.08, 0.5, 0.08],
    gridspec_kw=dict(wspace=0.05))

for i, (order, title) in enumerate([(as_sorted, 'as sorted'), (gathered, 'gathered')]):
    scgenome.pl.plot_dendrogram(signals, ax=axes[2 * i], cell_order=order)
    scgenome.pl.plot_obs_annotation(
        signals, 'cluster_id', ax=axes[2 * i + 1], cell_order=order)
    axes[2 * i].set_title(title, fontsize=9)

```


Reordering cannot unmix a merge that genuinely spans two clusters, so the
label does not collapse to one block per cluster. What survives is information:
it marks where the clustering that produced the dendrogram and the labels
disagree.


```python

def blocks(order):
    labels = signals.obs['cluster_id'].astype(str).reindex(order).tolist()
    return 1 + sum(1 for a, b in zip(labels, labels[1:]) if a != b)


print('clusters  :', signals.obs['cluster_id'].nunique())
print('as sorted :', blocks(as_sorted), 'blocks')
print('gathered  :', blocks(gathered), 'blocks')

```


For a tree rather than a dendrogram, `tl.align_tree_to_groups` does the same
thing to a `Bio.Phylo` tree and returns a rotated copy, which matters when the
tree came from a phylogenetics tool rather than from `sort_cells`.


## Driving the panels yourself

`CellGrid` is a convenience, not a requirement. `plot_heatmap`,
`plot_dendrogram` and the rest are ordinary plotting functions that fill
whatever axes you hand them, so you can lay them out with plain matplotlib when
you want a shape the grid does not offer. Pass the same order to each and the
rows still agree.


```python

order = scgenome.tl.resolve_cell_order(adata, fields=['cell_order'])

fig, axes = plt.subplots(
    ncols=2, figsize=(12, 4), width_ratios=[0.3, 1],
    gridspec_kw=dict(wspace=0.02))

scgenome.pl.plot_dendrogram(adata, ax=axes[0], cell_order=order)
scgenome.pl.plot_heatmap(
    adata, layer_name='state', ax=axes[1], palette='cn', cell_order=order)

```


Each describes its legend rather than drawing it, which is what lets a layout
collect legends and drop duplicates. Driving the layout yourself, you draw them
where you like. The result also supports `result['ax']` as well as `result.ax`,
so it stands in for the dictionaries these functions used to return.


```python

result = scgenome.pl.plot_heatmap(
    adata, layer_name='state', ax=plt.subplots(figsize=(8, 3))[1], palette='cn')

print(result.legend.kind, '|', result.legend.title,
      '|', len(result.legend.levels), 'levels')
print(sorted(result.keys()))

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
