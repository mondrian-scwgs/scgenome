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

Handed a tree, a grid takes that tree's leaf order as its row order. The tree
below is built from the stored linkage, which is also how you turn scgenome's
clustering into a `Bio.Phylo` tree.


```python

import io
import Bio.Phylo
import scipy.cluster.hierarchy as sch

record = adata.uns['cell_order']['cell_order']
ids = list(record['ids'])


def to_newick(node):
    if node.is_leaf():
        return f'{ids[node.id]}:{node.dist:.4f}'
    return f'({to_newick(node.left)},{to_newick(node.right)}):{node.dist:.4f}'


tree = Bio.Phylo.read(
    io.StringIO(to_newick(sch.to_tree(record['linkage'])) + ';'), 'newick')

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


## Rotating a tree to match a label

A tree arrives with an arbitrary child order. Newick does not canonicalise one,
and swapping any node's children leaves the topology and every branch length
untouched, so a tree that is correct can still be drawn in an order that makes
a per cell label look like noise.

This is easiest to see with a label the tree broadly agrees with but was not
built from. Here the tree is complete linkage on copy number and the groups
come from a ward clustering of the same cells, which is the ordinary situation
of having clusters from one method and a tree from another.


```python

import io
import random
import Bio.Phylo
import numpy as np
import scipy.cluster.hierarchy as sch

import scgenome.preprocessing.transform

signals = scgenome.datasets.OV081_Signals_reduced()
signals = scgenome.tl.sort_cells(signals, layer_name='copy')

record = signals.uns['cell_order']['cell_order']
ids = list(record['ids'])


def to_newick(node):
    if node.is_leaf():
        return f'{ids[node.id]}:{node.dist:.4f}'
    return f'({to_newick(node.left)},{to_newick(node.right)}):{node.dist:.4f}'


phylo = Bio.Phylo.read(
    io.StringIO(to_newick(sch.to_tree(record['linkage'])) + ';'), 'newick')

# a second clustering of the same cells, standing in for groups you already have
values = scgenome.preprocessing.transform.fill_missing(np.array(signals.layers['copy']))
signals.obs['group'] = [
    str(g) for g in sch.fcluster(sch.linkage(values, method='ward'), 6, criterion='maxclust')]

```


Shuffling the children simulates a tree as read from a file, where the child
order carries no meaning:


```python

def shuffle_children(tree, seed):
    tree = scgenome.tl.align_tree_to_order(tree, scgenome.tl.tree_leaf_order(tree))
    rng = random.Random(seed)
    stack = [tree.root]
    while stack:
        clade = stack.pop()
        rng.shuffle(clade.clades)
        stack.extend(clade.clades)
    return tree


as_read = shuffle_children(phylo, 3)
gathered = scgenome.tl.align_tree_to_groups(as_read, signals.obs['group'])

```


Drawn side by side, with the group as an annotation bar, the difference is the
whole point. Same tree, same topology, same branch lengths; only the order of
each node's children differs.


```python

def blocks(order):
    labels = signals.obs['group'].reindex(order).tolist()
    return 1 + sum(1 for a, b in zip(labels, labels[1:]) if a != b)


fig, axes = plt.subplots(
    ncols=4, figsize=(11, 5), width_ratios=[0.5, 0.08, 0.5, 0.08],
    gridspec_kw=dict(wspace=0.05))

for i, (tree_, title) in enumerate([(as_read, 'as read'), (gathered, 'rotated to the groups')]):
    order = scgenome.tl.tree_leaf_order(tree_)
    scgenome.pl.plot_tree(tree_, ax=axes[2 * i])
    scgenome.pl.plot_obs_annotation(signals, 'group', ax=axes[2 * i + 1], cell_order=order)
    axes[2 * i].set_title(f'{title} — {blocks(order)} blocks', fontsize=9)

```


Rotation can only gather a label as far as the tree's own structure allows. A
clade that genuinely mixes two groups cannot be unmixed by reordering it, which
is why the count above does not fall to one block per group. The speckle that
survives is information: it is telling you where the tree and the label
disagree.


```python

print('groups           :', signals.obs['group'].nunique())
print('as read          :', blocks(scgenome.tl.tree_leaf_order(as_read)), 'blocks')
print('rotated to groups:', blocks(scgenome.tl.tree_leaf_order(gathered)), 'blocks')

```


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
