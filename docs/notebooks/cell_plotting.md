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


# Cell Copy Number Plots

scgenome provides functionality for plotting the copy number of individual cells or clones with chromosomes on the x axis.


```python

import scgenome
import matplotlib.pyplot as plt

adata = scgenome.datasets.OV2295_HMMCopy_reduced()

```


## HMMCopy Copy Number Plot

The `scgenome.pl.plot_cn_profile` plots copy number as a scatter plot with copy number on the y axis and the genome on the x axis.  Scatter points can be colored using a standard copy number color palette.  The `value_layer_name` arg specifies which layer to use as the y value and `state_layer_name` which layer to use as the integer copy number state.  The `obs_id` specifies the `obs` index value to plot.


```python

cell_id = 'SA922-A90554B-R27-C43'

plt.figure(figsize=(10, 2))
scgenome.pl.plot_cn_profile(
    adata, cell_id,
    value_layer_name='copy',
    state_layer_name='state')

scgenome.pl.cn_legend(plt.gca())

```


Specific chromosomes can also be plotted using the `chromosome` keyword arg.


```python

plt.figure(figsize=(10, 2))
scgenome.pl.plot_cn_profile(
    adata, cell_id,
    value_layer_name='copy',
    state_layer_name='state',
    chromosome='2',
    squashy=True)

scgenome.pl.cn_legend(plt.gca())

```


Restrict to a specific region using the `start` and `end` arguments.


```python

plt.figure(figsize=(10, 2))
scgenome.pl.plot_cn_profile(
    adata, cell_id,
    value_layer_name='copy',
    state_layer_name='state',
    chromosome='2',
    start=80000000,
    end=200000000,
    squashy=True)

scgenome.pl.cn_legend(plt.gca())

```


Often the majority of the copy number information is in the range of 0-7 copies.  However, limiting the y axis to 0-7 would now show high level amplifications.  Set the `squashy` kwarg to `True` to use a non-linear scaling to compress the y-axis values, high-lighting the 0-7 copies range while still showing high level amplifications.


```python

plt.figure(figsize=(10, 2))
scgenome.pl.plot_cn_profile(
    adata, cell_id,
    value_layer_name='copy',
    state_layer_name='state',
    squashy=True)

scgenome.pl.cn_legend(plt.gca())

```


Set `rawy=True` to show raw copy number values or reads.  Below we set `value_layer_name=None` to use `X` for the y values, plotting the raw read counts that are stored in `X`.


```python

plt.figure(figsize=(10, 2))
scgenome.pl.plot_cn_profile(
    adata, cell_id,
    value_layer_name=None)

```


## A Shorthand for the Common Case

Naming the layers on every call gets repetitive when they follow the usual
convention. `scgenome.pl.plot_cell_tcn` is the same plot with `copy` and
`state` as defaults, `squashy` already on, and its own legend, so the standard
total copy number profile is a single call.


```python

plt.figure(figsize=(10, 2))
scgenome.pl.plot_cell_tcn(adata, cell_id)

```


It takes `chromosome`, `start` and `end` in the same way.


```python

plt.figure(figsize=(10, 2))
scgenome.pl.plot_cell_tcn(adata, cell_id, chromosome='2')

```


It also accepts a `region_mapper`, which lays several disjoint regions out on
one axis. See {doc}`multi_region_plotting`.


## Copy Number as Segments

A scatter plot shows the spread of the data within each state, which is what
you want when judging how well the states fit. When the integer states are the
point, `scgenome.pl.plot_cn_rect` draws them as filled blocks instead, which
reads more clearly at small figure sizes.


```python

fig, ax = plt.subplots(figsize=(10, 1.5))
scgenome.pl.plot_cn_rect(adata, obs_id=cell_id, ax=ax)

```


## Per Bin Covariates

`scgenome.pl.plot_profile` is the low level function underneath these plots. It
takes a dataframe with `chr`, `start` and `end` columns rather than an
`AnnData`, so it can plot any per-bin annotation from `adata.var` on the same
genome axis. Here it is used for GC content.


```python

plt.figure(figsize=(10, 2))
scgenome.pl.plot_profile(adata[:, adata.var['gc'] > 0].var, 'gc')

```


GC content is worth plotting against read depth as well, since GC bias is a
common artefact in whole genome amplification.
`scgenome.pl.plot_gc_reads` shows read count against GC content for one cell.


```python

fig, ax = plt.subplots(figsize=(4, 3))
scgenome.pl.plot_gc_reads(adata, cell_id, s=2, alpha=0.3, linewidth=0, ax=ax)

```


## Where to Next

- {doc}`allele_specific_plotting` for BAF and allele specific states, which
  need a dataset carrying allele layers.
- {doc}`multi_region_plotting` for plotting selected regions side by side.
- {doc}`rearrangements` for overlaying structural variant breakpoints.
- {doc}`heatmap` for many cells at once rather than one at a time.
