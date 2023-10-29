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

```python

plt.figure(figsize=(10, 2))
scgenome.pl.plot_profile(adata[:, adata.var['gc'] > 0].var, 'gc')

```

```python

```
