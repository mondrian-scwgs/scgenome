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


# Allele Specific Copy Number Plots

Total copy number tells you how many copies of a region a cell has, but not
which parental haplotype those copies came from. A region at 2 copies could be
one copy of each parent, or two copies of one parent and a loss of the other.
The second case is loss of heterozygosity, and it is invisible to total copy
number alone.

Allele specific data resolves this by tracking the two haplotypes separately.
scgenome plots it as B allele frequency (BAF) along the genome, coloured by
allele specific state.


```python

import matplotlib.pyplot as plt

import scgenome

adata = scgenome.datasets.OV081_Signals_reduced()
adata

```


This dataset is 100 cells from a high grade serous ovarian tumour, processed
with signals, which is why it carries allele layers that the HMMCopy example
dataset does not:

| Layer | Contents |
|-------|----------|
| `copy` | Total copy number, continuous |
| `state` | Total copy number, integer state |
| `A`, `B` | Allele specific integer copy number of each haplotype |
| `BAF` | B allele frequency |
| `alleleA`, `alleleB`, `totalcounts` | Raw allele counts, used for pseudobulk aggregation |

The cells fall into two populations, which is worth knowing before looking at
any of the plots below.


```python

adata.obs.groupby('cluster_id', observed=True).agg(
    n_cells=('quality', 'size'),
    mean_copy=('mean_copy', 'mean'),
    ploidy=('ploidy', 'mean'),
    fraction_loh=('fraction_loh', 'mean'),
    n_wgd=('n_wgd', 'mean'))

```


Clusters with `n_wgd` of 1 have undergone a whole genome doubling and sit near
4 copies; the rest are near 2. Both carry a similar fraction of the genome in
LOH, which is the signature of an ancestral event shared by all of them.

We pick a doubled cell to plot, since its profile has more structure to look
at.


```python

cell_id = adata.obs.query('n_wgd == 1').sort_values('quality', ascending=False).index[0]
cell_id

```


## Allele Specific Profile for One Cell

`scgenome.pl.plot_cell_ascn` plots BAF on the y axis, with each bin coloured by
its allele specific state. It reads the `BAF`, `A`, `B`, `copy` and `state`
layers.

The five states describe the relationship between the two haplotypes:
*Balanced* means equal copies of each, *A-Gained* and *B-Gained* mean more of
one than the other, and *A-Hom* and *B-Hom* mean one haplotype is lost
entirely, which is LOH.


```python

fig, ax = plt.subplots(figsize=(10, 2))
scgenome.pl.plot_cell_ascn(adata, cell_id, ax=ax)

```


Balanced bins sit at a BAF of 0.5. Bins where one haplotype is lost collapse to
0 or 1, and the intermediate bands are imbalanced states where one haplotype
outnumbers the other without being lost.


## Reading Total and Allele Specific Copy Number Together

Neither view is complete on its own. Stacking total copy number above BAF, with
a shared x axis, shows which changes in total copy number were allele balanced
and which were not.


```python

fig, axes = plt.subplots(nrows=2, figsize=(10, 4), sharex=True)

scgenome.pl.plot_cell_tcn(adata, cell_id, ax=axes[0], s=4)
scgenome.pl.plot_cell_ascn(adata, cell_id, ax=axes[1], s=4)

```


Regions that are flat in the top panel but sitting at BAF 0 or 1 in the bottom
panel are copy neutral LOH: the cell has the usual number of copies, but they
all descend from one parent.


## Allele Specific Heatmap

A profile shows one cell in detail. To see whether an event is shared across
the population or private to a few cells, plot every cell at once.
`scgenome.pl.plot_cell_ascn_matrix_fig` draws one row per cell, with each bin
coloured by the same five allele specific states used in the profiles above.


```python

fig = plt.figure(figsize=(6, 8), dpi=150)

g = scgenome.pl.plot_cell_ascn_matrix_fig(
    adata,
    cell_order_fields=['cell_order'],
    fig=fig)

```


There is no preparation step. The function derives the allele state from the
`A` and `B` layers itself, and if `var` carries a `has_allele_cn` column it
restricts to those bins, since bins without an allele specific call have no
state to colour.

This matters more than it sounds. Allele specific callers leave a substantial
fraction of bins without a call, and there is no state that honestly represents
them. They are excluded here rather than being folded into one of the five
states, where they would read as a confident result.

Everything `plot_cell_cn_matrix_fig` accepts works here too, including cell
ordering, a phylogeny, and annotation bars.


```python

fig = plt.figure(figsize=(6, 8), dpi=150)

g = scgenome.pl.plot_cell_ascn_matrix_fig(
    adata,
    cell_order_fields=['cluster_id', 'cell_order'],
    annotation_fields=['cluster_id', 'n_wgd'],
    fig=fig)

```


Ordering by `cluster_id` groups related cells together, and the annotation bars
on the right make the cluster and whole genome doubling status of each row
readable alongside it. Vertical bands of colour running through every cell are
ancestral events; patches confined to one block of rows are specific to a
subpopulation.

The colours come from a discrete palette rather than a colormap, so a given
state is always the same colour. Subsetting to a cluster or a chromosome will
not change what any colour means, which is what makes two of these plots safe
to compare side by side.


## Pseudobulk Profiles

Single cell profiles are noisy, especially in BAF, because each bin is
supported by relatively few reads. Aggregating across cells trades away the
cell-to-cell variation for a much cleaner consensus.

`scgenome.pl.plot_pseudobulk_tcn` averages total copy number across all cells.


```python

fig, ax = plt.subplots(figsize=(10, 2))
scgenome.pl.plot_pseudobulk_tcn(adata, ax=ax)

```


`scgenome.pl.plot_pseudobulk_ascn` does the same for the allele specific view.
It does not average the per cell BAF values. It sums the raw `alleleA`,
`alleleB` and `totalcounts` layers across cells and recomputes BAF from the
totals, so bins with more coverage carry more weight.


```python

fig, ax = plt.subplots(figsize=(10, 2))
scgenome.pl.plot_pseudobulk_ascn(adata, ax=ax)

```


Stacked, these give the consensus profile of the population.


```python

fig, axes = plt.subplots(nrows=2, figsize=(10, 4), sharex=True)

scgenome.pl.plot_pseudobulk_tcn(adata, ax=axes[0])
scgenome.pl.plot_pseudobulk_ascn(adata, ax=axes[1])

```


Note that this pseudobulk mixes the doubled and non-doubled populations, so the
total copy number levels are an average over two genuinely different genomes.
Aggregating within a cluster instead is usually what you want.


```python

fig, axes = plt.subplots(nrows=2, figsize=(10, 4), sharex=True)

wgd_cells = adata[adata.obs['n_wgd'] == 1]

scgenome.pl.plot_pseudobulk_tcn(wgd_cells, ax=axes[0])
scgenome.pl.plot_pseudobulk_ascn(wgd_cells, ax=axes[1])

```


To focus any of these on part of the genome, see
{doc}`multi_region_plotting`, which covers the `chromosome` and `region_mapper`
arguments that every one of these functions accepts.
