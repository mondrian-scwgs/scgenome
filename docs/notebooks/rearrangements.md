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


# Rearrangement Arcs

Copy number changes are the consequence of structural rearrangement, and the
two are far easier to interpret together than apart. A step in the copy number
profile with a breakpoint at exactly that position is a very different piece of
evidence from a step with nothing to explain it.

`scgenome.pl.plot_rearrangement_arcs` draws breakpoint calls over an existing
genome axis, so any of the copy number plots can be used as the backdrop.


```python

import matplotlib.pyplot as plt

import scgenome
from scgenome.pl import RegionMapper

adata = scgenome.datasets.OV081_Signals_reduced()
breakpoints = scgenome.datasets.OV081_breakpoints()

cell_id = adata.obs.query('n_wgd == 1').sort_values('quality', ascending=False).index[0]

breakpoints.head()

```


Each row is a pair of breakends, given as a chromosome, position and strand for
each side. The function needs only the six coordinate columns; everything else
is there so you can filter.


## Filtering by Read Support

Breakpoint callers are permissive, and the raw output is mostly low support
calls. Plotting all of them produces a figure dense enough to be useless.


```python

print(f'all calls:      {len(breakpoints)}')

supported = breakpoints.query('num_unique_reads > 5 and template_length_min > 200')

print(f'well supported: {len(supported)}')
print()
print(supported['rearrangement_type'].value_counts())

```


## One Chromosome

Passing `chromosome` restricts the arcs to breakpoints with both ends on that
chromosome, matching the `chromosome` argument of the copy number plot beneath.

The arcs are drawn above the axes, in axes-fraction coordinates, so the figure
needs headroom for them. `subplots_adjust` supplies it.


```python

fig, ax = plt.subplots(figsize=(10, 3))
fig.subplots_adjust(top=0.65)

scgenome.pl.plot_cell_tcn(adata, cell_id, chromosome='8', ax=ax)
scgenome.pl.plot_rearrangement_arcs(ax, supported, chromosome='8')

```


A vertical line marks each breakend, and an arc connects the two ends of a
pair. Height encodes the strand combination: the lowest rail is `+/-` and
`-/+`, the middle rail is `+/+` and `-/-`, and the highest rail is reserved for
breakends whose partner lies outside the plotted region, which are drawn as a
diagonal running off to the rail rather than as an arc.


## Zooming In

Restricting the view to an interval thins the arcs out enough to trace
individual ones against the copy number steps they explain.


```python

mapper = RegionMapper.for_chromosome('8', start=30_000_000, end=146_000_000)

fig, ax = plt.subplots(figsize=(10, 3))
fig.subplots_adjust(top=0.65)

scgenome.pl.plot_cell_tcn(adata, cell_id, region_mapper=mapper, ax=ax, s=6)
scgenome.pl.plot_rearrangement_arcs(ax, supported, region_mapper=mapper)

```


## Across Several Regions

Rearrangements between chromosomes cannot be drawn on a single chromosome axis
at all. A `RegionMapper` puts the relevant regions side by side, and arcs span
between them; see {doc}`multi_region_plotting` for how the mappers are built.

Select the breakpoints with *both* ends inside the plotted regions. Pairs with
one end elsewhere are still drawn, but as out-of-view diagonals rather than
arcs.


```python

chromosomes = ['1', '8']

in_regions = supported[
    supported['chromosome_1'].isin(chromosomes) &
    supported['chromosome_2'].isin(chromosomes)]

print(f'{len(in_regions)} breakpoints, of which '
      f'{(in_regions.chromosome_1 != in_regions.chromosome_2).sum()} are between chromosomes')

mapper_1_8 = RegionMapper.from_regions(chromosomes)

fig, ax = plt.subplots(figsize=(10, 3))
fig.subplots_adjust(top=0.65)

scgenome.pl.plot_cell_tcn(adata, cell_id, region_mapper=mapper_1_8, ax=ax, s=6)
scgenome.pl.plot_rearrangement_arcs(ax, in_regions, region_mapper=mapper_1_8)

```


The long arcs crossing the gap are the inter-chromosomal pairs. They are what
this layout exists for: on separate per-chromosome plots they would be two
unconnected vertical lines.


## Denser Views

With more regions the arcs start to overlap, and the line settings matter.
Thinner lines and some transparency keep the underlying copy number readable.


```python

chromosomes = ['1', '2', '8']

in_regions = supported[
    supported['chromosome_1'].isin(chromosomes) &
    supported['chromosome_2'].isin(chromosomes)]

mapper_3 = RegionMapper.from_regions(chromosomes)

fig, ax = plt.subplots(figsize=(12, 3))
fig.subplots_adjust(top=0.62)

scgenome.pl.plot_cell_tcn(
    adata, cell_id, region_mapper=mapper_3, squashy=True, ax=ax, s=4)

scgenome.pl.plot_rearrangement_arcs(
    ax, in_regions, region_mapper=mapper_3,
    linewidth=0.4, connector_linewidth=0.4, alpha=0.7)

```


The rails can be moved or turned off entirely when they crowd the figure.
`height_diff_strand`, `height_same_strand` and `height_out_of_view` set the
three rail heights as a fraction of the axes height, and `show_rail_labels`
hides the annotations at the right.


```python

fig, ax = plt.subplots(figsize=(10, 3))
fig.subplots_adjust(top=0.75)

scgenome.pl.plot_cell_tcn(adata, cell_id, chromosome='8', ax=ax)
scgenome.pl.plot_rearrangement_arcs(
    ax, supported, chromosome='8',
    height_diff_strand=0.05, height_same_strand=0.10, height_out_of_view=0.15,
    show_rail_labels=False)

```
