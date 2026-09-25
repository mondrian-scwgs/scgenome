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


# Multi Region Plots

A whole genome profile devotes most of its width to regions where nothing is
happening. When the interesting events are on a few chromosomes, it is better
to plot only those, side by side on one axis.

`scgenome.pl.RegionMapper` does this. It takes a list of genomic regions and
maps their coordinates onto a single continuous axis, inserting gaps and spine
breaks between them so the discontinuity is visible rather than misleading.


```python

import matplotlib.pyplot as plt

import scgenome
from scgenome.pl import GenomicRegion, RegionMapper

adata = scgenome.datasets.OV081_Signals_reduced()

cell_id = adata.obs.query('n_wgd == 1').sort_values('quality', ascending=False).index[0]
cell_id

```


## Building a Region Mapper

`RegionMapper.from_regions` accepts a list of string specifications. Whole
chromosomes, chromosome arms, and explicit intervals can be mixed freely.


```python

mapper = RegionMapper.from_regions(['1', '8', 'chr17:25000000-81195210'])

print('labels:   ', mapper.region_labels())
print('axis span:', mapper.xlim())
print('midpoints:', [f'{x/1e6:.0f}M' for x in mapper.region_midpoints()])

```


The mapper translates genomic coordinates into positions on that axis.
Positions outside every region map to `nan`, which is what keeps points and
arcs from the rest of the genome off the plot.


```python

print('chr1:100M  ->', mapper.map_position('1', 100e6))
print('chr3:50M   ->', mapper.map_position('3', 50e6))
print('contains chr8:60M?', mapper.contains('8', 60e6))

```


## Plotting Across Regions

Every profile plotting function takes a `region_mapper` argument. Passing the
same mapper to several of them lines the panels up exactly.


```python

fig, ax = plt.subplots(figsize=(10, 2.5))
scgenome.pl.plot_cell_tcn(adata, cell_id, region_mapper=mapper, ax=ax)

```


The three regions are drawn in the order given, separated by gaps, and each is
labelled beneath its own stretch of axis.


## Combining Panels

Because the mapper fixes the coordinate transform, panels sharing it are
directly comparable. Here total copy number, allele specific copy number, and
the population consensus are stacked over the same two chromosomes.


```python

mapper_1_8 = RegionMapper.from_regions(['1', '8'])

fig, axes = plt.subplots(nrows=3, figsize=(10, 6), sharex=True)

scgenome.pl.plot_cell_tcn(adata, cell_id, region_mapper=mapper_1_8, ax=axes[0])
scgenome.pl.plot_cell_ascn(adata, cell_id, region_mapper=mapper_1_8, ax=axes[1])
scgenome.pl.plot_pseudobulk_tcn(adata, region_mapper=mapper_1_8, ax=axes[2])

```


`scgenome.pl.plot_cn_rect`, which draws copy number as filled segments rather
than points, takes the same argument.


```python

fig, axes = plt.subplots(nrows=2, figsize=(10, 4), sharex=True)

scgenome.pl.plot_cell_tcn(adata, cell_id, region_mapper=mapper, squashy=True, ax=axes[0])
scgenome.pl.plot_cn_rect(adata, obs_id=cell_id, region_mapper=mapper, ax=axes[1])

```


So does `scgenome.pl.plot_profile`, which plots any per-bin column from
`adata.var` rather than copy number.


```python

var_data = adata.var.loc[adata.var['gc'] > 0, ['chr', 'start', 'end', 'gc']]

fig, ax = plt.subplots(figsize=(10, 2.5))
scgenome.pl.plot_profile(var_data, y='gc', region_mapper=mapper, ax=ax)
ax.set_ylabel('GC fraction')

```


## The Other Factory Methods

`from_regions` is the one you reach for when selecting regions by hand. Two
others exist, and both are used automatically on your behalf.

`RegionMapper.whole_genome` lays out every chromosome. This is what a plotting
function builds when you pass neither `chromosome` nor `region_mapper`.


```python

whole_genome = RegionMapper.whole_genome()
print(f'{len(whole_genome.regions)} regions:', whole_genome.region_labels())

```


Constructing it yourself is worth doing when you want to adjust the layout. On
a narrow figure the chromosome labels collide, and `min_label_spacing`
suppresses labels for chromosomes narrower than the given number of bases.


```python

sparse_labels = RegionMapper.whole_genome(min_label_spacing=150e6)

fig, ax = plt.subplots(figsize=(8, 2))
scgenome.pl.plot_cell_tcn(adata, cell_id, region_mapper=sparse_labels, squashy=True, ax=ax)

```


`RegionMapper.for_chromosome` covers a single chromosome, optionally trimmed to
an interval. This is what a plotting function builds when you pass
`chromosome='8'`, so constructing it explicitly is only necessary when you want
the trimmed form.


```python

print(RegionMapper.for_chromosome('8').regions[0])
print(RegionMapper.for_chromosome('8', start=50_000_000, end=100_000_000).regions[0])

```


## Constructing Regions Directly

When the string specifications are not expressive enough, build
`GenomicRegion` objects and pass them to `RegionMapper` yourself. Each takes a
chromosome, a start, an end and a label, which lets you name a region something
other than the coordinates it spans.


```python

regions = [
    GenomicRegion('1', 0, 249250621, 'chr1'),
    GenomicRegion('8', 0, 146364022, 'chr8'),
    GenomicRegion('17', 25000000, 81195210, '17q'),
]

manual = RegionMapper(regions)

fig, ax = plt.subplots(figsize=(10, 2.5))
scgenome.pl.plot_cell_tcn(adata, cell_id, region_mapper=manual, ax=ax)

```


Region mappers are also what makes it possible to draw rearrangements between
two distant loci on the same axis; see {doc}`rearrangements`.
