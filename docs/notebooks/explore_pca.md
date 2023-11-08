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

```python

import scgenome
import matplotlib.pyplot as plt

adata = scgenome.datasets.OV2295_HMMCopy_reduced()

```

```python

pcadata = scgenome.tl.pca_loadings(adata, layer=None)

fig = plt.figure(figsize=(20, 4))
scgenome.pl.plot_cn_profile(
    pcadata, 'PC1',
    value_layer_name=None)

fig = plt.figure(figsize=(20, 4))
scgenome.pl.plot_cn_profile(
    pcadata, 'PC2',
    value_layer_name=None)

fig = plt.figure(figsize=(20, 4))
scgenome.pl.plot_cn_profile(
    pcadata, 'PC3',
    value_layer_name=None)

fig = plt.figure(figsize=(20, 4))
scgenome.pl.plot_cn_profile(
    pcadata, 'PC4',
    value_layer_name=None)

```

```python

pcadata

```

```python

```
