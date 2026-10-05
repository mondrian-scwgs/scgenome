# `tl.cluster_cells` crashes in `_compute_kmean_bic` on small or degenerate data

`tl.cluster_cells(..., method='kmeans_bic')` — the **default** method — raises
`IndexError` or `ZeroDivisionError` instead of clustering. Both failures come from
`_compute_kmean_bic` in `scgenome/tools/cluster.py`, and both are triggered by ordinary
inputs, including two of the three bundled example datasets.

This is pre-existing and unrelated to any recent loader or plotting work.

| | |
|---|---|
| **File** | `scgenome/tools/cluster.py` |
| **Function** | `_compute_kmean_bic` (line 20), reached via `_kmeans_bic` (line 53) from `cluster_cells` (line 183) |
| **Affects** | `method='kmeans_bic'`, the default for `tl.cluster_cells` |
| **Does not affect** | `method='gmm_diag_bic'`, which uses scikit-learn's own `model.bic(X)` |

---

## Impact

`cluster_cells` sweeps `k` from `min_k` to `max_k` and takes the `argmax` BIC
(lines 174–191). The loop has no error tolerance, so **a single failing `k` aborts the
whole call**, discarding the valid BIC values already computed for every other `k`.

The practical effect is that the default clustering path fails on small datasets. Both
reproductions below use data shipped with the repo.

---

## Reproduction 1 — `ZeroDivisionError` (bundled OV2295, default arguments)

```python
import scgenome

adata = scgenome.datasets.OV2295_HMMCopy_reduced()
adata = scgenome.pp.calculate_filter_metrics(adata)
adata = scgenome.pp.filter_cells(adata)      # -> 25 cells

scgenome.tl.cluster_cells(adata, layer_name='copy')
```

```
ZeroDivisionError: float division by zero
  scgenome/tools/cluster.py:39 in _compute_kmean_bic
    cl_var = (1.0 / (N - n_clusters) / d) * sum(
```

## Reproduction 2 — `IndexError` (bundled OV081, the call the docstring advertises)

`cluster.py:143` documents `cluster_cells(adata, layer_name=['A', 'B'])`. That exact call
fails:

```python
import scgenome

adata = scgenome.datasets.OV081_Signals_reduced()
scgenome.tl.cluster_cells(adata[:60, :120].copy(), layer_name=['A', 'B'])
```

```
IndexError: index 53 is out of bounds for axis 0 with size 53
  scgenome/tools/cluster.py:45 in _compute_kmean_bic
    bic = np.sum([cluster_sizes[i] * np.log(cluster_sizes[i]) -
```

---

## Root cause

Two independent defects in the same function.

### Mechanism A — `k` is allowed to equal `N`, making the variance term divide by zero

```python
# cluster.py:154
max_k = min(adata.shape[0], max_k)   # caps INCLUSIVE of the cell count
# cluster.py:174
ks = range(min_k, max_k + 1)
```

`max_k` is clamped to the number of cells, but inclusively, so the sweep reaches
`k == N`. In `_compute_kmean_bic`:

```python
# cluster.py:39
cl_var = (1.0 / (N - n_clusters) / d) * sum(...)
```

`N - n_clusters` is then `0`. With the default `max_k=100`, any dataset of 100 cells or
fewer hits this — which is why the 25-cell OV2295 example fails on default arguments.

Minimal proof, no scgenome needed:

```python
import numpy as np, sklearn.cluster
from scgenome.tools.cluster import _compute_kmean_bic

X = np.random.default_rng(0).normal(size=(8, 4))
for k in (7, 8):
    m = sklearn.cluster.KMeans(n_clusters=k, init='k-means++', random_state=100).fit(X)
    _compute_kmean_bic(m, X)
```

```
k=7 -> bic=-53.54
k=8 -> ZeroDivisionError: float division by zero
```

### Mechanism B — `np.bincount` is shorter than `n_clusters` when a cluster is empty

```python
# cluster.py:35
cluster_sizes = np.bincount(labels)
...
# cluster.py:45
bic = np.sum([cluster_sizes[i] * ... for i in range(n_clusters)]) - const_term
```

`np.bincount(labels)` returns an array of length `labels.max() + 1`, **not** `n_clusters`.
When KMeans leaves high label indices unused — which it does whenever the data cannot
support `k` distinct centroids, e.g. duplicated or near-duplicated rows — the array is
shorter than `n_clusters` and the comprehension indexes past its end.

Minimal proof:

```python
import numpy as np, sklearn.cluster
from scgenome.tools.cluster import _compute_kmean_bic

X = np.repeat(np.array([[0., 0.], [10., 10.], [20., 20.]]), 6, axis=0)   # N=18, 3 distinct
for k in (3, 10, 14):
    m = sklearn.cluster.KMeans(n_clusters=k, init='k-means++', random_state=100).fit(X)
    print(k, m.n_clusters, len(np.bincount(m.labels_)))
    _compute_kmean_bic(m, X)
```

```
k=3   n_clusters=3   len(bincount)=3   -> ok
k=10  n_clusters=10  len(bincount)=3   -> IndexError: index 3 is out of bounds for axis 0 with size 3
k=14  n_clusters=14  len(bincount)=4   -> IndexError: index 4 is out of bounds for axis 0 with size 4
```

There is a quieter version of the same bug: when an *interior* label is unused,
`np.bincount` returns `0` at that position rather than being short, so line 45 evaluates
`np.log(0)`. That does not raise — it silently makes the BIC `-inf`, so that `k` can never
win the `argmax` even if it was the best choice.

---

## Workarounds (verified)

On the 25-cell OV2295 example:

| Call | Result |
|---|---|
| `cluster_cells(adata, layer_name='copy')` | `ZeroDivisionError` |
| `cluster_cells(adata, layer_name='copy', max_k=adata.shape[0] - 1)` | works, 6 clusters |
| `cluster_cells(adata, layer_name='copy', max_k=5)` | works, 5 clusters |
| `cluster_cells(adata, layer_name='copy', method='gmm_diag_bic')` | works, 2 clusters |

---

## Proposed fix

Three small changes, all in `scgenome/tools/cluster.py`:

1. **Line 154** — stop the sweep short of `N`, so the variance term always has at least one
   degree of freedom:

   ```python
   max_k = min(adata.shape[0] - 1, max_k)
   ```

2. **Line 35** — make `cluster_sizes` the length the loop assumes:

   ```python
   cluster_sizes = np.bincount(labels, minlength=n_clusters)
   ```

3. **Lines 39 and 45** — sum only over non-empty clusters, which removes the `log(0)`
   and leaves the BIC well defined when KMeans returns fewer effective clusters than `k`:

   ```python
   nonempty = [i for i in range(n_clusters) if cluster_sizes[i] > 0]
   ```

   and iterate `nonempty` instead of `range(n_clusters)` in both comprehensions.

A defensive `if N <= n_clusters: return -np.inf` guard at the top of
`_compute_kmean_bic` makes the function safe regardless of what the caller passes, and
lets a bad `k` lose the `argmax` rather than abort the sweep.

### Fix verification

Applied to three well-separated Gaussian blobs (N=42, d=5, true k=3), sweeping k=2..41:
all 40 BIC values were finite and `argmax` selected **k=3**, so the fix does not distort
model selection on well-behaved data.

Both minimal reproductions above pass with the fix applied.

---

## Note on a third edge case

With *exactly* duplicated rows, within-cluster variance is genuinely `0`, so
`np.log(2 * np.pi * cl_var)` at line 45 is `log(0)` and the BIC becomes `±inf` even after
the fix above. This is a distinct problem from A and B and is unlikely in real copy-number
data, but a zero-variance guard would make the function total. Worth a decision, not
necessarily a change.

---

## Suggested regression tests

None of the current 119 tests call `cluster_cells` with `method='kmeans_bic'`, which is why
this survived. Worth adding:

- `cluster_cells` on a small adata (N < default `max_k`) with default arguments — covers A.
- `cluster_cells` on an adata with duplicated rows — covers B.
- `_compute_kmean_bic` returns a finite float whenever `n_clusters < N`, and the
  `argmax` over a sweep picks the planted k on synthetic blobs.
