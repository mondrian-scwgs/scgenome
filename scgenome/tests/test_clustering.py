""" Tests for cluster_cells with method='kmeans_bic', the default.

The BIC formula had two ways to fail on ordinary input: the sweep reached
k == n_cells, where the pooled variance has no degrees of freedom left, and the
per cluster sums ran over range(k) while np.bincount only runs to the largest
label kmeans actually used. Either aborted the whole sweep, discarding the
valid BIC values already computed for every other k.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import sklearn.cluster

import scgenome
from scgenome.tools.cluster import _compute_kmean_bic, _kmeans_bic


def _adata(copy, state=None):
    """ An adata over the given cell by bin matrix """
    n_cells, n_bins = copy.shape

    var = pd.DataFrame({
        'chr': ['1'] * n_bins,
        'start': np.arange(n_bins) * 100 + 1,
        'end': (np.arange(n_bins) + 1) * 100,
    }, index=[f'bin{i}' for i in range(n_bins)])

    obs = pd.DataFrame(index=[f'c{i}' for i in range(n_cells)])

    adata = ad.AnnData(copy.copy(), obs=obs, var=var)
    adata.layers['copy'] = copy
    adata.layers['state'] = (copy if state is None else state).astype(int)
    adata.uns['genome'] = 'hg19'
    return adata


@pytest.fixture
def blobs():
    """ Forty two cells in three well separated clusters, over five bins """
    rng = np.random.default_rng(0)

    profiles = np.array([
        [2., 2., 2., 2., 2.],
        [4., 4., 4., 1., 1.],
        [1., 1., 3., 5., 2.],
    ])

    copy = profiles[np.repeat([0, 1, 2], 14)] + rng.normal(0, 0.05, size=(42, 5))
    return _adata(copy)


# --- the sweep no longer reaches k == n_cells ----------------------------

def test_default_max_k_on_small_data(blobs):
    """ Fewer cells than the default max_k used to divide by zero """
    adata = scgenome.tl.cluster_cells(blobs, layer_name='copy')

    assert adata.uns['clustering']['params']['opt_k'] == 3
    assert adata.obs['cluster_id'].nunique() == 3


def test_max_k_clamped_below_cell_count(blobs):
    adata = scgenome.tl.cluster_cells(blobs, layer_name='copy', max_k=1000)

    assert adata.uns['clustering']['params']['max_k'] == blobs.shape[0] - 1


def test_max_k_clamp_counts_only_clustered_cells(blobs):
    """ cell_ids subsets the cells, so the clamp has to follow it """
    cell_ids = blobs.obs.index[:6]
    adata = scgenome.tl.cluster_cells(
        blobs, layer_name='copy', cell_ids=cell_ids, max_k=100)

    assert adata.uns['clustering']['params']['max_k'] == 5
    assert set(adata.obs['cluster_id'][6:]) == {'-1'}


def test_min_k_follows_max_k_down():
    """ Two cells leave room for k=1 only, below the default min_k of 2 """
    adata = _adata(np.array([[1., 1., 1.], [5., 5., 5.]]))
    adata = scgenome.tl.cluster_cells(adata, layer_name='copy')

    assert adata.uns['clustering']['params']['opt_k'] == 1
    assert adata.obs['cluster_id'].nunique() == 1


def test_one_cell_is_rejected():
    adata = _adata(np.array([[1., 2., 3.]]))

    with pytest.raises(ValueError, match='at least 2 cells'):
        scgenome.tl.cluster_cells(adata, layer_name='copy')


def test_bic_of_k_equal_n_loses_the_argmax():
    """ k == N is undefined, not an exception: it just cannot win """
    X = np.random.default_rng(0).normal(size=(8, 4))
    model = sklearn.cluster.KMeans(
        n_clusters=8, init='k-means++', random_state=100).fit(X)

    assert _compute_kmean_bic(model, X) == -np.inf


# --- empty clusters no longer index past the end of bincount ------------

def test_duplicated_rows():
    """ Three distinct profiles, six copies each: k > 3 leaves clusters empty """
    copy = np.repeat(np.array([[0., 0.], [10., 10.], [20., 20.]]), 6, axis=0)
    adata = scgenome.tl.cluster_cells(_adata(copy), layer_name='copy', max_k=14)

    # the smallest k that fits the data exactly, so the smallest with no error
    assert adata.uns['clustering']['params']['opt_k'] == 3
    assert adata.obs['cluster_id'].nunique() == 3
    assert sorted(adata.obs['cluster_size']) == [6] * 18


@pytest.mark.parametrize('k', [10, 14])
def test_bic_with_fewer_effective_clusters_than_k(k):
    """ bincount is shorter than n_clusters here, which used to IndexError """
    X = np.repeat(np.array([[0., 0.], [10., 10.], [20., 20.]]), 6, axis=0)
    model = sklearn.cluster.KMeans(
        n_clusters=k, init='k-means++', random_state=100).fit(X)

    assert len(np.bincount(model.labels_)) < k
    assert _compute_kmean_bic(model, X) == np.inf


def test_unused_interior_label_does_not_zero_the_bic():
    """ An empty cluster between used ones gave bincount 0 there, and log(0) """
    rng = np.random.default_rng(1)
    X = np.repeat(np.array([[0., 0.], [10., 10.]]), 8, axis=0)
    X = X + rng.normal(0, 0.01, size=X.shape)

    labels = np.where(np.arange(16) < 8, 0, 2)
    centers = np.array([X[:8].mean(axis=0), [50., 50.], X[8:].mean(axis=0)])

    model = sklearn.cluster.KMeans(n_clusters=3)
    model.labels_ = labels
    model.cluster_centers_ = centers

    assert np.isfinite(_compute_kmean_bic(model, X))


# --- the criterion itself still selects the planted k -------------------

def test_sweep_is_finite_and_picks_the_planted_k(blobs):
    """ Every k from 2 to N-1 scores, and the best of them is the true k """
    X = np.array(blobs.layers['copy'])
    ks = range(2, X.shape[0])

    bics = np.array([_kmeans_bic(X, k)[1] for k in ks])

    assert np.isfinite(bics).all()
    assert ks[int(bics.argmax())] == 3


def test_allele_specific_layers():
    """ The layer_name=['A', 'B'] call the docstring advertises """
    rng = np.random.default_rng(2)
    a = np.repeat(np.array([[2., 2., 2.], [1., 1., 3.]]), 10, axis=0)
    b = np.repeat(np.array([[1., 1., 0.], [2., 0., 2.]]), 10, axis=0)

    adata = _adata(a + b)
    adata.layers['A'] = a + rng.normal(0, 0.02, size=a.shape)
    adata.layers['B'] = b + rng.normal(0, 0.02, size=b.shape)

    adata = scgenome.tl.cluster_cells(adata, layer_name=['A', 'B'])

    assert adata.uns['clustering']['params']['opt_k'] == 2
    assert adata.obs['cluster_id'].nunique() == 2
