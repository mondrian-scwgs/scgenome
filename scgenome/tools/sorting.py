import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dst
import sklearn.preprocessing
import pandas as pd

from anndata import AnnData
from typing import Union, Any, Dict, Iterable

import scgenome.preprocessing.transform


def _default_agg_fn(values):
    """ Aggregation to use for a matrix whose function was not specified

    Integer matrices hold copy number states, where a median keeps the result a
    valid state. Continuous matrices are averaged. Both skip NaN.
    """
    if np.issubdtype(values.dtype, np.integer):
        return 'median'
    return 'mean'


def _resolve_cluster_aggregation(adata, layer_name, agg_X, agg_layers):
    """ Ensure the aggregate will carry whatever the sort is about to read

    aggregate_clusters only aggregates what it is asked for, so with no
    agg_layers it returns clusters with no layers at all, and sorting them
    raises. Fill in the matrices named by ``layer_name``, leaving any
    explicitly supplied function alone.
    """
    if isinstance(layer_name, (str, type(None))):
        layer_names = [layer_name]
    else:
        layer_names = list(layer_name)

    missing = [n for n in layer_names if n is not None and n not in adata.layers]
    if missing:
        raise ValueError(
            f'sort_clusters: missing required layers {missing}. '
            f'Available layers: {list(adata.layers.keys())}')

    agg_layers = dict(agg_layers) if agg_layers is not None else {}

    for name in layer_names:
        if name is None:
            if adata.X is None:
                raise ValueError(
                    'sort_clusters: layer_name is None, which sorts on X, but '
                    'adata.X is None. Pass a layer name instead.')
            if agg_X is None:
                agg_X = _default_agg_fn(adata.X)

        elif name not in agg_layers:
            agg_layers[name] = _default_agg_fn(adata.layers[name])

    return agg_X, agg_layers


def store_linkage(adata, key, linkage, leaves, ids, layer=None,
                  metric='cityblock', method='complete', standardize=False):
    """ Record the hierarchical clustering behind an ordering

    An ordering column flattens a tree into a permutation, losing the structure
    a dendrogram needs. Keeping the linkage makes that structure recoverable
    without reclustering. Records are keyed by the obs column they explain, so
    cell level and cluster level orderings coexist.

    Parameters
    ----------
    adata : AnnData
        data to annotate
    key : str
        obs column this linkage explains, for instance 'cell_order'
    linkage : numpy.ndarray
        scipy linkage matrix
    leaves : numpy.ndarray
        leaf indices into ``ids``, in dendrogram order
    ids : numpy.ndarray
        labels the linkage rows refer to, in adata order
    layer : str, optional
        layer the distances were computed on, None for X
    metric, method : str, optional
        arguments the linkage was built with
    standardize : bool, optional
        whether values were standardized first

    Modifies
    --------
    adata.uns['cell_order'][key] : linkage, leaves, ids and clustering parameters
    """
    adata.uns.setdefault('cell_order', {})[key] = {
        'linkage': linkage,
        'leaves': np.asarray(leaves),
        'ids': np.asarray(ids),
        'layer': layer,
        'metric': metric,
        'method': method,
        'standardize': standardize,
    }


def sort_cells(
        adata: AnnData,
        layer_name: Union[None, str, Iterable[Union[None,str]]]='copy',
        cell_ids: Iterable[str]=None,
        bin_ids: Iterable[str]=None,
        standardize: bool=False,
    ) -> AnnData:
    """ Sort cells by hierarchical clustering on copy number values.

    Parameters
    ----------
    adata : AnnData
        copy number data
    layer_name : str, optional
        layer with copy number data to use for sorting, None for X, by default 'copy'
    cell_ids : str, optional
        subset of cells to cluster, by default None
    bin_ids : str, optional
        subset of bins to cluster, by default None
    standardize : bool
        standardize the data prior to sorting, by default False

    Returns
    -------
    AnnData
        copy number data with cell_order column added to obs

    Reads
    -----
    adata.layers[layer_name] : copy number matrix

    Modifies
    --------
    adata.obs['cell_order'] : integer ordering of cells by hierarchical clustering
    adata.uns['cell_order']['cell_order'] : the linkage behind that ordering, so a
        dendrogram can be drawn without reclustering

    Notes
    -----
    The rows of `adata` are not reordered. `cell_order` is a column that
    plotting functions sort on, so several orderings can coexist on one object
    and can be combined for a nested sort. If `cell_ids` restricts the sort to a
    subset, cells outside it get `NaN`.

    Examples
    --------

    >>> import scgenome
    >>> adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    >>> adata = scgenome.tl.sort_cells(adata, layer_name='copy')

    The result is a permutation of the cell indices:

    >>> sorted(adata.obs['cell_order'].astype(int)) == list(range(adata.shape[0]))
    True

    Combine with a cluster assignment to group cells first and order within
    each group second::

        scgenome.pl.plot_cell_tcn_matrix(
            adata, cell_order_fields=['cluster_id', 'cell_order'])

    """
    if cell_ids is None:
        cell_ids = adata.obs.index

    if bin_ids is None:
        bin_ids = adata.var.index

    if len(cell_ids) <= 1:
        adata.obs['cell_order'] = 0.
        return adata

    def __get_layer(layer_name):
        if layer_name is not None:
            return np.array(adata[cell_ids, bin_ids].layers[layer_name])
        else:
            return np.array(adata[cell_ids, bin_ids].X)

    if isinstance(layer_name, (str, type(None))):
        X = __get_layer(layer_name)
    elif isinstance(layer_name, Iterable):
        X = np.concatenate([__get_layer(l) for l in layer_name], axis=1)
    else:
        raise ValueError(f'layer_name was {layer_name}')

    X = scgenome.preprocessing.transform.fill_missing(X)

    if standardize:
        X = sklearn.preprocessing.StandardScaler().fit_transform(X)

    Y = sch.linkage(dst.pdist(X, 'cityblock'), method='complete')
    Z = sch.dendrogram(Y, color_threshold=-1, no_plot=True)
    idx = np.array(Z['leaves'])

    ordering = np.zeros(idx.shape[0], dtype=int)
    ordering[idx] = np.arange(idx.shape[0])

    adata.obs['cell_order'] = np.nan
    adata.obs.loc[cell_ids, 'cell_order'] = pd.Series(ordering, index=adata.obs.loc[cell_ids].index)

    # The linkage describes the structure the ordering flattens, and a
    # dendrogram cannot be drawn from the leaf order alone
    store_linkage(
        adata, 'cell_order', Y, idx,
        np.asarray(adata.obs.loc[cell_ids].index),
        layer=layer_name, standardize=standardize)

    return adata


def sort_clusters(
        adata: AnnData,
        layer_name: Union[None, str, Iterable[Union[None,str]]]='copy',
        cluster_col: str='cluster_id',
        agg_X: Any=None,
        agg_layers: Dict=None,
        cell_ids: Iterable[str]=None,
        bin_ids: Iterable[str]=None,
        standardize: bool=False,
    ) -> AnnData:
    """ Sort clusters by hierarchical clustering on aggregated copy number values.

    Parameters
    ----------
    adata : AnnData
        copy number data
    layer_name : str, optional
        layer with copy number data to use for sorting, None for X, by default 'copy'
    cluster_col : str, optional
        column of cluster labels to sort
    agg_X : Any
        function to aggregate X, by default None. Only needed when
        ``layer_name`` is None, in which case a default is chosen.
    agg_layers : Dict, optional
        functions to aggregate layers keyed by layer names, by default None.
        Whatever ``layer_name`` names is aggregated regardless, since the sort
        reads it; a median for integer layers and a mean otherwise. Functions
        given here are used as supplied.
    cell_ids : str, optional
        subset of cells to cluster, by default None
    bin_ids : str, optional
        subset of bins to cluster, by default None
    standardize : bool
        standardize the data prior to sorting, by default False

    Returns
    -------
    AnnData
        copy number data with cluster_order column added to obs

    Reads
    -----
    adata.layers[layer_name] : copy number matrix
    adata.obs[cluster_col] : cluster assignment per cell

    Modifies
    --------
    adata.obs['cluster_order'] : integer ordering of clusters
    adata.uns['cell_order']['cluster_order'] : the cluster level linkage behind
        that ordering
    """

    if cell_ids is None:
        cell_ids = adata.obs.index

    if bin_ids is None:
        bin_ids = adata.var.index

    agg_X, agg_layers = _resolve_cluster_aggregation(
        adata, layer_name, agg_X, agg_layers)

    adata_clusters = scgenome.tools.cluster.aggregate_clusters(
        adata[cell_ids, bin_ids], cluster_col=cluster_col, agg_X=agg_X, agg_layers=agg_layers)

    adata_clusters = sort_cells(
        adata_clusters,
        layer_name=layer_name,
        standardize=standardize)

    # aggregate_clusters indexes clusters by str(cluster id), so the lookup
    # back has to stringify too or a non string cluster column misses entirely
    cluster_keys = adata.obs.loc[cell_ids, cluster_col].astype(str)

    adata.obs['cluster_order'] = np.nan
    adata.obs.loc[cell_ids, 'cluster_order'] = pd.Series(
        adata_clusters.obs.loc[cluster_keys.values, 'cell_order'].values,
        index=cluster_keys.index)

    # sort_cells stored a linkage over the aggregated clusters, which is
    # discarded with the aggregate unless we carry it across
    clustering = adata_clusters.uns.get('cell_order', {}).get('cell_order')
    if clustering is not None:
        adata.uns.setdefault('cell_order', {})['cluster_order'] = dict(
            clustering, level='cluster', cluster_col=cluster_col)

    return adata

