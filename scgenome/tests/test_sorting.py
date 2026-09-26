""" Tests for sort_cells and sort_clusters.

sort_clusters aggregates cells into clusters, sorts those, then maps the result
back per cell. Each of those three steps had a way to fail on ordinary input:
the aggregate carried nothing to sort, X was never aggregated, and the map back
missed because aggregate_clusters stringifies cluster ids into its index.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scgenome
import scgenome.tools.cluster


@pytest.fixture
def adata():
    """ Twelve cells in three well separated clusters, over eight bins
    """
    rng = np.random.default_rng(0)

    profiles = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2],
        [4, 4, 4, 4, 1, 1, 1, 1],
        [1, 1, 3, 3, 5, 5, 2, 2],
    ])

    cluster_of_cell = np.repeat([0, 1, 2], 4)
    state = profiles[cluster_of_cell]
    copy = state + rng.normal(0, 0.01, size=state.shape)

    var = pd.DataFrame({
        'chr': ['1'] * 8,
        'start': np.arange(8) * 100 + 1,
        'end': (np.arange(8) + 1) * 100,
    }, index=[f'bin{i}' for i in range(8)])

    obs = pd.DataFrame({
        'cluster_id': [str(c) for c in cluster_of_cell],
        'int_cluster': cluster_of_cell,
        'cat_cluster': pd.Categorical([str(c) for c in cluster_of_cell]),
    }, index=[f'c{i}' for i in range(12)])

    adata = ad.AnnData(copy.copy(), obs=obs, var=var)
    adata.layers['copy'] = copy
    adata.layers['state'] = state.astype(int)
    adata.uns['genome'] = 'hg19'
    return adata


def _is_grouped(values):
    """ Whether equal values form contiguous runs """
    changes = sum(1 for a, b in zip(values, values[1:]) if a != b)
    return changes == len(set(values)) - 1


# --- sort_clusters works on its own defaults -----------------------------


def test_sort_clusters_works_with_default_arguments(adata):
    """ aggregate_clusters carries no layers unless asked, so the sort had
    nothing to read and raised KeyError on sort_clusters' own defaults
    """
    result = scgenome.tl.sort_clusters(adata)

    assert result.obs['cluster_order'].notnull().all()
    assert set(result.obs['cluster_order']) == {0, 1, 2}


def test_cluster_order_is_constant_within_a_cluster(adata):
    result = scgenome.tl.sort_clusters(adata)

    per_cluster = result.obs.groupby('cluster_id')['cluster_order'].nunique()

    assert (per_cluster == 1).all()


def test_cluster_order_groups_cells_when_sorted_on(adata):
    result = scgenome.tl.sort_clusters(adata)

    ordered = result.obs.sort_values('cluster_order')['cluster_id'].tolist()

    assert _is_grouped(ordered)


def test_sort_clusters_on_an_integer_layer(adata):
    result = scgenome.tl.sort_clusters(adata, layer_name='state')

    assert result.obs['cluster_order'].notnull().all()


def test_sort_clusters_on_x(adata):
    """ layer_name=None sorts on X, which needs agg_X rather than agg_layers """
    result = scgenome.tl.sort_clusters(adata, layer_name=None)

    assert result.obs['cluster_order'].notnull().all()


def test_sort_clusters_on_several_layers(adata):
    result = scgenome.tl.sort_clusters(adata, layer_name=['copy', 'state'])

    assert result.obs['cluster_order'].notnull().all()


# --- cluster ids are not always strings ----------------------------------


def test_integer_cluster_column(adata):
    """ aggregate_clusters indexes by str(cluster id); the map back must agree """
    result = scgenome.tl.sort_clusters(adata, cluster_col='int_cluster')

    assert result.obs['cluster_order'].notnull().all()
    per_cluster = result.obs.groupby('int_cluster')['cluster_order'].nunique()
    assert (per_cluster == 1).all()


def test_categorical_cluster_column(adata):
    result = scgenome.tl.sort_clusters(adata, cluster_col='cat_cluster')

    assert result.obs['cluster_order'].notnull().all()


def test_integer_and_string_cluster_columns_agree(adata):
    """ The same partition written two ways must sort the same """
    by_str = scgenome.tl.sort_clusters(adata.copy(), cluster_col='cluster_id')
    by_int = scgenome.tl.sort_clusters(adata.copy(), cluster_col='int_cluster')

    assert (by_str.obs['cluster_order'].values == by_int.obs['cluster_order'].values).all()


# --- explicit arguments are still honoured -------------------------------


def test_explicit_agg_layers_is_not_overridden(adata):
    calls = []

    def spy(values):
        calls.append(1)
        return np.nanmedian(values)

    scgenome.tl.sort_clusters(adata, layer_name='copy', agg_layers={'copy': spy})

    assert calls, 'the supplied aggregation function was not used'


def test_agg_layers_is_extended_not_replaced(adata):
    """ A layer the caller asked for but does not sort on is still aggregated """
    scgenome.tl.sort_clusters(
        adata, layer_name='copy', agg_layers={'state': np.nanmedian})

    assert adata.obs['cluster_order'].notnull().all()


def test_caller_agg_layers_dict_is_not_modified(adata):
    agg_layers = {'state': np.nanmedian}

    scgenome.tl.sort_clusters(adata, layer_name='copy', agg_layers=agg_layers)

    assert list(agg_layers) == ['state']


def test_cell_ids_subset_leaves_other_cells_null(adata):
    subset = adata.obs.index[:8]

    result = scgenome.tl.sort_clusters(adata, cell_ids=subset)

    assert result.obs.loc[subset, 'cluster_order'].notnull().all()
    assert result.obs['cluster_order'].isnull().sum() == 4


# --- errors name the problem ---------------------------------------------


def test_missing_layer_is_reported_clearly(adata):
    with pytest.raises(ValueError, match='missing required layers'):
        scgenome.tl.sort_clusters(adata, layer_name='nope')


def test_sorting_on_x_when_there_is_no_x_is_reported(adata):
    empty = ad.AnnData(obs=adata.obs.copy(), var=adata.var.copy())

    with pytest.raises(ValueError, match='adata.X is None'):
        scgenome.tl.sort_clusters(empty, layer_name=None)


# --- sort_cells, for contrast --------------------------------------------


def test_sort_cells_orders_every_cell(adata):
    result = scgenome.tl.sort_cells(adata, layer_name='copy')

    assert sorted(result.obs['cell_order'].astype(int)) == list(range(adata.shape[0]))


def test_sort_cells_groups_clusters_together(adata):
    """ Well separated clusters should come out contiguous """
    result = scgenome.tl.sort_cells(adata, layer_name='copy')

    ordered = result.obs.sort_values('cell_order')['cluster_id'].tolist()

    assert _is_grouped(ordered)
