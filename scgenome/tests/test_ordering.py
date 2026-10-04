""" Tests for cell and bin ordering as a value rather than a plotting side effect.

Two behaviours here previously had no way to be expressed at all: sharing one
ordering between panels, and drawing a dendrogram, which needs the linkage that
`sort_cells` used to discard. A third, combining a tree with sort fields, was
refused outright rather than checked.
"""

import io

import Bio.Phylo
import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import scgenome
from scgenome.tools.ordering import OrderConflict


@pytest.fixture
def adata():
    """ Four cells over six bins, deliberately out of genomic order in var
    """
    var = pd.DataFrame({
        'chr': ['2', '1', '1', '2', '1', '2'],
        'start': [201, 101, 1, 1, 201, 101],
        'end': [300, 200, 100, 100, 300, 200],
    }, index=[f'bin{i}' for i in range(6)])

    obs = pd.DataFrame({
        'group': ['b', 'b', 'a', 'a'],
        'rank': [1, 0, 1, 0],
    }, index=[f'c{i}' for i in range(4)])

    adata = ad.AnnData(np.arange(24, dtype=float).reshape(4, 6), obs=obs, var=var)
    adata.layers['state'] = np.ones((4, 6))
    adata.uns['genome'] = 'hg19'
    return adata


@pytest.fixture
def tree():
    """ Two clades of two: (c0, c1) and (c2, c3)
    """
    return Bio.Phylo.read(io.StringIO('((c0:1,c1:1):1,(c2:1,c3:1):1);'), 'newick')


# --- ordering as a value -------------------------------------------------


def test_resolve_cell_order_sorts_on_fields_first_primary(adata):
    order = scgenome.tl.resolve_cell_order(adata, fields=['group', 'rank'])

    assert list(order) == ['c3', 'c2', 'c1', 'c0']


def test_resolve_cell_order_defaults_to_obs_order(adata):
    assert list(scgenome.tl.resolve_cell_order(adata)) == list(adata.obs.index)


def test_resolve_cell_order_does_not_modify_adata(adata):
    before = list(adata.obs.columns)

    scgenome.tl.resolve_cell_order(adata, fields=['group'])

    assert list(adata.obs.columns) == before


def test_resolve_cell_order_reports_missing_fields(adata):
    with pytest.raises(ValueError, match='missing obs columns'):
        scgenome.tl.resolve_cell_order(adata, fields=['nope'])


def test_resolve_bin_order_sorts_by_chromosome_then_start(adata):
    order = scgenome.tl.resolve_bin_order(adata)

    assert list(order) == ['bin2', 'bin1', 'bin4', 'bin3', 'bin5', 'bin0']


def test_resolve_bin_order_rejects_unknown_chromosome(adata):
    adata.var['chr'] = ['zz'] * 6

    with pytest.raises(ValueError, match='mismatching chromosomes'):
        scgenome.tl.resolve_bin_order(adata)


# --- a tree constrains the order, it does not compete with it ------------


def test_tree_leaf_order_is_the_order_a_tree_implies(adata, tree):
    assert scgenome.tl.tree_leaf_order(tree) == ['c0', 'c1', 'c2', 'c3']


def test_reconciling_a_tree_with_an_order_is_explicit(adata, tree):
    """ A tree and sort fields are reconciled by rotating the tree, by hand

    Plotting takes a tree or an order, never both, so this is the only way the
    two ever meet, and it is always something the caller asked for.
    """
    order = scgenome.tl.resolve_cell_order(adata, fields=['group', 'rank'])
    aligned = scgenome.tl.align_tree_to_order(tree, order)

    assert scgenome.tl.tree_leaf_order(aligned) == list(order)


def test_incompatible_order_raises_rather_than_drawing_a_lie(adata, tree):
    """ Interleaving the two clades cannot be drawn by any rotation
    """
    interleaved = ['c0', 'c2', 'c1', 'c3']

    with pytest.raises(OrderConflict) as excinfo:
        scgenome.tl.align_tree_to_order(tree, interleaved)

    assert excinfo.value.n_leaves == 2
    assert excinfo.value.hi - excinfo.value.lo + 1 > excinfo.value.n_leaves
    assert 'reorder' in str(excinfo.value)


def test_reorder_lets_the_tree_win(tree):
    """ Fields survive as within clade rotation rather than being discarded
    """
    aligned = scgenome.tl.align_tree_to_order(
        tree, ['c0', 'c2', 'c1', 'c3'], on_conflict='reorder')

    assert scgenome.tl.tree_leaf_order(aligned) == ['c0', 'c1', 'c2', 'c3']


def test_align_tree_rotates_to_match(tree):
    aligned = scgenome.tl.align_tree_to_order(tree, ['c3', 'c2', 'c1', 'c0'])

    assert scgenome.tl.tree_leaf_order(aligned) == ['c3', 'c2', 'c1', 'c0']


def test_align_tree_does_not_modify_the_caller_tree(tree):
    scgenome.tl.align_tree_to_order(tree, ['c3', 'c2', 'c1', 'c0'])

    assert scgenome.tl.tree_leaf_order(tree) == ['c0', 'c1', 'c2', 'c3']


def test_a_non_leaf_cell_splitting_a_clade_is_a_conflict(tree):
    """ Cells absent from the tree are caught by the same contiguity test
    """
    with pytest.raises(OrderConflict):
        scgenome.tl.align_tree_to_order(tree, ['c0', 'extra', 'c1', 'c2', 'c3'])


def test_deep_ladder_tree_does_not_exhaust_the_stack():
    """ Ladder shaped trees are deep in proportion to cell count
    """
    n = 2000
    newick = 'c0:1'
    for i in range(1, n):
        newick = f'({newick},c{i}:1):1'
    tree = Bio.Phylo.read(io.StringIO(newick + ';'), 'newick')

    order = [f'c{i}' for i in range(n)]
    aligned = scgenome.tl.align_tree_to_order(tree, order)

    assert scgenome.tl.tree_leaf_order(aligned) == order


# --- the linkage survives sorting ----------------------------------------


def test_sort_cells_keeps_the_linkage():
    adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    adata = scgenome.tl.sort_cells(adata, layer_name='copy')

    record = adata.uns['cell_order']['cell_order']

    assert record['linkage'].shape == (adata.shape[0] - 1, 4)
    assert record['layer'] == 'copy'
    assert list(record['ids']) == list(adata.obs.index)


def test_stored_leaves_agree_with_the_cell_order_column():
    adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    adata = scgenome.tl.sort_cells(adata, layer_name='copy')

    record = adata.uns['cell_order']['cell_order']
    by_linkage = list(np.asarray(record['ids'])[record['leaves']])
    by_column = list(adata.obs.sort_values('cell_order').index)

    assert by_linkage == by_column


def test_sort_clusters_keeps_the_cluster_level_linkage():
    adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    adata = scgenome.tl.sort_clusters(adata, layer_name='copy')

    record = adata.uns['cell_order']['cluster_order']

    assert record['level'] == 'cluster'
    assert record['cluster_col'] == 'cluster_id'
    assert len(record['ids']) == adata.obs['cluster_id'].nunique()


def test_retained_linkage_round_trips_to_a_tree_and_catches_the_documented_idiom():
    """ The point of keeping the linkage: rebuild the tree it describes

    Also covers the conflict end to end, since sort_cells' own docstring
    recommends ordering by cluster first, which splits clades of that tree.
    """
    import scipy.cluster.hierarchy as sch

    adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    adata = scgenome.tl.sort_cells(adata, layer_name='copy')

    record = adata.uns['cell_order']['cell_order']
    ids = np.asarray(record['ids'])

    def to_newick(node):
        if node.is_leaf():
            return f'{ids[node.id]}:{node.dist:.4f}'
        return f'({to_newick(node.left)},{to_newick(node.right)}):{node.dist:.4f}'

    tree = Bio.Phylo.read(
        io.StringIO(to_newick(sch.to_tree(record['linkage'])) + ';'), 'newick')

    assert tree.count_terminals() == adata.shape[0]

    # The ordering the linkage produced is realizable, by construction
    by_cell = scgenome.tl.resolve_cell_order(adata, fields=['cell_order'])
    scgenome.tl.align_tree_to_order(tree, by_cell)
    assert list(by_cell) == list(adata.obs.sort_values('cell_order').index)

    # Grouping by cluster first splits clades, and must be refused
    grouped = scgenome.tl.resolve_cell_order(adata, fields=['cluster_id', 'cell_order'])
    with pytest.raises(OrderConflict):
        scgenome.tl.align_tree_to_order(tree, grouped)

    # The escape hatch must yield a tree that can actually be drawn
    rescued = scgenome.tl.align_tree_to_order(tree, grouped, on_conflict='reorder')
    scgenome.tl.align_tree_to_order(rescued, scgenome.tl.tree_leaf_order(rescued))


# --- plotting takes an ordering instead of computing its own -------------


def test_cell_order_matches_cell_order_fields(adata):
    plt.figure()
    by_fields = scgenome.pl.plot_cell_matrix(adata, cell_order_fields=['group', 'rank'])

    order = scgenome.tl.resolve_cell_order(adata, fields=['group', 'rank'])
    plt.figure()
    by_order = scgenome.pl.plot_cell_matrix(adata, cell_order=order)

    assert list(by_fields['adata'].obs.index) == list(by_order['adata'].obs.index)
    np.testing.assert_array_equal(
        np.asarray(by_fields['im'].get_array()),
        np.asarray(by_order['im'].get_array()))
    plt.close('all')


def test_cell_order_and_cell_order_fields_are_mutually_exclusive(adata):
    plt.figure()
    with pytest.raises(ValueError, match='cannot provide both'):
        scgenome.pl.plot_cell_matrix(
            adata, cell_order_fields=['group'], cell_order=list(adata.obs.index))
    plt.close('all')


def test_unknown_cells_in_cell_order_are_reported(adata):
    plt.figure()
    with pytest.raises(ValueError, match='not in adata'):
        scgenome.pl.plot_cell_matrix(adata, cell_order=['c0', 'nope'])
    plt.close('all')


def test_tree_does_not_modify_the_caller_adata(adata, tree):
    before = list(adata.obs.columns)

    scgenome.pl.plot_cell_matrix_fig(adata, layer_name='state', tree=tree, fig=plt.figure())

    assert list(adata.obs.columns) == before
    assert 'phylo_order' not in adata.obs.columns
    plt.close('all')


def test_a_tree_orders_the_rows_by_its_leaves(adata, tree):
    g = scgenome.pl.plot_cell_matrix_fig(
        adata, layer_name='state', tree=tree, fig=plt.figure())

    assert list(g['adata'].obs.index) == scgenome.tl.tree_leaf_order(tree)
    plt.close('all')


def test_a_tree_and_sort_fields_together_are_refused(adata, tree):
    """ Reconciling them is align_tree_to_order, called by the user
    """
    with pytest.raises(ValueError, match='ordered one way'):
        scgenome.pl.plot_cell_matrix_fig(
            adata, layer_name='state', tree=tree,
            cell_order_fields=['group'], fig=plt.figure())
    plt.close('all')


def test_a_rotated_tree_is_how_fields_and_a_tree_are_combined(adata, tree):
    order = scgenome.tl.resolve_cell_order(adata, fields=['group', 'rank'])
    aligned = scgenome.tl.align_tree_to_order(tree, order)

    g = scgenome.pl.plot_cell_matrix_fig(
        adata, layer_name='state', tree=aligned, fig=plt.figure())

    assert list(g['adata'].obs.index) == list(order)
    plt.close('all')


def test_show_subsets_does_not_grow_the_callers_list():
    adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    fields = ['cluster_id']

    scgenome.pl.plot_cell_tcn_matrix_fig(
        adata, annotation_fields=fields, show_subsets=True, fig=plt.figure())

    assert fields == ['cluster_id']
    plt.close('all')
