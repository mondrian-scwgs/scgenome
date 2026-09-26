""" Tests for panel primitives and the CellGrid layout.

The properties worth protecting are the ones that were impossible before:
several panels sharing one row order, panel widths that do not depend on how
many annotation bars sit beside them, one legend for panels that describe the
same values, and a dendrogram drawn from the clustering an ordering came from.
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
from scgenome.plotting import panels
from scgenome.tools.ordering import OrderConflict


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)

    profiles = np.array([
        [2, 2, 2, 2, 2, 2],
        [4, 4, 4, 1, 1, 1],
        [1, 1, 3, 5, 5, 2],
    ])
    cluster_of_cell = np.repeat([0, 1, 2], 4)
    state = profiles[cluster_of_cell]
    copy = state + rng.normal(0, 0.01, size=state.shape)

    var = pd.DataFrame({
        'chr': ['1', '1', '1', '2', '2', '2'],
        'start': [1, 101, 201, 1, 101, 201],
        'end': [100, 200, 300, 100, 200, 300],
        'gc': [0.4, 0.5, 0.6, 0.45, 0.55, 0.65],
    }, index=[f'bin{i}' for i in range(6)])

    obs = pd.DataFrame({
        'cluster_id': [str(c) for c in cluster_of_cell],
        'quality': rng.random(12),
    }, index=[f'c{i}' for i in range(12)])

    adata = ad.AnnData(copy.copy(), obs=obs, var=var)
    adata.layers['copy'] = copy
    adata.layers['state'] = state.astype(int)
    adata.uns['genome'] = 'hg19'
    return scgenome.tl.sort_cells(adata, layer_name='copy')


@pytest.fixture
def tree(adata):
    leaves = list(adata.obs.sort_values('cell_order').index)
    newick = '(' + ','.join(f'{c}:1' for c in leaves) + ');'
    return Bio.Phylo.read(io.StringIO(newick), 'newick')


# --- panels draw into an axes and nothing else ---------------------------


def test_heatmap_panel_returns_a_legend_spec(adata):
    fig, ax = plt.subplots()
    result = panels.heatmap(adata, ax, layer='state', palette='cn')

    assert result.legend.kind == 'patches'
    assert len(result.legend.levels) == len(result.legend.colors)
    plt.close('all')


def test_continuous_heatmap_panel_describes_a_colorbar(adata):
    fig, ax = plt.subplots()
    result = panels.heatmap(adata, ax, layer='copy', cmap='viridis')

    assert result.legend.kind == 'colorbar'
    assert result.legend.mappable is result.im
    plt.close('all')


def test_panels_do_not_create_figures(adata):
    """ A panel draws into the axes it is given and allocates nothing """
    fig, ax = plt.subplots()
    before = len(plt.get_fignums())

    panels.heatmap(adata, ax, layer='state', palette='cn')
    panels.obs_annotation(adata, plt.subplots()[1], 'cluster_id')

    assert len(plt.get_fignums()) == before + 1  # only the one we made
    plt.close('all')


def test_missing_cells_raise_by_default(adata):
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match='not in adata'):
        panels.heatmap(adata, ax, layer='state', cell_order=['c0', 'nope'])
    plt.close('all')


def test_missing_cells_can_be_blanked(adata):
    fig, ax = plt.subplots()
    result = panels.heatmap(
        adata, ax, layer='state', cell_order=['c0', 'nope', 'c1'],
        on_missing='blank')

    drawn = np.asarray(result.im.get_array())

    assert drawn.shape[0] == 3
    assert not result.extras['present'][1]
    plt.close('all')


def test_obs_annotation_reports_a_missing_column(adata):
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match='missing obs column'):
        panels.obs_annotation(adata, ax, 'nope')
    plt.close('all')


def test_legend_specs_of_one_palette_share_a_key(adata):
    fig, axes = plt.subplots(ncols=2)
    a = panels.heatmap(adata, axes[0], layer='state', palette='cn')
    b = panels.heatmap(adata, axes[1], layer='state', palette='cn')

    assert a.legend.key == b.legend.key
    plt.close('all')


# --- dendrogram ----------------------------------------------------------


def test_dendrogram_draws_from_the_stored_linkage(adata):
    fig, ax = plt.subplots()
    order = adata.obs.sort_values('cell_order').index

    result = panels.dendrogram(adata, ax, cell_order=order)

    assert result.extras['linkage'].shape == (adata.shape[0] - 1, 4)
    assert ax.get_ylim() == (adata.shape[0] - 0.5, -0.5)
    plt.close('all')


def test_dendrogram_needs_a_stored_linkage(adata):
    del adata.uns['cell_order']
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match='no stored linkage'):
        panels.dendrogram(adata, ax)
    plt.close('all')


def test_dendrogram_refuses_an_order_that_would_cross_its_brackets(adata):
    """ Same contiguity rule as a tree, applied to the linkage """
    fig, ax = plt.subplots()
    order = list(adata.obs.sort_values('cell_order').index)
    scrambled = order[::2] + order[1::2]

    with pytest.raises(OrderConflict):
        panels.dendrogram(adata, ax, cell_order=scrambled)
    plt.close('all')


def test_dendrogram_row_extent_matches_a_heatmap(adata):
    fig, axes = plt.subplots(ncols=2)
    order = adata.obs.sort_values('cell_order').index

    panels.dendrogram(adata, axes[0], cell_order=order)
    panels.heatmap(adata, axes[1], layer='state', palette='cn', cell_order=order)

    assert axes[0].get_ylim() == axes[1].get_ylim()
    plt.close('all')


# --- CellGrid ------------------------------------------------------------


def test_two_heatmaps_share_one_row_order(adata):
    g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'])
         .add_heatmap('state', palette='cn', name='A')
         .add_heatmap('copy', cmap='viridis', name='B')
         .plot())

    rows_a = list(g.panels['A'].extras['adata'].obs.index)
    rows_b = list(g.panels['B'].extras['adata'].obs.index)

    assert rows_a == rows_b == list(g.cell_order)
    plt.close('all')


def test_heatmap_width_does_not_depend_on_annotation_count(adata):
    """ The defect that made two separate figures incomparable """
    g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'])
         .add_heatmap('state', palette='cn', name='A')
         .add_heatmap('copy', cmap='viridis', name='B')
         .add_obs_annotation(['cluster_id', 'quality'])
         .plot())

    width_a = g.axes['A'].get_position().bounds[2]
    width_b = g.axes['B'].get_position().bounds[2]

    assert width_a == pytest.approx(width_b)
    plt.close('all')


def test_panels_share_a_row_extent(adata):
    g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'])
         .add_dendrogram()
         .add_heatmap('state', palette='cn', name='A')
         .add_obs_annotation('cluster_id')
         .plot())

    extents = {g.axes[n].get_ylim() for n in ('dendrogram', 'A', 'cluster_id')}

    assert len(extents) == 1
    plt.close('all')


def test_identical_legends_collapse(adata):
    g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'])
         .add_heatmap('state', palette='cn', name='A')
         .add_heatmap('state', palette='cn', name='B')
         .plot())

    assert len(g.legends) == 1
    plt.close('all')


def test_different_legends_are_kept_apart(adata):
    g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'])
         .add_heatmap('state', palette='cn', name='A')
         .add_heatmap('copy', cmap='viridis', name='B')
         .plot())

    assert len(g.legends) == 2
    plt.close('all')


def test_a_panel_may_carry_its_own_adata(adata):
    """ Two samples side by side, aligned by the shared order """
    half = adata[adata.obs.index[:6]].copy()

    g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'])
         .add_heatmap('state', palette='cn', name='all')
         .add_heatmap('state', adata=half, palette='cn', name='half')
         .plot())

    drawn = np.asarray(g.panels['half'].im.get_array())

    assert drawn.shape[0] == adata.shape[0]
    assert g.panels['half'].extras['present'].sum() == 6
    plt.close('all')


def test_tree_and_heatmap_compose(adata, tree):
    g = (scgenome.pl.CellGrid(adata, tree=tree)
         .add_tree()
         .add_heatmap('state', palette='cn', name='A')
         .plot())

    assert list(g.cell_order) == scgenome.tl.tree_leaf_order(tree)
    assert g.axes['tree'].get_ylim() == (adata.shape[0] + 0.5, 0.5)
    plt.close('all')


def test_var_annotation_sits_above_its_heatmap(adata):
    g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'])
         .add_heatmap('state', palette='cn', name='A')
         .add_var_annotation('gc')
         .plot())

    heat = g.axes['A'].get_position()
    var = g.axes['A:gc'].get_position()

    assert var.y0 > heat.y0
    assert var.x0 == pytest.approx(heat.x0)
    plt.close('all')


def test_var_annotation_needs_a_heatmap(adata):
    with pytest.raises(ValueError, match='add a heatmap before'):
        scgenome.pl.CellGrid(adata).add_var_annotation('gc')


def test_plot_needs_a_panel(adata):
    with pytest.raises(ValueError, match='at least one panel'):
        scgenome.pl.CellGrid(adata).plot()


def test_duplicate_panel_names_are_made_unique(adata):
    g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'])
         .add_heatmap('state', palette='cn')
         .add_heatmap('state', palette='cn')
         .plot())

    assert sorted(g.panels) == ['state', 'state 2']
    plt.close('all')


def test_grid_does_not_modify_adata(adata, tree):
    before = list(adata.obs.columns)

    (scgenome.pl.CellGrid(adata, tree=tree)
     .add_tree()
     .add_heatmap('state', palette='cn')
     .plot())

    assert list(adata.obs.columns) == before
    plt.close('all')


def test_cell_order_and_fields_are_mutually_exclusive(adata):
    with pytest.raises(ValueError, match='cannot provide both'):
        scgenome.pl.CellGrid(
            adata, cell_order_fields=['cell_order'], cell_order=adata.obs.index)


# --- the presets still return what they did ------------------------------


def test_fig_preset_returns_the_documented_keys(adata):
    g = scgenome.pl.plot_cell_tcn_matrix_fig(
        adata, cell_order_fields=['cell_order'],
        annotation_fields=['cluster_id'], fig=plt.figure())

    for key in ('fig', 'axes', 'tree_ax', 'heatmap_ax', 'adata', 'im',
                'legend_info', 'annotation_info'):
        assert key in g, key

    assert 'cluster_id' in g['annotation_info']
    assert 'level_colors' in g['annotation_info']['cluster_id']
    plt.close('all')


def test_fig_preset_exposes_the_grid(adata):
    g = scgenome.pl.plot_cell_matrix_fig(
        adata, layer_name='copy', cell_order_fields=['cell_order'], fig=plt.figure())

    assert isinstance(g['grid'], scgenome.pl.GridResult)
    plt.close('all')


def test_plot_tree_cn_is_deprecated_but_works(adata, tree):
    with pytest.warns(DeprecationWarning, match='CellGrid'):
        fig = scgenome.pl.plot_tree_cn(
            tree, adata, layer_name='state', palette='cn', fig=plt.figure())

    assert fig is not None
    assert 'phylo_order' not in adata.obs.columns
    plt.close('all')
