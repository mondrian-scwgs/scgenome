""" Every rename from the naming consistency pass.

Two rules are checked for each rename: the new name works without warning, and
the old name still works but warns. A third rule applies to anything stored in
an AnnData -- obs columns and uns keys are persisted in .h5ad files, so they
keep their names no matter what the arguments are called.
"""

import matplotlib
matplotlib.use('Agg')

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import scgenome


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def adata():
    return scgenome.datasets.OV2295_HMMCopy_reduced()


@pytest.fixture
def sorted_adata(adata):
    return scgenome.tl.sort_cells(adata, layer_name='copy')


@pytest.fixture
def allele_adata():
    return scgenome.datasets.OV081_Signals_reduced()


@pytest.fixture
def ax():
    return plt.subplots()[1]


# --------------------------------------------------------------------------
# the new names exist and are exported
# --------------------------------------------------------------------------

@pytest.mark.parametrize('name', [
    'plot_tcn_profile', 'plot_ascn_profile',
    'plot_tcn_heatmap', 'plot_ascn_heatmap',
    'plot_heatmap_fig', 'plot_tcn_heatmap_fig', 'plot_ascn_heatmap_fig',
])
def test_new_plotting_name_is_exported(name):
    assert callable(getattr(scgenome.pl, name))


@pytest.mark.parametrize('name', [
    'plot_cn_profile', 'plot_cell_tcn', 'plot_cell_ascn',
    'plot_cell_tcn_matrix', 'plot_cell_ascn_matrix', 'plot_cell_cn_matrix',
    'plot_cell_matrix_fig', 'plot_cell_tcn_matrix_fig', 'plot_cell_ascn_matrix_fig',
])
def test_old_plotting_name_is_still_exported(name):
    """ No removal date: the aliases stay reachable. """
    assert callable(getattr(scgenome.pl, name))


# --------------------------------------------------------------------------
# renamed arguments: new spelling silent, old spelling warns
# --------------------------------------------------------------------------

def test_obs_ids_and_var_ids(adata):
    cells, bins = list(adata.obs.index[:10]), list(adata.var.index[:40])

    result = scgenome.tl.sort_cells(
        adata.copy(), layer_name='copy', obs_ids=cells, var_ids=bins)
    assert result.obs['cell_order'].notna().sum() == len(cells)

    with pytest.warns(DeprecationWarning, match='cell_ids is deprecated'):
        scgenome.tl.sort_cells(
            adata.copy(), layer_name='copy', cell_ids=cells, var_ids=bins)

    with pytest.warns(DeprecationWarning, match='bin_ids is deprecated'):
        scgenome.tl.sort_cells(
            adata.copy(), layer_name='copy', obs_ids=cells, bin_ids=bins)


def test_passing_both_spellings_raises(adata):
    cells = list(adata.obs.index[:10])
    with pytest.raises(TypeError, match='deprecated alias'):
        scgenome.tl.sort_cells(
            adata.copy(), layer_name='copy', obs_ids=cells, cell_ids=cells)


def test_obs_order_on_annotation_panels(sorted_adata, ax):
    order = scgenome.tl.resolve_cell_order(sorted_adata, fields=['cell_order'])

    scgenome.pl.plot_obs_annotation(sorted_adata, 'quality', obs_order=order, ax=ax)

    with pytest.warns(DeprecationWarning, match='cell_order is deprecated'):
        scgenome.pl.plot_obs_annotation(
            sorted_adata, 'quality', cell_order=order, ax=plt.subplots()[1])


def test_var_order_on_annotation_panels(adata, ax):
    bins = scgenome.tl.resolve_bin_order(adata)

    scgenome.pl.plot_var_annotation(adata, 'gc', var_order=bins, ax=ax)

    with pytest.warns(DeprecationWarning, match='bin_order is deprecated'):
        scgenome.pl.plot_var_annotation(adata, 'gc', bin_order=bins, ax=plt.subplots()[1])


def test_obs_order_fields_on_heatmap(sorted_adata, ax):
    scgenome.pl.plot_heatmap(
        sorted_adata, layer_name='state', obs_order_fields=['cell_order'], ax=ax)

    with pytest.warns(DeprecationWarning, match='cell_order_fields is deprecated'):
        scgenome.pl.plot_heatmap(
            sorted_adata, layer_name='state', cell_order_fields=['cell_order'],
            ax=plt.subplots()[1])


@pytest.mark.parametrize('kwargs', [
    dict(obs_order_fields=['cell_order'], obs_order=True),
    dict(cell_order_fields=['cell_order'], cell_order=True),
    dict(obs_order_fields=['cell_order'], cell_order=True),
    dict(cell_order_fields=['cell_order'], obs_order=True),
])
def test_order_and_order_fields_stay_mutually_exclusive(sorted_adata, kwargs, ax):
    """ The exclusion must hold across every mix of old and new spellings. """
    order = scgenome.tl.resolve_cell_order(sorted_adata, fields=['cell_order'])
    kwargs = {k: (order if v is True else v) for k, v in kwargs.items()}

    with warnings.catch_warnings():
        # the old spellings warn; this test is about the exclusion, not the warning
        warnings.simplefilter('ignore', DeprecationWarning)
        with pytest.raises((ValueError, TypeError), match='both'):
            scgenome.pl.plot_heatmap(
                sorted_adata, layer_name='state', ax=ax, **kwargs)


def test_show_obs_ids_on_heatmap(sorted_adata, ax):
    scgenome.pl.plot_heatmap(
        sorted_adata, layer_name='state', show_obs_ids=True, ax=ax)

    with pytest.warns(DeprecationWarning, match='show_cell_ids is deprecated'):
        scgenome.pl.plot_heatmap(
            sorted_adata, layer_name='state', show_cell_ids=True,
            ax=plt.subplots()[1])


def test_layer_name_on_pca(adata):
    scgenome.tl.pca_loadings(adata.copy(), layer_name='copy', n_components=3)

    with pytest.warns(DeprecationWarning, match='layer is deprecated'):
        scgenome.tl.pca_loadings(adata.copy(), layer='copy', n_components=3)


def test_cluster_field_on_aggregate(adata):
    clustered = scgenome.tl.cluster_cells(adata, layer_name='copy', max_k=4)

    scgenome.tl.aggregate_clusters(clustered, cluster_field='cluster_id')

    with pytest.warns(DeprecationWarning, match='cluster_col is deprecated'):
        scgenome.tl.aggregate_clusters(clustered, cluster_col='cluster_id')


def test_bin_size_on_create_bins():
    expected = scgenome.tl.create_bins(bin_size=int(1e7), genome='hg19')

    with pytest.warns(DeprecationWarning, match='binsize is deprecated'):
        legacy = scgenome.tl.create_bins(binsize=int(1e7), genome='hg19')

    assert len(expected) == len(legacy)


# --------------------------------------------------------------------------
# the profile collapse, and the squashy hazard it carries
# --------------------------------------------------------------------------

def _ylim(fn):
    fig, axes = plt.subplots()
    fn(axes)
    return tuple(np.round(axes.get_ylim(), 6))


def test_plot_cell_tcn_alias_keeps_its_squashy_default(adata):
    """ plot_cell_tcn defaulted squashy=True; plot_tcn_profile defaults False.

    The alias must pin its own default, or existing calls silently change what
    they draw.
    """
    cell = str(adata.obs.index[0])

    with pytest.warns(DeprecationWarning, match='plot_cell_tcn is deprecated'):
        legacy = _ylim(lambda a: scgenome.pl.plot_cell_tcn(adata, cell, ax=a))

    squashed = _ylim(
        lambda a: scgenome.pl.plot_tcn_profile(adata, cell, ax=a, squashy=True))
    unsquashed = _ylim(
        lambda a: scgenome.pl.plot_tcn_profile(adata, cell, ax=a))

    assert legacy == squashed
    assert legacy != unsquashed


def test_plot_cn_profile_alias_keeps_its_own_layer_defaults(adata):
    """ plot_cn_profile drew X with no state coloring unless told otherwise. """
    cell = str(adata.obs.index[0])

    with pytest.warns(DeprecationWarning, match='plot_cn_profile is deprecated'):
        legacy = _ylim(lambda a: scgenome.pl.plot_cn_profile(adata, cell, ax=a))

    explicit = _ylim(lambda a: scgenome.pl.plot_tcn_profile(
        adata, cell, ax=a, value_layer_name=None, state_layer_name=None))

    assert legacy == explicit


def test_plot_ascn_profile_matches_its_alias(allele_adata):
    cell = str(allele_adata.obs.index[0])

    new = _ylim(lambda a: scgenome.pl.plot_ascn_profile(allele_adata, cell, ax=a))

    with pytest.warns(DeprecationWarning, match='plot_cell_ascn is deprecated'):
        old = _ylim(lambda a: scgenome.pl.plot_cell_ascn(allele_adata, cell, ax=a))

    assert new == old


# --------------------------------------------------------------------------
# the heatmap family
# --------------------------------------------------------------------------

def test_tcn_heatmap_matches_its_alias(adata):
    expected = scgenome.pl.plot_tcn_heatmap(adata, ax=plt.subplots()[1])

    with pytest.warns(DeprecationWarning, match='plot_cell_tcn_matrix is deprecated'):
        legacy = scgenome.pl.plot_cell_tcn_matrix(adata, ax=plt.subplots()[1])

    np.testing.assert_array_equal(
        legacy['im'].get_array(), expected['im'].get_array())


def test_ascn_heatmap_matches_its_alias(allele_adata):
    expected = scgenome.pl.plot_ascn_heatmap(allele_adata, ax=plt.subplots()[1])

    with pytest.warns(DeprecationWarning, match='plot_cell_ascn_matrix is deprecated'):
        legacy = scgenome.pl.plot_cell_ascn_matrix(allele_adata, ax=plt.subplots()[1])

    np.testing.assert_array_equal(
        legacy['im'].get_array(), expected['im'].get_array())


@pytest.mark.parametrize('old_name,new_name', [
    ('plot_cell_matrix_fig', 'plot_heatmap_fig'),
    ('plot_cell_tcn_matrix_fig', 'plot_tcn_heatmap_fig'),
])
def test_fig_preset_aliases_warn(adata, old_name, new_name):
    getattr(scgenome.pl, new_name)(adata)
    plt.close('all')

    with pytest.warns(DeprecationWarning, match=f'{old_name} is deprecated'):
        getattr(scgenome.pl, old_name)(adata)


# --------------------------------------------------------------------------
# plot_rearrangement_arcs took ax first, unlike everything else
# --------------------------------------------------------------------------

@pytest.fixture
def breakpoints():
    data = scgenome.datasets.OV081_breakpoints()
    return data[data['num_unique_reads'] > 20]


def _arc_artists(fn):
    fig, axes = plt.subplots()
    axes.set_xlim(0, 250e6)
    axes.set_ylim(0, 8)
    fn(axes)
    return len(axes.lines) + len(axes.patches)


def test_arcs_new_argument_order(breakpoints):
    assert _arc_artists(lambda a: scgenome.pl.plot_rearrangement_arcs(
        breakpoints, chromosome='1', ax=a)) > 0


@pytest.mark.parametrize('call', [
    'two_positional', 'three_positional', 'mixed',
])
def test_arcs_legacy_ax_first_order_still_works(breakpoints, call):
    """ Every legacy positional form must bind to the same arguments. """
    expected = _arc_artists(lambda a: scgenome.pl.plot_rearrangement_arcs(
        breakpoints, chromosome='1', ax=a))

    calls = {
        'two_positional': lambda a: scgenome.pl.plot_rearrangement_arcs(
            a, breakpoints, '1'),
        'three_positional': lambda a: scgenome.pl.plot_rearrangement_arcs(
            a, breakpoints, '1', None),
        'mixed': lambda a: scgenome.pl.plot_rearrangement_arcs(
            a, breakpoints, chromosome='1'),
    }

    with pytest.warns(DeprecationWarning, match='passing ax first'):
        assert _arc_artists(calls[call]) == expected


# --------------------------------------------------------------------------
# R1: nothing stored inside the AnnData is renamed
# --------------------------------------------------------------------------

def test_stored_obs_columns_keep_their_names(sorted_adata):
    assert 'cell_order' in sorted_adata.obs.columns


def test_stored_uns_keys_keep_their_names(adata):
    cells, bins = list(adata.obs.index[:10]), list(adata.var.index[:40])

    clustered = scgenome.tl.cluster_cells(
        adata.copy(), layer_name='copy', max_k=4, obs_ids=cells, var_ids=bins)
    params = clustered.uns['clustering']['params']

    assert 'cell_ids' in params and 'obs_ids' not in params
    assert 'bin_ids' in params and 'var_ids' not in params


def test_stored_linkage_keys_keep_their_names(sorted_adata):
    record = sorted_adata.uns['cell_order']['cell_order']

    assert 'layer' in record and 'layer_name' not in record


def test_stored_cluster_order_key_keeps_its_name(adata):
    clustered = scgenome.tl.cluster_cells(adata, layer_name='copy', max_k=4)
    result = scgenome.tl.sort_clusters(clustered, layer_name='copy')
    record = result.uns['cell_order']['cluster_order']

    assert record['cluster_col'] == 'cluster_id'
    assert 'cluster_field' not in record


def test_grid_result_cell_order_is_readable_but_warns(sorted_adata):
    grid = (scgenome.pl.CellGrid(sorted_adata, obs_order_fields=['cell_order'])
            .add_heatmap('state')
            .plot())

    assert len(grid.obs_order) == sorted_adata.shape[0]

    with pytest.warns(DeprecationWarning, match='GridResult.cell_order'):
        assert list(grid.cell_order) == list(grid.obs_order)
