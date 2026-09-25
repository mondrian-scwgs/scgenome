"""Regression tests for allele specific state encoding and colors.

Both behaviours tested here previously failed silently, producing a plausible
looking but wrong heatmap rather than an error.
"""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import warnings

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import scgenome
from scgenome.plotting import cn_colors


@pytest.fixture
def allele_adata():
    """ Four cells covering every allele state, plus a bin with no allele cn
    """
    #                A-Hom  A-Gain  Bal  B-Gain  B-Hom  missing
    A = np.array([[2., 3., 1., 1., 0., np.nan]] * 4)
    B = np.array([[0., 1., 1., 3., 2., np.nan]] * 4)

    var = pd.DataFrame({
        'chr': ['1'] * 6,
        'start': np.arange(6) * 100 + 1,
        'end': (np.arange(6) + 1) * 100,
    }, index=[f'bin{i}' for i in range(6)])

    obs = pd.DataFrame(index=[f'cell{i}' for i in range(4)])

    adata = ad.AnnData(np.zeros((4, 6)), obs=obs, var=var)
    adata.layers['A'] = A
    adata.layers['B'] = B
    adata.uns['genome'] = 'hg19'

    return adata


def test_allele_state_encoding(allele_adata):
    """ States are encoded by their index in allele_state_names
    """
    result = cn_colors.add_allele_state_layer(allele_adata)
    state = result.layers['allele_state']

    assert cn_colors.allele_state_names == [
        'A-Hom', 'A-Gained', 'Balanced', 'B-Gained', 'B-Hom']

    np.testing.assert_array_equal(state[0, :5], [0, 1, 2, 3, 4])


def test_missing_allele_cn_is_nan_not_a_hom(allele_adata):
    """ Bins with no allele specific copy number must not read as A-Hom

    Seeding the state array with zeros made every nan bin fall through to
    state 0, painting missing data as confident loss of the B allele.
    """
    result = cn_colors.add_allele_state_layer(allele_adata)
    state = result.layers['allele_state']

    assert np.isnan(state[:, 5]).all()
    assert np.isfinite(state[:, :5]).all()


def test_missing_allele_cn_is_white(allele_adata):
    """ Bins with no allele specific copy number are not given a state color
    """
    result = cn_colors.add_allele_state_layer(allele_adata)
    colors = cn_colors.map_allele_state_colors(result.layers['allele_state'])

    white = cn_colors.hex_to_rgb('#ffffff')
    np.testing.assert_array_equal(colors[:, 5], np.tile(white, (4, 1)))

    a_hom = cn_colors.hex_to_rgb(cn_colors.allele_state_colors['A-Hom'])
    np.testing.assert_array_equal(colors[0, 0], a_hom)


def test_allele_state_colors_are_subset_invariant():
    """ A state maps to the same color regardless of the other states present

    Passing a ListedColormap to imshow let the norm be inferred from the data,
    so a matrix missing the top state silently remapped every color.
    """
    full = cn_colors.map_allele_state_colors(np.array([[0., 1., 2., 3., 4.]]))
    partial = cn_colors.map_allele_state_colors(np.array([[0., 1., 2., 3.]]))

    np.testing.assert_array_equal(full[0, :4], partial[0, :4])

    balanced = cn_colors.hex_to_rgb(cn_colors.allele_state_colors['Balanced'])
    np.testing.assert_array_equal(partial[0, 2], balanced)


def test_cn_colors_are_subset_invariant():
    """ The same guarantee holds for the total copy number palette
    """
    full = cn_colors.map_cn_colors(np.array([[0., 1., 2., 3., 4.]]))
    partial = cn_colors.map_cn_colors(np.array([[0., 1., 2.]]))

    np.testing.assert_array_equal(full[0, :3], partial[0, :3])


def test_resolve_palette_rejects_unknown():
    with pytest.raises(ValueError, match='unknown palette'):
        cn_colors.resolve_palette('not-a-palette')


def test_add_allele_state_layer_requires_allele_layers(allele_adata):
    del allele_adata.layers['B']

    with pytest.raises(ValueError, match="missing layer 'B'"):
        cn_colors.add_allele_state_layer(allele_adata)


def test_cmap_and_palette_are_mutually_exclusive(allele_adata):
    adata = cn_colors.add_allele_state_layer(allele_adata)

    with pytest.raises(ValueError, match='cannot provide both'):
        scgenome.pl.plot_cell_matrix(
            adata, layer_name='allele_state', cmap='viridis', palette='cn')

    plt.close('all')


def test_plot_cell_matrix_defaults_to_continuous(allele_adata):
    """ The generic matrix makes no assumption about what the values mean
    """
    g = scgenome.pl.plot_cell_matrix(allele_adata, layer_name='A')

    assert g['palette_info'] is None
    assert g['im'].cmap.name == 'viridis'

    plt.close('all')


def test_plot_cell_tcn_matrix_uses_the_cn_palette(allele_adata):
    allele_adata.layers['state'] = np.ones(allele_adata.shape)

    g = scgenome.pl.plot_cell_tcn_matrix(allele_adata)

    colors = g['im'].get_array()
    np.testing.assert_array_equal(
        colors[0, 0], cn_colors.hex_to_rgb(cn_colors.color_reference[1]))

    plt.close('all')


def test_plot_cell_tcn_matrix_warns_on_continuous_layer(allele_adata):
    """ The cn palette matches by equality, so continuous values render white
    """
    allele_adata.layers['copy'] = np.full(allele_adata.shape, 2.34)

    with pytest.warns(UserWarning, match='non integer values'):
        scgenome.pl.plot_cell_tcn_matrix(allele_adata, layer_name='copy')

    plt.close('all')


def test_plot_cell_tcn_matrix_quiet_on_integer_layer(allele_adata):
    allele_adata.layers['state'] = np.ones(allele_adata.shape)

    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        scgenome.pl.plot_cell_tcn_matrix(allele_adata)

    plt.close('all')


def test_deprecated_cn_matrix_matches_tcn_matrix(allele_adata):
    """ The alias must reproduce the old default behaviour exactly
    """
    allele_adata.layers['state'] = np.ones(allele_adata.shape)

    with pytest.warns(DeprecationWarning, match='plot_cell_cn_matrix is deprecated'):
        deprecated = scgenome.pl.plot_cell_cn_matrix(allele_adata)

    expected = scgenome.pl.plot_cell_tcn_matrix(allele_adata)

    np.testing.assert_array_equal(
        deprecated['im'].get_array(), expected['im'].get_array())

    plt.close('all')


def test_deprecated_raw_selects_a_continuous_colormap(allele_adata):
    """ raw=True meant 'this layer is continuous', which is now cmap
    """
    with pytest.warns(DeprecationWarning, match='raw is deprecated'):
        g = scgenome.pl.plot_cell_cn_matrix(allele_adata, layer_name='A', raw=True)

    assert g['palette_info'] is None
    assert g['im'].cmap.name == 'viridis'

    plt.close('all')


def test_plot_cell_ascn_matrix_does_not_modify_input(allele_adata):
    """ Plotting functions must not mutate the caller's adata
    """
    scgenome.pl.plot_cell_ascn_matrix(allele_adata)

    assert 'allele_state' not in allele_adata.layers

    plt.close('all')


def test_plot_cell_ascn_matrix_plots_all_bins(allele_adata):
    """ Bins are not filtered, callers subset adata themselves
    """
    g = scgenome.pl.plot_cell_ascn_matrix(allele_adata)
    assert g['adata'].shape == (4, 6)

    plt.close('all')


def test_plot_cell_ascn_matrix_reuses_existing_layer(allele_adata):
    """ An allele_state layer already on the adata is plotted as given
    """
    adata = cn_colors.add_allele_state_layer(allele_adata.copy())
    adata.layers['allele_state'][:] = 2.

    g = scgenome.pl.plot_cell_ascn_matrix(adata)

    np.testing.assert_array_equal(g['adata'].layers['allele_state'], 2.)

    plt.close('all')


def test_plot_cell_ascn_matrix_fig_legend_lists_states(allele_adata):
    """ The allele heatmap gets a patch legend of named states, not a colorbar
    """
    g = scgenome.pl.plot_cell_ascn_matrix_fig(allele_adata)

    legend = g['legend_info']['legend']
    assert legend.get_title().get_text() == 'AS CN state'

    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == cn_colors.allele_state_names

    plt.close('all')
