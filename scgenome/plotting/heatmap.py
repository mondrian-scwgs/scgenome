import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import Bio.Phylo
import collections.abc
import warnings

from anndata import AnnData

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.cm

import scgenome.refgenome
from scgenome.tools.ordering import (
    align_tree_to_order, resolve_bin_order, resolve_cell_order)
from . import cn_colors
from . import panels
from .grid import CellGrid


def plot_cell_matrix(
        adata: AnnData,
        layer_name=None,
        cell_order_fields=(),
        cell_order=None,
        ax=None,
        vmin=None,
        vmax=None,
        cmap=None,
        palette=None,
        show_cell_ids=False,
        style='black',
        rasterized=False):
    """ Plot a matrix of per cell values across the genome

    Makes no assumption about what the values mean. Values are colored with a
    continuous colormap unless a discrete `palette` is given. For total copy
    number or allele specific states, prefer `plot_cell_tcn_matrix` or
    `plot_cell_ascn_matrix`, which select the matching palette for you.

    Parameters
    ----------
    adata : AnnData
        per cell data with var describing genomic bins
    layer_name : str, optional
        layer with values to plot, None for X, by default None
    cell_order_fields : list, optional
        columns of obs on which to sort cells, by default None
    cell_order : pandas.Index, optional
        explicit cell ids in plot order, from `scgenome.tl.resolve_cell_order`.
        Mutually exclusive with cell_order_fields. Pass one order to several
        panels so their rows are guaranteed to line up.
    ax : matplotlib.axes.Axes, optional
        existing axis to plot into, by default None
    vmin, vmax : float, optional
        vmin and vmax define the data range that the colormap covers, see `matplotlib.pyplot.imshow`.
        Applies to `cmap` only, discrete palettes map values to colors directly.
    cmap : str or matplotlib.colors.Colormap, optional
        continuous colormap to use, by default 'viridis'. Mutually exclusive
        with palette.
    palette : str or dict, optional
        discrete palette to use, 'cn' for total copy number states,
        'allele_state' for allele specific states, or a dict mapping value to
        color. Mutually exclusive with cmap. Unlike cmap, a palette maps values
        to colors directly, so colors do not depend on the range of values
        present.
    show_cell_ids : bool, optional
        show cell ids on heatmap axis, by default False
    style : str, optional
        style for spines and chromosome dividing lines and other plot elements,
        by default 'black'
    rasterized : bool, optional
        rasterize the plot, by default False

    Returns
    -------
    dict
        Dictionary of plot and data elements

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()
        scgenome.pl.plot_cell_matrix(adata, layer_name='copy', vmin=0, vmax=4)

    """

    if ax is None:
        ax = plt.gca()

    if cell_order is not None and len(cell_order_fields) > 0:
        raise ValueError(
            'cannot provide both cell_order and cell_order_fields, '
            'cell_order_fields is sugar for resolve_cell_order(adata, fields=...)')

    if cell_order is None:
        cell_order = resolve_cell_order(adata, fields=cell_order_fields)

    drawn = panels.heatmap(
        adata, ax,
        layer=layer_name,
        cell_order=cell_order,
        palette=palette,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_cell_ids=show_cell_ids,
        style=style,
        rasterized=rasterized)

    return {
        'ax': drawn.ax,
        'im': drawn.im,
        'adata': drawn.extras['adata'],
        'palette_info': drawn.extras['palette_info'],
    }


def map_catagorigal_colors(values, cmap=None):
    """ Map categorical values to colors

    .. deprecated::
        Misspelled, use `scgenome.pl.map_categorical_colors`.
    """
    warnings.warn(
        'map_catagorigal_colors is deprecated, use map_categorical_colors',
        DeprecationWarning, stacklevel=2)

    return panels.map_categorical_colors(values, cmap=cmap)


# Adapted from: https://github.com/bernatgel/karyoploteR/blob/master/R/color.R
cyto_band_giemsa_stain_colors = {
    'gneg': '#FFFFFF',
    'gpos25': '#C8C8C8',
    # 'gpos33': '#D2D2D2',
    'gpos50': '#C8C8C8',
    # 'gpos66': '#A0A0A0',
    'gpos75': '#828282',
    'gpos100': '#000000',
    'gpos': '#000000',
    'stalk': '#647FA4', # repetitive areas
    'acen': '#D92F27', # centromeres
    'gvar': '#DCACAC', # previously '#DCDCDC'
}


def plot_cell_matrix_fig(
        adata: AnnData,
        layer_name=None,
        tree=None,
        cell_order_fields=None,
        cell_order=None,
        annotation_fields=None,
        annotation_cmap=None,
        var_annotation_fields=None,
        var_annotation_cmap=None,
        fig=None,
        vmin=None,
        vmax=None,
        cmap=None,
        palette=None,
        show_cell_ids=False,
        show_subsets=False,
        style='black'):
    """ Plot a matrix of per cell values with annotations and a legend

    Makes no assumption about what the values mean. Values are colored with a
    continuous colormap unless a discrete `palette` is given. For total copy
    number or allele specific states, prefer `plot_cell_tcn_matrix_fig` or
    `plot_cell_ascn_matrix_fig`, which select the matching palette for you.

    Parameters
    ----------
    adata : AnnData
        per cell data with var describing genomic bins
    layer_name : str, optional
        layer with values to plot, None for X, by default None
    tree : Bio.Phylo.BaseTree.Tree, optional
        phylogenetic tree
    cell_order_fields : list, optional
        columns of obs on which to sort cells, by default None
    cell_order : pandas.Index, optional
        explicit cell ids in plot order, from `scgenome.tl.resolve_cell_order`.
        Mutually exclusive with cell_order_fields and tree.
    annotation_fields : list, optional
        column of obs to use as an annotation colorbar, by default 'cluster_id'
    fig : matplotlib.figure.Figure, optional
        existing figure to plot into, by default None
    vmin, vmax : float, optional
        vmin and vmax define the data range that the colormap covers, see `matplotlib.pyplot.imshow`.
        Applies to `cmap` only, discrete palettes map values to colors directly.
    cmap : str or matplotlib.colors.Colormap, optional
        continuous colormap to use, by default 'viridis'. Mutually exclusive
        with palette.
    palette : str or dict, optional
        discrete palette to use, 'cn' for total copy number states,
        'allele_state' for allele specific states, or a dict mapping value to
        color. Mutually exclusive with cmap.
    annotation_cmap, var_annotation_cmap : dict, optional
        colors for each annotation field, keyed by field name. The dtype of
        the column decides how each entry is read: numeric columns take a
        continuous colormap name, categorical columns take either a colormap
        name or a dict mapping level to color.
    show_cell_ids : bool, optional
        show cell ids on heatmap axis, by default False
    show_subsets : bool, optional
        show subset/superset categoricals to allow identification of cell sets
    style : str, optional
        style for spines and chromosome dividing lines and other plot elements,
        by default 'black'

    Returns
    -------
    dict
        Dictionary of plot and data elements

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()

        g = scgenome.pl.plot_cell_matrix_fig(
            adata,
            layer_name='copy', vmin=0, vmax=4,
            cell_order_fields=['cell_order'],
            annotation_fields=['cluster_id', 'sample_id', 'quality'])

    """

    annotation_fields = list(annotation_fields) if annotation_fields is not None else []
    var_annotation_fields = (
        list(var_annotation_fields) if var_annotation_fields is not None else [])
    annotation_cmap = annotation_cmap or {}
    var_annotation_cmap = var_annotation_cmap or {}

    if cell_order is not None and tree is not None:
        raise ValueError('cannot provide both cell_order and tree')

    if cell_order is not None:
        order = pd.Index(cell_order)
    else:
        order = resolve_cell_order(
            adata, fields=cell_order_fields or [], tree=tree)

    if show_subsets:
        # Subsets number the rows as drawn, so they follow display position
        # rather than anything in obs. Copy so the caller's adata is untouched.
        adata = adata.copy()
        position = pd.Series(np.arange(len(order)), index=order).reindex(adata.obs.index)
        adata.obs['superset'] = pd.Series(
            np.floor_divide(position.values, 200), index=adata.obs.index, dtype='category')
        adata.obs['subset'] = pd.Series(
            np.mod(np.floor_divide(position.values, 40), 5),
            index=adata.obs.index, dtype='category')
        annotation_fields = annotation_fields + ['superset', 'subset']

    grid = CellGrid(adata, cell_order=order, tree=tree, fig=fig, style=style)

    if tree is not None:
        grid.add_tree(tree)

    grid.add_heatmap(
        layer_name, name='heatmap', palette=palette, cmap=cmap,
        vmin=vmin, vmax=vmax, show_cell_ids=show_cell_ids)

    if var_annotation_fields:
        grid.add_var_annotation(var_annotation_fields, cmap=var_annotation_cmap)

    if annotation_fields:
        grid.add_obs_annotation(annotation_fields, cmap=annotation_cmap)

    result = grid.plot()

    heat = result.panels['heatmap']

    def _legacy(panel):
        info = {'ax': panel.ax, 'im': panel.im}
        info.update(panel.extras)
        if panel.legend is not None:
            info.update(result.legends.get(panel.legend.title, {}))
        return info

    annotation_info = {}
    for annotation_field in annotation_fields:
        annotation_info[annotation_field] = _legacy(result.panels[annotation_field])
    for annotation_field in var_annotation_fields:
        annotation_info[annotation_field] = _legacy(
            result.panels['heatmap:' + annotation_field])

    legend_info = dict(result.legends.get(heat.legend.title, {}))

    return {
        'fig': result.fig,
        'axes': result.axes['_grid'],
        'tree_ax': result.axes.get('tree'),
        'heatmap_ax': heat.ax,
        'adata': heat.extras['adata'],
        'im': heat.im,
        'legend_info': legend_info,
        'annotation_info': annotation_info,
        'grid': result,
    }


def _warn_if_not_integer(adata, layer_name):
    """ Warn if a layer holds continuous values

    The total copy number palette maps values to colors by equality against
    integer states, so a continuous layer misses every state and renders
    almost entirely white.
    """
    X = adata.layers[layer_name] if layer_name is not None else adata.X
    X = np.asarray(X, dtype=float)

    finite = X[np.isfinite(X)]
    if finite.size == 0 or np.array_equal(finite, np.round(finite)):
        return

    name = layer_name if layer_name is not None else 'X'
    warnings.warn(
        f'{name} holds non integer values, which the total copy number palette '
        f'maps by equality and will render almost entirely white. Use '
        f'plot_cell_matrix for continuous values.',
        UserWarning, stacklevel=3)


def plot_cell_tcn_matrix(adata: AnnData, layer_name='state', **kwargs):
    """ Plot a total copy number matrix

    Colors integer copy number states with the total copy number palette.

    Parameters
    ----------
    adata : AnnData
        copy number data with integer states in `layer_name`
    layer_name : str, optional
        layer with copy number states to plot, None for X, by default 'state'
    **kwargs : dict
        additional arguments passed to `plot_cell_matrix`

    Returns
    -------
    dict
        Dictionary of plot and data elements

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()
        scgenome.pl.plot_cell_tcn_matrix(adata)

    """
    _warn_if_not_integer(adata, layer_name)

    return plot_cell_matrix(
        adata, layer_name=layer_name, palette='cn', **kwargs)


def plot_cell_tcn_matrix_fig(adata: AnnData, layer_name='state', **kwargs):
    """ Plot a total copy number matrix with annotations and a legend

    Colors integer copy number states with the total copy number palette.

    Parameters
    ----------
    adata : AnnData
        copy number data with integer states in `layer_name`
    layer_name : str, optional
        layer with copy number states to plot, None for X, by default 'state'
    **kwargs : dict
        additional arguments passed to `plot_cell_matrix_fig`

    Returns
    -------
    dict
        Dictionary of plot and data elements

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()

        g = scgenome.pl.plot_cell_tcn_matrix_fig(
            adata,
            cell_order_fields=['cell_order'],
            annotation_fields=['cluster_id', 'sample_id', 'quality'])

    """
    _warn_if_not_integer(adata, layer_name)

    return plot_cell_matrix_fig(
        adata, layer_name=layer_name, palette='cn', **kwargs)


def _with_allele_state_layer(adata):
    """ Add the allele state layer if absent, without modifying the input
    """
    if 'allele_state' in adata.layers:
        return adata

    # Copy so that adding the layer does not modify the caller's adata, and so
    # that we are not adding a layer to a view
    return cn_colors.add_allele_state_layer(adata.copy())


def plot_cell_ascn_matrix(adata: AnnData, **kwargs):
    """ Plot an allele specific copy number matrix

    Plots the allele specific state of each bin in each cell, colored by the
    allele state palette. Adds `layers['allele_state']` if not already present.

    Bins with no allele specific copy number are left white. Subset `adata`
    before plotting to drop them, for instance on a `has_allele_cn` column of
    `var` if the allele specific caller provided one.

    Parameters
    ----------
    adata : AnnData
        copy number data with layers['A'] and layers['B']
    **kwargs : dict
        additional arguments passed to `plot_cell_matrix`

    Returns
    -------
    dict
        Dictionary of plot and data elements

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV081_Signals_reduced()
        adata = adata[:, adata.var['has_allele_cn']]
        scgenome.pl.plot_cell_ascn_matrix(adata, cell_order_fields=['cell_order'])

    """
    return plot_cell_matrix(
        _with_allele_state_layer(adata),
        layer_name='allele_state', palette='allele_state', **kwargs)


def plot_cell_ascn_matrix_fig(adata: AnnData, **kwargs):
    """ Plot an allele specific copy number matrix with annotations and legend

    Plots the allele specific state of each bin in each cell, colored by the
    allele state palette. Adds `layers['allele_state']` if not already present.

    Bins with no allele specific copy number are left white. Subset `adata`
    before plotting to drop them, for instance on a `has_allele_cn` column of
    `var` if the allele specific caller provided one.

    Parameters
    ----------
    adata : AnnData
        copy number data with layers['A'] and layers['B']
    **kwargs : dict
        additional arguments passed to `plot_cell_matrix_fig`

    Returns
    -------
    dict
        Dictionary of plot and data elements

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV081_Signals_reduced()
        adata = adata[:, adata.var['has_allele_cn']]

        g = scgenome.pl.plot_cell_ascn_matrix_fig(
            adata,
            cell_order_fields=['cell_order'],
            annotation_fields=['cluster_id', 'n_wgd'])

    """
    return plot_cell_matrix_fig(
        _with_allele_state_layer(adata),
        layer_name='allele_state', palette='allele_state', **kwargs)


def _deprecated_cn_matrix_args(layer_name, cmap, palette, raw):
    """ Translate the pre-palette arguments of plot_cell_cn_matrix

    Reproduces the old behaviour exactly: the total copy number palette by
    default, a continuous colormap when one was given or when raw was set.
    """
    if raw is not None:
        warnings.warn(
            'raw is deprecated, use plot_cell_matrix with a cmap for continuous values',
            DeprecationWarning, stacklevel=3)

    if cmap is None and palette is None:
        if raw:
            cmap = 'viridis'
        else:
            palette = 'cn'

    return dict(layer_name=layer_name, cmap=cmap, palette=palette)


def plot_cell_cn_matrix(adata: AnnData, layer_name='state', cmap=None, palette=None, raw=None, **kwargs):
    """ Plot a copy number matrix

    .. deprecated::
        Use `plot_cell_tcn_matrix` for total copy number states,
        `plot_cell_ascn_matrix` for allele specific states, or
        `plot_cell_matrix` for any other values.
    """
    warnings.warn(
        'plot_cell_cn_matrix is deprecated, use plot_cell_tcn_matrix for total copy '
        'number states or plot_cell_matrix for other values',
        DeprecationWarning, stacklevel=2)

    return plot_cell_matrix(
        adata, **_deprecated_cn_matrix_args(layer_name, cmap, palette, raw), **kwargs)


def plot_cell_cn_matrix_fig(adata: AnnData, layer_name='state', cmap=None, palette=None, raw=None, **kwargs):
    """ Plot a copy number matrix with annotations and a legend

    .. deprecated::
        Use `plot_cell_tcn_matrix_fig` for total copy number states,
        `plot_cell_ascn_matrix_fig` for allele specific states, or
        `plot_cell_matrix_fig` for any other values.
    """
    warnings.warn(
        'plot_cell_cn_matrix_fig is deprecated, use plot_cell_tcn_matrix_fig for total '
        'copy number states or plot_cell_matrix_fig for other values',
        DeprecationWarning, stacklevel=2)

    return plot_cell_matrix_fig(
        adata, **_deprecated_cn_matrix_args(layer_name, cmap, palette, raw), **kwargs)
