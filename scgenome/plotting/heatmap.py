""" Matrices of per cell values, and the annotation bars that flank them.

:func:`plot_heatmap` and the annotation bars each fill one axes and nothing
else: no figure is created and no layout decided. :class:`scgenome.pl.CellGrid`
arranges them into a composed figure, but nothing here depends on it and any of
them works on its own.

Two conventions make them compose. Rows are drawn in the order given by
``obs_order``, with row ``i`` at ``y == i``, matching the row coordinates of
an imshow, so handing one order to several of them is enough to line their rows
up. And each describes its legend rather than drawing it, see
:mod:`scgenome.plotting.results`.
"""

import warnings

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

import scgenome.refgenome
from scgenome._deprecate import renamed_arguments
from scgenome.tools.ordering import resolve_bin_order, resolve_cell_order
from . import cn_colors
from .cn_colors import map_categorical_colors
from .results import LegendSpec, PanelResult


def _ordered(adata, obs_order):
    """ Rows of adata in obs_order, with blanks where a cell is absent

    Returns the reordered adata and a boolean mask of rows that exist, so a
    panel can leave the others empty rather than dropping them and falling out
    of step with the other panels.
    """
    obs_order = pd.Index(obs_order)
    positions = adata.obs.index.get_indexer(obs_order)
    present = positions >= 0

    return adata[positions[present]], present


def _blank_rows(values, present, fill=np.nan):
    """ Expand per present row values back to one row per requested cell """
    full = np.full((len(present),) + values.shape[1:], fill, dtype=float)
    full[present] = values
    return full


@renamed_arguments(cell_order='obs_order', cell_order_fields='obs_order_fields',
                   bin_order='var_order', show_cell_ids='show_obs_ids')
def plot_heatmap(
        adata,
        layer_name=None,
        ax=None,
        obs_order=None,
        obs_order_fields=(),
        var_order=None,
        palette=None,
        cmap=None,
        vmin=None,
        vmax=None,
        show_obs_ids=False,
        style='black',
        rasterized=False,
        title=None,
        genome=None,
        on_missing='raise'):
    """ Draw a matrix of per cell values across the genome

    Parameters
    ----------
    adata : AnnData
        per cell data with var describing genomic bins
    layer_name : str, optional
        layer to draw, None for X
    ax : matplotlib.axes.Axes, optional
        axes to draw into, by default the current axes
    obs_order : pandas.Index, optional
        cell ids in row order, by default the existing obs order
    obs_order_fields : list, optional
        obs columns to sort rows on, sugar for
        ``resolve_cell_order(adata, fields=...)``. Mutually exclusive with
        ``obs_order``.
    var_order : pandas.Index, optional
        bin ids in column order, by default genomic order
    palette : str or dict, optional
        discrete palette, mutually exclusive with cmap
    cmap : str or matplotlib.colors.Colormap, optional
        continuous colormap, mutually exclusive with palette
    vmin, vmax : float, optional
        data range the colormap covers
    show_obs_ids : bool, optional
        label rows with cell ids, by default False
    style : str, optional
        'black' or 'white' chromosome dividers and spines, by default 'black'
    rasterized : bool, optional
        rasterize the image, by default False
    title : str, optional
        legend title, by default the layer name
    genome : str or RefGenomeInfo, optional
        genome version, by default resolved from adata
    on_missing : str, optional
        what to do with a cell in ``obs_order`` that is not in ``adata``:
        'raise', or 'blank' to draw an empty row. A panel carrying its own
        AnnData wants 'blank', since a shared order spans cells it may not
        have; a panel over the order's own AnnData wants 'raise', since a
        missing cell there is a mistake. By default 'raise'.

    Returns
    -------
    PanelResult
    """
    if on_missing not in ('raise', 'blank'):
        raise ValueError(
            f"unknown on_missing {on_missing!r}, expected 'raise' or 'blank'")
    if cmap is not None and palette is not None:
        raise ValueError('cannot provide both cmap and palette')

    if ax is None:
        ax = plt.gca()

    if obs_order is not None and len(obs_order_fields) > 0:
        raise ValueError(
            'cannot provide both obs_order and obs_order_fields, '
            'obs_order_fields is sugar for resolve_cell_order(adata, fields=...)')

    if obs_order is None and len(obs_order_fields) > 0:
        obs_order = resolve_cell_order(adata, fields=obs_order_fields)

    genome_info = scgenome.refgenome.get_genome_info(adata, genome=genome)

    if var_order is None:
        var_order = resolve_bin_order(adata, genome=genome_info)

    bin_positions = adata.var.index.get_indexer(pd.Index(var_order))
    if (bin_positions < 0).any():
        missing = pd.Index(var_order)[bin_positions < 0]
        raise ValueError(
            f'{len(missing)} bins in var_order are not in adata, for instance '
            f'{list(missing[:3])}')

    adata = adata[:, bin_positions]

    if obs_order is None:
        obs_order = adata.obs.index
        present = np.ones(adata.shape[0], dtype=bool)
        ordered = adata
    else:
        ordered, present = _ordered(adata, obs_order)

        if not present.all() and on_missing == 'raise':
            absent = pd.Index(obs_order)[~present]
            raise ValueError(
                f'{len(absent)} cells in obs_order are not in adata, for instance '
                f"{list(absent[:3])}. Pass on_missing='blank' to draw them as "
                f'empty rows instead.')

    X = np.asarray(
        ordered.layers[layer_name] if layer_name is not None else ordered.X,
        dtype=float)

    if not present.all():
        X = _blank_rows(X, present)

    value_title = title if title is not None else (
        layer_name if layer_name is not None else 'value')

    if palette is not None:
        # Discrete palettes map values to colors directly, bypassing any norm,
        # so the same value gets the same color regardless of what else is in X
        palette_info = cn_colors.resolve_palette(palette)
        X_colors = palette_info['mapper'](X)
        im = ax.imshow(
            X_colors, aspect='auto', interpolation='none', rasterized=rasterized)

        legend = LegendSpec(
            kind='patches',
            title=palette_info['title'] if palette_info['title'] is not None else value_title,
            levels=palette_info['levels'],
            colors=palette_info['colors'])

    else:
        palette_info = None
        if cmap is None:
            cmap = 'viridis'
        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]
        im = ax.imshow(
            X, aspect='auto', cmap=cmap, interpolation='none',
            vmin=vmin, vmax=vmax, rasterized=rasterized)

        legend = LegendSpec(kind='colorbar', title=value_title, mappable=im)

    chr_index = (
        genome_info.chromosome_info
        .astype({'chr': str})
        .set_index('chr')['chr_index'])
    mat_chrom_idxs = ordered.var['chr'].astype(str).map(chr_index).values.astype(int)
    chrom_boundaries = np.array(
        [0]
        + list(1 + np.where(mat_chrom_idxs[1:] != mat_chrom_idxs[:-1])[0])
        + [mat_chrom_idxs.shape[0]])
    chrom_sizes = chrom_boundaries[1:] - chrom_boundaries[:-1]
    chrom_mids = chrom_boundaries[:-1] + chrom_sizes / 2
    ordered_mat_chrom_idxs = mat_chrom_idxs[
        np.where(np.array([1] + list(np.diff(mat_chrom_idxs))) != 0)]
    chrom_names = np.array(genome_info.plot_chromosomes)[ordered_mat_chrom_idxs]

    ax.set_xticks(chrom_mids - 0.5)
    ax.set_xticklabels(chrom_names, fontsize='6')
    ax.set_xlabel('chromosome', fontsize=8)

    if show_obs_ids:
        ax.set(yticks=range(len(obs_order)))
        ax.set(yticklabels=list(obs_order))
        ax.tick_params(axis='y', labelrotation=0)
    else:
        ax.set(yticks=[])
        ax.set(yticklabels=[])

    divider_color = 'white' if style == 'white' else 'black'
    for val in chrom_boundaries[1:-1]:
        ax.axvline(x=val - 0.5, linewidth=0.5, color=divider_color, zorder=100)
    ax.spines[:].set_visible(style != 'white')

    return PanelResult(
        ax=ax, im=im, legend=legend,
        extras={
            'adata': ordered,
            'palette_info': palette_info,
            'present': present,
            'chrom_boundaries': chrom_boundaries,
        })


def _is_discrete(series):
    """ Whether an annotation column takes a discrete rather than continuous map

    Checking against a fixed list of dtype names misses pandas >= 3 string
    columns, whose dtype is named 'str' rather than 'object'. Treat anything
    non-numeric as discrete, and bool as discrete despite being numeric.
    """
    return series.dtype.name == 'bool' or not pd.api.types.is_numeric_dtype(series)


def _annotation(values, ax, title, horizontal, cmap, style):
    """ Draw one annotation bar, discrete or continuous by dtype """
    if _is_discrete(values):
        array = values.values.reshape((1, -1) if horizontal else (-1, 1))
        level_colors, value_colors = map_categorical_colors(array, cmap=cmap)
        im = ax.imshow(value_colors, aspect='auto', interpolation='none')

        legend = LegendSpec(
            kind='patches', title=title,
            levels=list(level_colors.keys()),
            colors=list(level_colors.values()))
        extras = {'level_colors': level_colors, 'value_colors': value_colors}

    else:
        array = np.asarray(values.values, dtype=float)
        array = array.reshape((1, -1) if horizontal else (-1, 1))
        im = ax.imshow(
            array, aspect='auto', interpolation='none',
            cmap=cmap if cmap is not None else 'Reds')

        legend = LegendSpec(kind='colorbar', title=title, mappable=im)
        extras = {}

    ax.grid(False)
    if horizontal:
        # The bar is one row, so only the title belongs on the cross axis and
        # the long axis carries the heatmap's own ticks, not a copy of them
        ax.set_yticks([0.], [title], rotation=0, fontsize='6')
        ax.set_xticks([])
        ax.tick_params(axis='x', left=False, right=False)
    else:
        ax.set_xticks([0.], [title], rotation=90, fontsize='6')
        ax.set_yticks([])
        ax.tick_params(axis='y', left=False, right=False)

    if style == 'white':
        ax.spines[:].set_visible(False)

    return PanelResult(ax=ax, im=im, legend=legend, extras=extras)


@renamed_arguments(cell_order='obs_order', cell_order_fields='obs_order_fields', bin_order='var_order')
def plot_obs_annotation(adata, field, ax=None, obs_order=None, cmap=None,
                        style='black'):
    """ Draw a vertical bar of one obs column, one row per cell

    Parameters
    ----------
    adata : AnnData
        per cell data
    field : str
        obs column to draw
    ax : matplotlib.axes.Axes, optional
        axes to draw into, by default the current axes
    obs_order : pandas.Index, optional
        cell ids in row order, by default the existing obs order
    cmap : str or dict, optional
        colormap name for numeric columns, or a colormap name or level to
        color mapping for categorical ones
    style : str, optional
        'black' or 'white' spines, by default 'black'

    Returns
    -------
    PanelResult
    """
    if ax is None:
        ax = plt.gca()

    if field not in adata.obs.columns:
        raise ValueError(
            f'missing obs column {field!r}. '
            f'Available obs columns: {list(adata.obs.columns)}')

    values = adata.obs[field]
    if obs_order is not None:
        values = values.reindex(pd.Index(obs_order))

    return _annotation(values, ax, field, horizontal=False, cmap=cmap, style=style)


@renamed_arguments(cell_order='obs_order', cell_order_fields='obs_order_fields', bin_order='var_order')
def plot_var_annotation(adata, field, ax=None, var_order=None, cmap=None,
                        style='black'):
    """ Draw a horizontal bar of one var column, one column per bin

    Parameters
    ----------
    adata : AnnData
        data with var describing genomic bins
    field : str
        var column to draw
    ax : matplotlib.axes.Axes, optional
        axes to draw into, by default the current axes
    var_order : pandas.Index, optional
        bin ids in column order, by default genomic order
    cmap : str or dict, optional
        colormap name, or a level to color mapping for categorical columns
    style : str, optional
        'black' or 'white' spines, by default 'black'

    Returns
    -------
    PanelResult
    """
    if ax is None:
        ax = plt.gca()

    if field not in adata.var.columns:
        raise ValueError(
            f'missing var column {field!r}. '
            f'Available var columns: {list(adata.var.columns)}')

    values = adata.var[field]
    if var_order is None:
        var_order = resolve_bin_order(adata)
    values = values.reindex(pd.Index(var_order))

    return _annotation(values, ax, field, horizontal=True, cmap=cmap, style=style)


@renamed_arguments(cell_order='obs_order', cell_order_fields='obs_order_fields',
                   bin_order='var_order', show_cell_ids='show_obs_ids')
def plot_cell_matrix(
        adata: AnnData,
        layer_name=None,
        obs_order_fields=(),
        obs_order=None,
        ax=None,
        vmin=None,
        vmax=None,
        cmap=None,
        palette=None,
        show_obs_ids=False,
        style='black',
        rasterized=False):
    """ Plot a matrix of per cell values across the genome

    .. deprecated::
        Renamed to `scgenome.pl.plot_heatmap`, since the rows are not always
        cells and the plot is a heatmap either way. The result still supports
        `result['ax']` as well as `result.ax`.
    """
    warnings.warn(
        'plot_cell_matrix is deprecated, use plot_heatmap',
        DeprecationWarning, stacklevel=2)

    return plot_heatmap(
        adata,
        layer_name=layer_name,
        ax=ax,
        obs_order=obs_order,
        obs_order_fields=obs_order_fields,
        palette=palette,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_obs_ids=show_obs_ids,
        style=style,
        rasterized=rasterized)


def map_catagorigal_colors(values, cmap=None):
    """ Map categorical values to colors

    .. deprecated::
        Misspelled, use `scgenome.pl.map_categorical_colors`.
    """
    warnings.warn(
        'map_catagorigal_colors is deprecated, use map_categorical_colors',
        DeprecationWarning, stacklevel=2)

    return map_categorical_colors(values, cmap=cmap)


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
        f'plot_heatmap for continuous values.',
        UserWarning, stacklevel=3)


def plot_tcn_heatmap(adata: AnnData, layer_name='state', **kwargs):
    """ Plot a total copy number heatmap

    Colors integer copy number states with the total copy number palette.

    Parameters
    ----------
    adata : AnnData
        copy number data with integer states in `layer_name`
    layer_name : str, optional
        layer with copy number states to plot, None for X, by default 'state'
    **kwargs : dict
        additional arguments passed to `scgenome.pl.plot_heatmap`

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
        scgenome.pl.plot_tcn_heatmap(adata)

    """
    _warn_if_not_integer(adata, layer_name)

    return plot_heatmap(adata, layer_name=layer_name, palette='cn', **kwargs)


def _with_allele_state_layer(adata):
    """ Add the allele state layer if absent, without modifying the input
    """
    if 'allele_state' in adata.layers:
        return adata

    # Copy so that adding the layer does not modify the caller's adata, and so
    # that we are not adding a layer to a view
    return cn_colors.add_allele_state_layer(adata.copy())


def plot_ascn_heatmap(adata: AnnData, **kwargs):
    """ Plot an allele specific copy number heatmap

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
        additional arguments passed to `scgenome.pl.plot_heatmap`

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
        scgenome.pl.plot_ascn_heatmap(adata, obs_order_fields=['cell_order'])

    """
    return plot_heatmap(
        _with_allele_state_layer(adata),
        layer_name='allele_state', palette='allele_state', **kwargs)


def _deprecated_cn_matrix_args(layer_name, cmap, palette, raw):
    """ Translate the pre-palette arguments of plot_cell_cn_matrix

    Reproduces the old behaviour exactly: the total copy number palette by
    default, a continuous colormap when one was given or when raw was set.
    """
    if raw is not None:
        warnings.warn(
            'raw is deprecated, use plot_heatmap with a cmap for continuous values',
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
        Use `plot_tcn_heatmap` for total copy number states,
        `plot_ascn_heatmap` for allele specific states, or
        `plot_heatmap` for any other values.
    """
    warnings.warn(
        'plot_cell_cn_matrix is deprecated, use plot_tcn_heatmap for total copy '
        'number states or plot_heatmap for other values',
        DeprecationWarning, stacklevel=2)

    return plot_heatmap(
        adata, **_deprecated_cn_matrix_args(layer_name, cmap, palette, raw), **kwargs)


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

