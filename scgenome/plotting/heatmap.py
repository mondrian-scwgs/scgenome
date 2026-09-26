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

    if cmap is not None and palette is not None:
        raise ValueError('cannot provide both cmap and palette')

    if ax is None:
        ax = plt.gca()

    genome_info = scgenome.refgenome.get_genome_info(adata)

    if cell_order is not None and len(cell_order_fields) > 0:
        raise ValueError(
            'cannot provide both cell_order and cell_order_fields, '
            'cell_order_fields is sugar for resolve_cell_order(adata, fields=...)')

    if cell_order is None:
        cell_order = resolve_cell_order(adata, fields=cell_order_fields)

    bin_order = resolve_bin_order(adata, genome=genome_info)

    # Back to positions, so the behaviour matches a positional reorder and a
    # cell id absent from adata is reported rather than silently dropped
    cell_ordering = adata.obs.index.get_indexer(pd.Index(cell_order))
    if (cell_ordering < 0).any():
        unknown = pd.Index(cell_order)[cell_ordering < 0]
        raise ValueError(
            f'{len(unknown)} cells in cell_order are not in adata, for instance '
            f'{list(unknown[:3])}')

    genome_ordering = adata.var.index.get_indexer(bin_order)

    adata = adata[cell_ordering, genome_ordering]

    if layer_name is not None:
        X = adata.layers[layer_name].copy()
    else:
        X = adata.X.copy()

    if palette is not None:
        # Discrete palettes map values to colors directly, bypassing any norm,
        # so the same value gets the same color regardless of what else is in X
        palette_info = cn_colors.resolve_palette(palette)
        X_colors = palette_info['mapper'](X)
        im = ax.imshow(X_colors, aspect='auto', interpolation='none', rasterized=rasterized)

    else:
        palette_info = None
        if cmap is None:
            cmap = 'viridis'
        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]
        im = ax.imshow(X, aspect='auto', cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax, rasterized=rasterized)

    chr_index = (
        genome_info.chromosome_info
        .astype({'chr': str})
        .set_index('chr')['chr_index'])
    mat_chrom_idxs = adata.var['chr'].astype(str).map(chr_index).values.astype(int)
    chrom_boundaries = np.array([0] + list(1 + np.where(mat_chrom_idxs[1:] != mat_chrom_idxs[:-1])[0]) + [mat_chrom_idxs.shape[0]])
    chrom_sizes = chrom_boundaries[1:] - chrom_boundaries[:-1]
    chrom_mids = chrom_boundaries[:-1] + chrom_sizes / 2
    ordered_mat_chrom_idxs = mat_chrom_idxs[np.where(np.array([1] + list(np.diff(mat_chrom_idxs))) != 0)]
    chrom_names = np.array(genome_info.plot_chromosomes)[ordered_mat_chrom_idxs]

    ax.set_xticks(chrom_mids - 0.5)
    ax.set_xticklabels(chrom_names, fontsize='6')
    ax.set_xlabel('chromosome', fontsize=8)

    if show_cell_ids:
        ax.set(yticks=range(len(adata.obs.index)))
        ax.set(yticklabels=adata.obs.index.values)
        ax.tick_params(axis='y', labelrotation=0)
    else:
        ax.set(yticks=[])
        ax.set(yticklabels=[])

    if style == 'black':
        for val in chrom_boundaries[1:-1]:
            ax.axvline(x=val-0.5, linewidth=0.5, color='black', zorder=100)
        ax.spines[:].set_visible(True)
    elif style == 'white':
        for val in chrom_boundaries[1:-1]:
            ax.axvline(x=val-0.5, linewidth=0.5, color='white', zorder=100)
        ax.spines[:].set_visible(False)

    return {
        'ax': ax,
        'im': im,
        'adata': adata,
        'palette_info': palette_info,
    }


def map_catagorigal_colors(values, cmap=None):
    level_colors = None
    cmap_name = None
    if isinstance(cmap, str):
        cmap_name = cmap
    elif isinstance(cmap, collections.abc.Mapping):
        level_colors = cmap

    levels = np.unique(values)
    n_levels = len(levels)

    assert level_colors is None or cmap_name is None, 'cmap_name not necessary if specifying level_colors'

    if level_colors is None:
        if cmap_name is None:
            if n_levels <= 10:
                cmap_name = 'tab10'
            elif n_levels <= 20:
                cmap_name = 'tab20'
            else:
                cmap_name = 'hsv'

        cmap = matplotlib.colormaps[cmap_name]

        level_colors = dict(zip(levels, cmap(np.linspace(0, 1, n_levels))))

    else:
        for l, c in level_colors.items():
            if isinstance(c, str) and c.startswith('#'):
                # Convert to rgba 0-1
                # TODO: refactor colors
                c = c.lstrip('#')
                c = np.array(tuple(np.uint8(int(c[i:i+2], 16)) for i in (0, 2 ,4)) + (255,), dtype=int)
                c = c / 255.
                level_colors[l] = c

    value_colors = np.zeros(values.shape + (4,))
    for l, c in level_colors.items():
        value_colors[values == l, :] = c

    return level_colors, value_colors


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


def _plot_categorical_annotation(values, ax, ax_legend=None, title='', horizontal=False, cmap=None):
    level_colors, value_colors = map_catagorigal_colors(values, cmap=cmap)

    im = ax.imshow(value_colors, aspect='auto', interpolation='none')

    levels = []
    patches = []
    for s, h in level_colors.items():
        levels.append(s)
        patches.append(Patch(facecolor=h, edgecolor=h))
    ncol = min(3, int(len(levels)**(1/2)))

    if ax_legend is not None:
        legend = ax_legend.legend(patches, levels, ncol=ncol,
            frameon=True, loc=2, bbox_to_anchor=(0., 1.),
            facecolor='white', edgecolor='white', fontsize='4',
            title=title, title_fontsize='6')
    else:
        legend = None

    ax.grid(False)
    if horizontal:
        ax.set_yticks([0.], [title], rotation=0, fontsize='6')
        ax.tick_params(axis='x', left=False, right=False)
    else:
        ax.set_xticks([0.], [title], rotation=90, fontsize='6')
        ax.tick_params(axis='y', left=False, right=False)

    annotation_info = {}
    annotation_info['ax'] = ax
    annotation_info['im'] = im
    annotation_info['ax_legend'] = ax_legend
    annotation_info['legend'] = legend
    annotation_info['level_colors'] = level_colors
    annotation_info['value_colors'] = value_colors

    return annotation_info


def _plot_continuous_legend(ax_legend, im, title):
    ax_legend.grid(False)
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])

    axins = ax_legend.inset_axes([0.5, 0.1, 0.05, 0.8])

    cbar = plt.colorbar(im, cax=axins)
    axins.set_title(title, fontsize='6')
    cbar.ax.tick_params(labelsize='4')

    annotation_info = {}
    annotation_info['ax_legend'] = ax_legend
    annotation_info['axins'] = axins
    annotation_info['cbar'] = cbar

    return annotation_info


def _is_discrete(series):
    """ Whether an annotation column should get a discrete rather than continuous colormap.

    Checking against a fixed list of dtype names misses pandas >= 3 string
    columns, whose dtype is named 'str' rather than 'object'. Treat anything
    non-numeric as discrete, and bool as discrete despite being numeric.
    """
    return series.dtype.name == 'bool' or not pd.api.types.is_numeric_dtype(series)


def _plot_continuous_annotation(values, ax, ax_legend, title, horizontal=False, cmap=None):
    if cmap is None:
        cmap = 'Reds'

    im = ax.imshow(values, aspect='auto', interpolation='none', cmap=cmap)

    ax.grid(False)
    if horizontal:
        ax.set_yticks([0.], [title], rotation=0, fontsize='6')
        ax.tick_params(axis='x', left=False, right=False)
    else:
        ax.set_xticks([0.], [title], rotation=90, fontsize='6')
        ax.tick_params(axis='y', left=False, right=False)

    annotation_info = _plot_continuous_legend(ax_legend, im, title)

    annotation_info['ax'] = ax
    annotation_info['im'] = im

    return annotation_info


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

    if fig is None:
        fig = plt.figure()

    if cell_order_fields is None:
        cell_order_fields = []

    # Copied, not aliased: show_subsets appends and must not grow the
    # caller's list across calls
    annotation_fields = list(annotation_fields) if annotation_fields is not None else []

    if annotation_cmap is None:
        annotation_cmap = {}

    var_annotation_fields = list(var_annotation_fields) if var_annotation_fields is not None else []

    if var_annotation_cmap is None:
        var_annotation_cmap = {}

    if tree is not None:
        if cell_order is not None:
            raise ValueError('cannot provide both cell_order and tree')

        # A tree constrains the row order rather than competing with it, so
        # sort fields are allowed and order cells within clades. Resolving
        # reads the tree instead of writing phylo_order onto the caller's
        # adata, and raises OrderConflict if the tree cannot be drawn against
        # the requested order.
        cell_order = resolve_cell_order(adata, fields=cell_order_fields, tree=tree)

        # Rotate a copy of the tree so the drawn leaves match the rows
        tree = align_tree_to_order(tree, cell_order)

        cell_order_fields = []
        num_phylo = 1
        tree_ax_idx = 0
        heatmap_ax_col_idx = 1

    else:
        num_phylo = 0
        heatmap_ax_col_idx = 0

    # Account for number of var annotations above heatmap
    heatmap_ax_row_idx = len(var_annotation_fields) + 1

    # Account for additional annotation fields that will be added after plotting the matrix
    # when we are adding annotation fields to identify cells as per show_subsets
    num_annotations = len(annotation_fields)
    if show_subsets:
        num_annotations += 2

    num_var_annotations = len(var_annotation_fields)

    fig_main, fig_legends = fig.subfigures(nrows=2, ncols=1, height_ratios=[5, 1], squeeze=True)
    fig_legends.patch.set_alpha(0.0)

    width_ratios = [0.5] * num_phylo + [1] + [0.005] + [0.02] * num_annotations
    height_ratios = [0.02] * num_var_annotations + [0.01] + [1]

    axes = fig_main.subplots(
        nrows=len(height_ratios), ncols=len(width_ratios),
        width_ratios=width_ratios, height_ratios=height_ratios,
        squeeze=False, gridspec_kw=dict(hspace=0.02, wspace=0.02))

    # Turn off axes for all annotation rows and columns
    for ax in axes[:heatmap_ax_row_idx, :].flatten():
        ax.set_axis_off()
    for ax in axes[:, heatmap_ax_col_idx+1:].flatten():
        ax.set_axis_off()

    # Re-enable axes and remove ticks for row annotations
    for ax in axes[heatmap_ax_row_idx, heatmap_ax_col_idx+2:].flatten():
        ax.set_axis_on()
        ax.set_yticks([])

    # Re-enable axes and remove ticks for column annotations
    for ax in axes[:heatmap_ax_row_idx-1, heatmap_ax_col_idx].flatten():
        ax.set_axis_on()
        ax.set_xticks([])

    axes_legends = fig_legends.subplots(
        nrows=1, ncols=1+num_annotations+num_var_annotations, squeeze=False)[0]
    for ax in axes_legends:
        ax.set_axis_off()
        ax.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    tree_ax = None
    if tree is not None:
        # Plot phylogenetic tree
        tree_ax = axes[heatmap_ax_row_idx, tree_ax_idx]
        tree_ax.spines['top'].set_visible(False)
        tree_ax.spines['right'].set_visible(False)
        tree_ax.spines['bottom'].set_visible(True)
        tree_ax.spines['left'].set_visible(False)
        with plt.rc_context({'lines.linewidth': 0.5}):
            Bio.Phylo.draw(tree, label_func=lambda a: '', axes=tree_ax, do_show=False)
        tree_ax.tick_params(axis='x', labelsize=6)
        tree_ax.set_xlabel('branch length', fontsize=8)
        tree_ax.set_ylabel('')
        tree_ax.set_yticks([])
        tree_ax.set_ylim((tree.count_terminals() + 0.5, 0.5))

    heatmap_ax = axes[heatmap_ax_row_idx, heatmap_ax_col_idx]
    ax_legend = axes_legends[0]
    g = plot_cell_matrix(
        adata, layer_name=layer_name,
        cell_order_fields=cell_order_fields,
        cell_order=cell_order,
        ax=heatmap_ax, vmin=vmin, vmax=vmax, cmap=cmap, palette=palette,
        show_cell_ids=show_cell_ids,
        style=style)

    adata = g['adata']
    im = g['im']
    palette_info = g['palette_info']

    value_title = layer_name if layer_name is not None else 'value'

    # A discrete palette gets a patch legend of its levels, a continuous
    # colormap gets a colorbar
    if palette_info is not None:
        title = palette_info['title'] if palette_info['title'] is not None else value_title
        legend_info = {'ax_legend': ax_legend}
        legend_info['legend'] = palette_info['legend'](ax_legend, title)

    else:
        legend_info = _plot_continuous_legend(ax_legend, im, value_title)

    if show_subsets:
        # Need to copy the adata to avoid modifying a view
        adata = adata.copy()
        adata.obs['subset'] = pd.Series(np.mod(np.floor_divide(range(adata.shape[0]), 40), 5), index=adata.obs.index, dtype='category')
        adata.obs['superset'] = pd.Series(np.floor_divide(range(adata.shape[0]), 200), index=adata.obs.index, dtype='category')
        annotation_fields = annotation_fields + ['superset', 'subset']

    annotation_info = {}

    for ax, ax_legend, annotation_field in zip(axes[heatmap_ax_row_idx, heatmap_ax_col_idx+2:], axes_legends[1:], annotation_fields):
        if _is_discrete(adata.obs[annotation_field]):
            values = adata.obs[[annotation_field]].values
            annotation_info[annotation_field] = _plot_categorical_annotation(values, ax, ax_legend, annotation_field, cmap=annotation_cmap.get(annotation_field))

        else:
            values = adata.obs[[annotation_field]].values
            annotation_info[annotation_field] = _plot_continuous_annotation(values, ax, ax_legend, annotation_field, cmap=annotation_cmap.get(annotation_field))
        
        if style == 'white':
            ax.spines[:].set_visible(False)

    for ax, ax_legend, annotation_field in zip(axes[:, heatmap_ax_col_idx], axes_legends[1+len(annotation_fields):], var_annotation_fields):
        if _is_discrete(adata.var[annotation_field]):
            values = adata.var[[annotation_field]].copy().values.T
            annotation_info[annotation_field] = _plot_categorical_annotation(values, ax, ax_legend, annotation_field, horizontal=True, cmap=var_annotation_cmap.get(annotation_field))

        else:
            values = adata.var[[annotation_field]].copy().values.T
            annotation_info[annotation_field] = _plot_continuous_annotation(values, ax, ax_legend, annotation_field, horizontal=True, cmap=var_annotation_cmap.get(annotation_field))

        if style == 'white':
            ax.spines[:].set_visible(False)

    return {
        'fig': fig,
        'axes': axes,
        'tree_ax': tree_ax,
        'heatmap_ax': heatmap_ax,
        'adata': adata,
        'im': im,
        'legend_info': legend_info,
        'annotation_info': annotation_info,
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
