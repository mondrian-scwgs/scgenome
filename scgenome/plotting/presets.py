""" Ready made figures, built on :class:`scgenome.pl.CellGrid`.

Each of these resolves an order, allocates a grid, adds the panels a common
figure wants and returns the pieces. They are the top of the stack: they use
the grid, which uses the drawing functions, which use nothing above them.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import gridspec

import Bio.Phylo
from scgenome.tools.ordering import (
    align_tree_to_order, resolve_cell_order, tree_leaf_order)
from .cn_colors import map_categorical_colors
from scgenome._deprecate import renamed_arguments, warn_renamed
from .grid import CellGrid
from .heatmap import (
    _deprecated_cn_matrix_args, _warn_if_not_integer, _with_allele_state_layer)


@renamed_arguments(cell_order='obs_order', cell_order_fields='obs_order_fields',
                   bin_order='var_order', show_cell_ids='show_obs_ids')
def plot_heatmap_fig(
        adata: AnnData,
        layer_name=None,
        tree=None,
        obs_order_fields=None,
        obs_order=None,
        annotation_fields=None,
        annotation_cmap=None,
        var_annotation_fields=None,
        var_annotation_cmap=None,
        fig=None,
        vmin=None,
        vmax=None,
        cmap=None,
        palette=None,
        show_obs_ids=False,
        show_subsets=False,
        style='black'):
    """ Plot a matrix of per cell values with annotations and a legend

    Makes no assumption about what the values mean. Values are colored with a
    continuous colormap unless a discrete `palette` is given. For total copy
    number or allele specific states, prefer `plot_tcn_heatmap_fig` or
    `plot_ascn_heatmap_fig`, which select the matching palette for you.

    Parameters
    ----------
    adata : AnnData
        per cell data with var describing genomic bins
    layer_name : str, optional
        layer with values to plot, None for X, by default None
    tree : Bio.Phylo.BaseTree.Tree, optional
        phylogenetic tree
    obs_order_fields : list, optional
        columns of obs on which to sort cells, by default None
    obs_order : pandas.Index, optional
        explicit cell ids in plot order, from `scgenome.tl.resolve_cell_order`.
        Mutually exclusive with obs_order_fields and tree: a tree is already
        an order, so reconciling one with another is
        `scgenome.tl.align_tree_to_order`, called explicitly.
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
    show_obs_ids : bool, optional
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
            obs_order_fields=['cell_order'],
            annotation_fields=['cluster_id', 'sample_id', 'quality'])

    """

    annotation_fields = list(annotation_fields) if annotation_fields is not None else []
    var_annotation_fields = (
        list(var_annotation_fields) if var_annotation_fields is not None else [])
    annotation_cmap = annotation_cmap or {}
    var_annotation_cmap = var_annotation_cmap or {}

    given = []
    if obs_order is not None:
        given.append('obs_order')
    if obs_order_fields:
        given.append('obs_order_fields')
    if tree is not None:
        given.append('tree')

    if len(given) > 1:
        raise ValueError(
            f'rows can only be ordered one way, but {given} were given. A tree is '
            f'already an order; to make one agree with another, rotate it first '
            f'with scgenome.tl.align_tree_to_order.')

    if tree is not None:
        order = pd.Index(tree_leaf_order(tree))
    elif obs_order is not None:
        order = pd.Index(obs_order)
    else:
        order = resolve_cell_order(adata, fields=obs_order_fields or [])

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

    grid = CellGrid(adata, obs_order=order, fig=fig, style=style)

    if tree is not None:
        grid.add_tree(tree)

    grid.add_heatmap(
        layer_name, name='heatmap', palette=palette, cmap=cmap,
        vmin=vmin, vmax=vmax, show_obs_ids=show_obs_ids)

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


def plot_tcn_heatmap_fig(adata: AnnData, layer_name='state', **kwargs):
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

        g = scgenome.pl.plot_tcn_heatmap_fig(
            adata,
            obs_order_fields=['cell_order'],
            annotation_fields=['cluster_id', 'sample_id', 'quality'])

    """
    _warn_if_not_integer(adata, layer_name)

    return plot_heatmap_fig(
        adata, layer_name=layer_name, palette='cn', **kwargs)


def plot_ascn_heatmap_fig(adata: AnnData, **kwargs):
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

        g = scgenome.pl.plot_ascn_heatmap_fig(
            adata,
            obs_order_fields=['cell_order'],
            annotation_fields=['cluster_id', 'n_wgd'])

    """
    return plot_heatmap_fig(
        _with_allele_state_layer(adata),
        layer_name='allele_state', palette='allele_state', **kwargs)


def plot_cell_cn_matrix_fig(adata: AnnData, layer_name='state', cmap=None, palette=None, raw=None, **kwargs):
    """ Plot a copy number matrix with annotations and a legend

    .. deprecated::
        Use `plot_tcn_heatmap_fig` for total copy number states,
        `plot_ascn_heatmap_fig` for allele specific states, or
        `plot_heatmap_fig` for any other values.
    """
    warnings.warn(
        'plot_cell_cn_matrix_fig is deprecated, use plot_tcn_heatmap_fig for total '
        'copy number states or plot_heatmap_fig for other values',
        DeprecationWarning, stacklevel=2)

    return plot_heatmap_fig(
        adata, **_deprecated_cn_matrix_args(layer_name, cmap, palette, raw), **kwargs)


def plot_tree_cn(
        tree,
        adata,
        chrom_segments=True,
        layer_name=None,
        obs_annotation=None,
        obs_cmap=None,
        var_label=None,
        fig=None,
        cmap=None,
        palette=None,
        raw=None,
        max_cn=None):
    """ Plot a tree aligned to a CN values matrix heatmap

    .. deprecated::
        Use :class:`~scgenome.pl.CellGrid`, which places a tree beside any
        number of heatmaps against one shared row order::

            g = (scgenome.pl.CellGrid(adata, tree=tree)
                 .add_tree()
                 .add_heatmap('state', palette='cn')
                 .add_obs_annotation(['cluster_id'])
                 .plot())

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        phylogenetic tree
    adata : AnnData
        Copy number data, either genes or segments
    chrom_segments : bool, optional
        whether adata is genes or segments, by default True
    layer_name : str, optional
        layer to plot for copy number heatmap, by default None
    obs_annotation : str, optional
        column of adata.obs to annotate cells, by default None
    obs_cmap : matplotlib.colors.Colormap, optional
        color map for cell annotations, by default None
    var_label : str, optional
        column of adata.var to use for heatmap var labels, by default None use index
    fig : matplotlib.figure.Figure, optional
        existing figure to plot into, by default None
    raw : bool, optional
        raw plotting, no integer color map, by default False
    max_cn : int, optional
        clip cn at max value, by default 13

    Returns
    -------
    matplotlib.figure.Figure
        the figure drawn into
    """
    warnings.warn(
        'plot_tree_cn is deprecated, use scgenome.pl.CellGrid with .add_tree() '
        'and .add_heatmap()',
        DeprecationWarning, stacklevel=2)

    if fig is None:
        fig = plt.figure(figsize=(16, 12), dpi=150)

    if chrom_segments:
        if raw is not None:
            warnings.warn(
                'raw is deprecated, pass cmap for a continuous colormap or '
                "palette='cn' for total copy number states",
                DeprecationWarning, stacklevel=2)

        if max_cn is not None:
            warnings.warn(
                'max_cn has no effect and will be removed',
                DeprecationWarning, stacklevel=2)

        if cmap is None and palette is None and raw is False:
            palette = 'cn'

        grid = CellGrid(adata, tree=tree, fig=fig)
        grid.add_tree(tree)
        grid.add_heatmap(layer_name, name='heatmap', cmap=cmap, palette=palette)

        if obs_annotation is not None:
            grid.add_obs_annotation(
                obs_annotation, cmap={obs_annotation: obs_cmap} if obs_cmap else None)

        grid.plot()

        return fig

    # Gene level data has no genomic bins to lay out, so it keeps the old
    # seaborn path rather than moving onto CellGrid
    order = tree_leaf_order(tree)
    aligned = align_tree_to_order(tree, order)

    gs = gridspec.GridSpec(1, 3, width_ratios=(0.4, 0.58, 0.02))

    ax = fig.add_subplot(gs[0, 0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    Bio.Phylo.draw(aligned, label_func=lambda a: '', axes=ax, do_show=False)

    ax = fig.add_subplot(gs[0, 1])

    positions = adata.obs.index.get_indexer(order)
    mat = adata[positions, :].to_df(layer=layer_name)
    if var_label is not None:
        mat = mat.rename(columns=adata.var[var_label])

    cbar_ax = fig.add_axes([0.45, 0.0, 0.2, 0.01])
    sns.heatmap(mat, ax=ax, cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'})
    ax.set_yticks([])

    if obs_annotation is not None:
        ax = fig.add_subplot(gs[0, 2])
        values = adata.obs[obs_annotation].reindex(order).values.reshape(-1, 1)
        _, value_colors = map_categorical_colors(values, cmap=obs_cmap)
        ax.imshow(value_colors, aspect='auto', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(left=0.065, right=0.97, top=0.96, bottom=0.065, wspace=0.01)

    return fig


def plot_cell_matrix_fig(adata: AnnData, **kwargs):
    """ Plot a cell by bin matrix with annotations and a legend

    .. deprecated::
        Use `plot_heatmap_fig`.
    """
    warn_renamed('plot_cell_matrix_fig', 'plot_heatmap_fig', stacklevel=2)
    return plot_heatmap_fig(adata, **kwargs)


def plot_cell_tcn_matrix_fig(adata: AnnData, layer_name='state', **kwargs):
    """ Plot a total copy number matrix with annotations and a legend

    .. deprecated::
        Use `plot_tcn_heatmap_fig`.
    """
    warn_renamed('plot_cell_tcn_matrix_fig', 'plot_tcn_heatmap_fig', stacklevel=2)
    return plot_tcn_heatmap_fig(adata, layer_name=layer_name, **kwargs)


def plot_cell_ascn_matrix_fig(adata: AnnData, **kwargs):
    """ Plot an allele specific copy number matrix with annotations and a legend

    .. deprecated::
        Use `plot_ascn_heatmap_fig`.
    """
    warn_renamed('plot_cell_ascn_matrix_fig', 'plot_ascn_heatmap_fig', stacklevel=2)
    return plot_ascn_heatmap_fig(adata, **kwargs)
