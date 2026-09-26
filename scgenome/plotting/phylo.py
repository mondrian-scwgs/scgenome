import warnings

import Bio.Phylo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from .grid import CellGrid
from .panels import map_categorical_colors


def map_annotations_to_colors(annotation, cmap):
    """ Map annotation values to colors

    .. deprecated::
        Use `scgenome.pl.map_categorical_colors`, which returns the same
        mapping keyed by level.
    """
    warnings.warn(
        'map_annotations_to_colors is deprecated, use map_categorical_colors',
        DeprecationWarning, stacklevel=2)

    values = np.asarray(annotation)
    level_colors, _ = map_categorical_colors(values, cmap=cmap)

    return [level_colors[value] for value in values], level_colors


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
    from scgenome.tools.ordering import align_tree_to_order, tree_leaf_order

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
