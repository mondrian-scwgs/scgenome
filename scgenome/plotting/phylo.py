""" Trees and dendrograms drawn beside cell rows.

Both fill one axes and draw rows at ``y == i`` for row ``i`` of the order they
are given, so they line up with a heatmap drawn in the same order. See
:mod:`scgenome.plotting.heatmap` for that convention and
:mod:`scgenome.plotting.results` for how legends are reported.
"""

import warnings

import Bio.Phylo
import matplotlib.pyplot as plt
import numpy as np

from scgenome.tools.ordering import OrderConflict, linkage_order_conflict
from .cn_colors import map_categorical_colors
from .results import PanelResult


def plot_tree(tree, ax=None, linewidth=0.5):
    """ Draw a phylogenetic tree beside heatmap rows

    Draws the tree exactly as given, top to bottom. Rows line up when the row
    order is the tree's own leaf order, which is what
    :func:`~scgenome.tl.tree_leaf_order` returns and what
    :class:`~scgenome.pl.CellGrid` uses when handed a tree. To draw a tree
    against some other order, rotate it first with
    :func:`~scgenome.tl.align_tree_to_order`.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        tree whose leaf names are cell ids, not modified
    ax : matplotlib.axes.Axes, optional
        axes to draw into, by default the current axes
    linewidth : float, optional
        width of tree branches, by default 0.5

    Returns
    -------
    PanelResult
    """
    if ax is None:
        ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

    with plt.rc_context({'lines.linewidth': linewidth}):
        Bio.Phylo.draw(tree, label_func=lambda a: '', axes=ax, do_show=False)

    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlabel('branch length', fontsize=8)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_ylim((tree.count_terminals() + 0.5, 0.5))

    return PanelResult(ax=ax, extras={'tree': tree})


def plot_dendrogram(adata, ax=None, cell_order=None, key='cell_order', color='black',
               linewidth=0.5, orientation='left'):
    """ Draw the hierarchical clustering behind an ordering

    Reads the linkage :func:`~scgenome.tl.sort_cells` stored in
    ``uns['cell_order'][key]``, so the dendrogram describes the same clustering
    the ordering came from rather than a fresh one.

    Leaves are placed at the rows given by ``cell_order``, so the dendrogram
    lines up with a heatmap drawn in that order. If some merge's leaves are
    split by that order its brackets would cross, and
    ``OrderConflict`` is raised instead.

    Parameters
    ----------
    adata : AnnData
        data sorted by :func:`~scgenome.tl.sort_cells`
    ax : matplotlib.axes.Axes, optional
        axes to draw into, by default the current axes
    cell_order : pandas.Index, optional
        cell ids in row order, by default the order the linkage produced
    key : str, optional
        which stored ordering to draw, by default 'cell_order'
    color : str, optional
        branch color, by default 'black'
    linewidth : float, optional
        branch width, by default 0.5
    orientation : str, optional
        'left' to put leaves on the right, adjacent to a heatmap, or 'right'
        for the mirror image, by default 'left'

    Returns
    -------
    PanelResult

    Reads
    -----
    adata.uns['cell_order'][key] : linkage, leaves and ids
    """
    if ax is None:
        ax = plt.gca()

    records = adata.uns.get('cell_order', {})
    if key not in records:
        raise ValueError(
            f'no stored linkage {key!r}, run scgenome.tl.sort_cells first. '
            f'Available: {sorted(records.keys())}')

    record = records[key]
    linkage = np.asarray(record['linkage'])
    ids = list(np.asarray(record['ids']))

    if cell_order is None:
        cell_order = [ids[i] for i in np.asarray(record['leaves'])]
    cell_order = list(cell_order)

    conflict = linkage_order_conflict(linkage, ids, cell_order)
    if conflict is not None:
        lo, hi, n_leaves = conflict
        raise OrderConflict(
            None, lo, hi, n_leaves, remedy=OrderConflict.DENDROGRAM_REMEDY)

    positions = {cell_id: i for i, cell_id in enumerate(cell_order)}
    missing = [i for i in ids if i not in positions]
    if missing:
        raise ValueError(
            f'{len(missing)} cells in the stored linkage are not in cell_order, '
            f'for instance {missing[:3]}')

    n = len(ids)
    node_position = {i: positions[ids[i]] for i in range(n)}
    node_height = {i: 0.0 for i in range(n)}

    for k, row in enumerate(linkage):
        left, right, height = int(row[0]), int(row[1]), float(row[2])

        xs = [node_height[left], height, height, node_height[right]]
        ys = [node_position[left], node_position[left],
              node_position[right], node_position[right]]
        ax.plot(xs, ys, color=color, linewidth=linewidth, solid_joinstyle='miter')

        node_position[n + k] = (node_position[left] + node_position[right]) / 2.
        node_height[n + k] = height

    max_height = float(linkage[:, 2].max()) if len(linkage) else 1.

    ax.set_ylim(len(cell_order) - 0.5, -0.5)
    if orientation == 'left':
        ax.set_xlim(max_height * 1.05, 0)
    else:
        ax.set_xlim(0, max_height * 1.05)

    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlabel('distance', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return PanelResult(ax=ax, extras={'linkage': linkage, 'ids': ids})


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
