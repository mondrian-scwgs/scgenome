""" Panels that draw one thing into one axes.

Each panel takes an axes, draws, and returns a :class:`PanelResult`. None of
them create a figure, allocate axes, or decide where anything goes, so a caller
can lay them out however it likes. :class:`scgenome.pl.CellGrid` is one such
caller.

Panels describe their legend rather than drawing it. Deferring the draw is what
lets a layout collect legends from several panels, drop duplicates, and size a
single legend strip before anything is rendered.

All panels draw rows in the order given by ``cell_order``, with row ``i`` at
``y == i``, matching the row coordinates of an imshow. Handing one order to
several panels is therefore enough to guarantee their rows line up.
"""

import collections.abc
from dataclasses import dataclass, field
from typing import Any, List, Optional

import Bio.Phylo
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

import scgenome.refgenome
from scgenome.tools.ordering import (
    OrderConflict, linkage_order_conflict, resolve_bin_order)
from . import cn_colors


@dataclass
class LegendSpec:
    """ A legend a panel needs, described rather than drawn

    Parameters
    ----------
    kind : str
        'patches' for a discrete palette, 'colorbar' for a continuous one
    title : str
        legend title
    levels : list, optional
        values a 'patches' legend labels
    colors : list, optional
        colors a 'patches' legend shows, parallel to ``levels``
    mappable : matplotlib.cm.ScalarMappable, optional
        artist a 'colorbar' legend draws from
    """

    kind: str
    title: str
    levels: Optional[List[Any]] = None
    colors: Optional[List[Any]] = None
    mappable: Any = None

    @property
    def key(self):
        """ Identity used to drop duplicate legends

        Two heatmaps of the same palette describe the same legend, and a layout
        should show it once.
        """
        if self.kind == 'patches':
            return ('patches', self.title,
                    tuple(str(level) for level in self.levels or ()),
                    tuple(str(color) for color in self.colors or ()))

        norm = getattr(self.mappable, 'norm', None)
        cmap = getattr(self.mappable, 'cmap', None)
        return ('colorbar', self.title, getattr(cmap, 'name', None),
                getattr(norm, 'vmin', None), getattr(norm, 'vmax', None))


@dataclass
class PanelResult:
    """ What a panel drew

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes drawn into
    im : matplotlib.image.AxesImage, optional
        image artist, for panels that draw one
    legend : LegendSpec, optional
        legend this panel needs, for the layout to draw
    extras : dict
        panel specific detail, for instance the ordered adata a heatmap drew
    """

    ax: Axes
    im: Any = None
    legend: Optional[LegendSpec] = None
    extras: dict = field(default_factory=dict)


def _ordered(adata, cell_order):
    """ Rows of adata in cell_order, with blanks where a cell is absent

    Returns the reordered adata and a boolean mask of rows that exist, so a
    panel can leave the others empty rather than dropping them and falling out
    of step with the other panels.
    """
    cell_order = pd.Index(cell_order)
    positions = adata.obs.index.get_indexer(cell_order)
    present = positions >= 0

    return adata[positions[present]], present


def _blank_rows(values, present, fill=np.nan):
    """ Expand per present row values back to one row per requested cell """
    full = np.full((len(present),) + values.shape[1:], fill, dtype=float)
    full[present] = values
    return full


def heatmap(
        adata,
        ax,
        layer=None,
        cell_order=None,
        bin_order=None,
        palette=None,
        cmap=None,
        vmin=None,
        vmax=None,
        show_cell_ids=False,
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
    ax : matplotlib.axes.Axes
        axes to draw into
    layer : str, optional
        layer to draw, None for X
    cell_order : pandas.Index, optional
        cell ids in row order, by default the existing obs order
    bin_order : pandas.Index, optional
        bin ids in column order, by default genomic order
    palette : str or dict, optional
        discrete palette, mutually exclusive with cmap
    cmap : str or Colormap, optional
        continuous colormap, mutually exclusive with palette
    vmin, vmax : float, optional
        data range the colormap covers
    show_cell_ids : bool, optional
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
        what to do with a cell in ``cell_order`` that is not in ``adata``:
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

    genome_info = scgenome.refgenome.get_genome_info(adata, genome=genome)

    if bin_order is None:
        bin_order = resolve_bin_order(adata, genome=genome_info)

    bin_positions = adata.var.index.get_indexer(pd.Index(bin_order))
    if (bin_positions < 0).any():
        missing = pd.Index(bin_order)[bin_positions < 0]
        raise ValueError(
            f'{len(missing)} bins in bin_order are not in adata, for instance '
            f'{list(missing[:3])}')

    adata = adata[:, bin_positions]

    if cell_order is None:
        cell_order = adata.obs.index
        present = np.ones(adata.shape[0], dtype=bool)
        ordered = adata
    else:
        ordered, present = _ordered(adata, cell_order)

        if not present.all() and on_missing == 'raise':
            absent = pd.Index(cell_order)[~present]
            raise ValueError(
                f'{len(absent)} cells in cell_order are not in adata, for instance '
                f"{list(absent[:3])}. Pass on_missing='blank' to draw them as "
                f'empty rows instead.')

    X = np.asarray(
        ordered.layers[layer] if layer is not None else ordered.X, dtype=float)

    if not present.all():
        X = _blank_rows(X, present)

    value_title = title if title is not None else (layer if layer is not None else 'value')

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

    if show_cell_ids:
        ax.set(yticks=range(len(cell_order)))
        ax.set(yticklabels=list(cell_order))
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


def map_categorical_colors(values, cmap=None):
    """ Map categorical values to colors

    Parameters
    ----------
    values : numpy.ndarray
        values to map
    cmap : str, Colormap or dict, optional
        colormap name or instance, or a mapping of level to color

    Returns
    -------
    tuple
        ``(level_colors, value_colors)``
    """
    level_colors = None
    cmap_name = None
    colormap = None
    if isinstance(cmap, str):
        cmap_name = cmap
    elif isinstance(cmap, collections.abc.Mapping):
        level_colors = dict(cmap)
    elif isinstance(cmap, matplotlib.colors.Colormap):
        colormap = cmap

    levels = np.unique(values)
    n_levels = len(levels)

    if level_colors is None:
        if cmap_name is None and colormap is None:
            if n_levels <= 10:
                cmap_name = 'tab10'
            elif n_levels <= 20:
                cmap_name = 'tab20'
            else:
                cmap_name = 'hsv'

        if colormap is None:
            colormap = matplotlib.colormaps[cmap_name]
        level_colors = dict(zip(levels, colormap(np.linspace(0, 1, n_levels))))

    else:
        for level, color in list(level_colors.items()):
            if isinstance(color, str) and color.startswith('#'):
                level_colors[level] = np.array(
                    list(cn_colors.hex_to_rgb(color)) + [255], dtype=float) / 255.

    value_colors = np.zeros(values.shape + (4,))
    for level, color in level_colors.items():
        value_colors[values == level, :] = color

    return level_colors, value_colors


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


def obs_annotation(adata, ax, field, cell_order=None, cmap=None, style='black'):
    """ Draw a vertical bar of one obs column, one row per cell

    Parameters
    ----------
    adata : AnnData
        per cell data
    ax : matplotlib.axes.Axes
        axes to draw into
    field : str
        obs column to draw
    cell_order : pandas.Index, optional
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
    if field not in adata.obs.columns:
        raise ValueError(
            f'missing obs column {field!r}. '
            f'Available obs columns: {list(adata.obs.columns)}')

    values = adata.obs[field]
    if cell_order is not None:
        values = values.reindex(pd.Index(cell_order))

    return _annotation(values, ax, field, horizontal=False, cmap=cmap, style=style)


def var_annotation(adata, ax, field, bin_order=None, cmap=None, style='black'):
    """ Draw a horizontal bar of one var column, one column per bin

    Parameters
    ----------
    adata : AnnData
        data with var describing genomic bins
    ax : matplotlib.axes.Axes
        axes to draw into
    field : str
        var column to draw
    bin_order : pandas.Index, optional
        bin ids in column order, by default genomic order
    cmap : str or dict, optional
        colormap name, or a level to color mapping for categorical columns
    style : str, optional
        'black' or 'white' spines, by default 'black'

    Returns
    -------
    PanelResult
    """
    if field not in adata.var.columns:
        raise ValueError(
            f'missing var column {field!r}. '
            f'Available var columns: {list(adata.var.columns)}')

    values = adata.var[field]
    if bin_order is None:
        bin_order = resolve_bin_order(adata)
    values = values.reindex(pd.Index(bin_order))

    return _annotation(values, ax, field, horizontal=True, cmap=cmap, style=style)


def tree(tree, ax, cell_order=None, linewidth=0.5, on_conflict='raise'):
    """ Draw a phylogenetic tree aligned to heatmap rows

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        tree whose leaf names are cell ids, not modified
    ax : matplotlib.axes.Axes
        axes to draw into
    cell_order : pandas.Index, optional
        cell ids in row order. The tree is rotated to match, and
        :class:`~scgenome.tl.OrderConflict` is raised if no rotation does.
    linewidth : float, optional
        width of tree branches, by default 0.5
    on_conflict : str, optional
        passed to :func:`~scgenome.tl.align_tree_to_order`, by default 'raise'

    Returns
    -------
    PanelResult
    """
    from scgenome.tools.ordering import align_tree_to_order

    if cell_order is not None:
        tree = align_tree_to_order(tree, cell_order, on_conflict=on_conflict)

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


def dendrogram(adata, ax, cell_order=None, key='cell_order', color='black',
               linewidth=0.5, orientation='left'):
    """ Draw the hierarchical clustering behind an ordering

    Reads the linkage :func:`~scgenome.tl.sort_cells` stored in
    ``uns['cell_order'][key]``, so the dendrogram describes the same clustering
    the ordering came from rather than a fresh one.

    Leaves are placed at the rows given by ``cell_order``, so the dendrogram
    lines up with a heatmap drawn in that order. If some merge's leaves are
    split by that order its brackets would cross, and
    :class:`~scgenome.tl.OrderConflict` is raised instead.

    Parameters
    ----------
    adata : AnnData
        data sorted by :func:`~scgenome.tl.sort_cells`
    ax : matplotlib.axes.Axes
        axes to draw into
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
        raise OrderConflict(None, lo, hi, n_leaves)

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


def draw_legend(spec, ax, title=None):
    """ Render a :class:`LegendSpec` into an axes

    Parameters
    ----------
    spec : LegendSpec
        legend to draw
    ax : matplotlib.axes.Axes
        axes to draw into
    title : str, optional
        override the spec's title

    Returns
    -------
    dict
        the drawn elements, keyed 'legend' for patches or 'cbar' for colorbars
    """
    title = title if title is not None else spec.title

    if spec.kind == 'patches':
        patches = [Patch(facecolor=c, edgecolor=c) for c in spec.colors]
        ncol = min(3, int(max(len(spec.levels), 1) ** (1 / 2)))
        legend = ax.legend(
            patches, spec.levels, ncol=ncol,
            frameon=True, loc=2, bbox_to_anchor=(0., 1.),
            facecolor='white', edgecolor='white', fontsize='4',
            title=title, title_fontsize='6')
        legend.set_zorder(level=200)
        return {'ax_legend': ax, 'legend': legend}

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    axins = ax.inset_axes([0.5, 0.1, 0.05, 0.8])
    cbar = plt.colorbar(spec.mappable, cax=axins)
    axins.set_title(title, fontsize='6')
    cbar.ax.tick_params(labelsize='4')

    return {'ax_legend': ax, 'axins': axins, 'cbar': cbar}
