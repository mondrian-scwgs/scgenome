""" Lay out several cell panels against one shared row order.

:class:`CellGrid` owns exactly two things: allocating axes and collecting
legends. Everything drawn into those axes comes from
:mod:`scgenome.plotting.panels`.

Three properties follow from doing it in one place. The row order is resolved
once and handed to every panel, so a tree, a dendrogram and any number of
heatmaps are guaranteed to agree. Panel widths are declared and summed once, so
adding an annotation bar cannot resize the matrices next to it. And panels
describe their legends rather than drawing them, so identical legends collapse
into one.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scgenome.tools.ordering import resolve_cell_order
from . import panels as _panels


# Widths are in the same units the old figure layout used, so a grid holding
# one heatmap and some annotations reproduces it exactly
WIDTH_HEATMAP = 1.0
WIDTH_TREE = 0.5
WIDTH_ANNOTATION = 0.02
WIDTH_SPACER = 0.005
HEIGHT_VAR_ANNOTATION = 0.02
HEIGHT_SPACER = 0.01


@dataclass
class _Panel:
    """ A column of the grid, and how to draw it """
    kind: str
    name: str
    width: float
    draw: Callable
    var_fields: List[tuple] = field(default_factory=list)


@dataclass
class GridResult:
    """ What :meth:`CellGrid.plot` drew

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        the figure
    axes : dict
        panel axes keyed by panel name
    panels : dict
        :class:`~scgenome.plotting.panels.PanelResult` keyed by panel name
    legends : dict
        drawn legend elements keyed by legend title
    cell_order : pandas.Index
        the row order every panel was drawn in
    """

    fig: Any
    axes: dict = field(default_factory=dict)
    panels: dict = field(default_factory=dict)
    legends: dict = field(default_factory=dict)
    cell_order: Any = None

    def __getitem__(self, key):
        return getattr(self, key)


class CellGrid:
    """ Build a figure of cell panels sharing one row order

    Parameters
    ----------
    adata : AnnData
        per cell data, not modified. Panels may carry their own AnnData, which
        is reindexed to this object's row order.
    cell_order_fields : list, optional
        obs columns to sort rows on, first is primary
    cell_order : pandas.Index, optional
        explicit cell ids in row order, mutually exclusive with
        ``cell_order_fields``
    tree : Bio.Phylo.BaseTree.Tree, optional
        tree constraining the row order. Does not itself add a tree panel, use
        :meth:`add_tree` for that.
    on_conflict : str, optional
        'raise' to refuse a row order the tree cannot reproduce, 'reorder' to
        let the tree drive it, by default 'raise'
    figsize : tuple, optional
        figure size, used only when creating a figure
    fig : matplotlib.figure.Figure, optional
        existing figure to draw into, by default a new one
    style : str, optional
        'black' or 'white' spines and dividers, by default 'black'

    Examples
    --------

    .. plot::
        :context: close-figs

        import scgenome

        adata = scgenome.datasets.OV2295_HMMCopy_reduced()
        adata = scgenome.tl.sort_cells(adata, layer_name='copy')

        g = (scgenome.pl.CellGrid(adata, cell_order_fields=['cell_order'], figsize=(12, 5))
             .add_dendrogram()
             .add_heatmap('state', palette='cn', name='Total CN')
             .add_heatmap('copy', cmap='viridis', vmin=0, vmax=4, name='Copy')
             .add_obs_annotation(['cluster_id', 'sample_id'])
             .plot())

    """

    def __init__(self, adata, cell_order_fields=None, cell_order=None, tree=None,
                 on_conflict='raise', figsize=None, fig=None, style='black'):
        if cell_order is not None and cell_order_fields:
            raise ValueError(
                'cannot provide both cell_order and cell_order_fields, '
                'cell_order_fields is sugar for resolve_cell_order(adata, fields=...)')

        self.adata = adata
        self.style = style
        self.figsize = figsize
        self.fig = fig

        if cell_order is not None:
            self.cell_order = pd.Index(cell_order)
        else:
            self.cell_order = resolve_cell_order(
                adata, fields=cell_order_fields, tree=tree, on_conflict=on_conflict)

        self._panels = []
        self._tree = tree
        self._on_conflict = on_conflict

    # --- panels ---------------------------------------------------------

    def _add(self, panel):
        self._panels.append(panel)
        return self

    def _unique_name(self, name):
        taken = {p.name for p in self._panels}
        if name not in taken:
            return name
        i = 2
        while f'{name} {i}' in taken:
            i += 1
        return f'{name} {i}'

    def add_heatmap(self, layer=None, adata=None, name=None, legend_title=None,
                    width=WIDTH_HEATMAP, **kwargs):
        """ Add a matrix of per cell values

        Parameters
        ----------
        layer : str, optional
            layer to draw, None for X
        adata : AnnData, optional
            data for this panel, by default the grid's. Reindexed to the grid's
            row order, with blank rows where a cell is absent, which is how two
            samples are compared side by side.
        name : str, optional
            panel name, used to address the axes and panel result
        legend_title : str, optional
            legend title, by default the layer name. Panels describing the
            same values share one legend, so leaving this alone is what lets
            two heatmaps of one palette collapse to a single legend.
        width : float, optional
            relative width of this column
        **kwargs :
            passed to :func:`~scgenome.plotting.panels.heatmap`

        Returns
        -------
        CellGrid
            self, so calls chain
        """
        name = self._unique_name(
            name if name is not None else (layer if layer is not None else 'value'))
        source = adata if adata is not None else self.adata

        # A panel carrying its own data is expected not to hold every cell of
        # the shared order; one over the grid's own data is not
        kwargs.setdefault('on_missing', 'blank' if adata is not None else 'raise')

        def draw(ax, _source=source, _layer=layer, _title=legend_title, _kwargs=kwargs):
            return _panels.heatmap(
                _source, ax, layer=_layer, cell_order=self.cell_order,
                style=self.style, title=_title, **_kwargs)

        return self._add(_Panel('heatmap', name, width, draw))

    def add_obs_annotation(self, fields, cmap=None, width=WIDTH_ANNOTATION):
        """ Add one narrow bar per obs column

        Parameters
        ----------
        fields : str or list
            obs columns to draw
        cmap : dict, optional
            colormaps keyed by field name
        width : float, optional
            relative width of each bar

        Returns
        -------
        CellGrid
            self, so calls chain
        """
        if isinstance(fields, str):
            fields = [fields]
        cmap = cmap or {}

        for f in fields:
            name = self._unique_name(f)

            def draw(ax, _f=f, _cmap=cmap.get(f)):
                return _panels.obs_annotation(
                    self.adata, ax, _f, cell_order=self.cell_order,
                    cmap=_cmap, style=self.style)

            self._add(_Panel('obs_annotation', name, width, draw))

        return self

    def add_var_annotation(self, fields, cmap=None, on=None):
        """ Add one bar per var column, above a heatmap

        Parameters
        ----------
        fields : str or list
            var columns to draw
        cmap : dict, optional
            colormaps keyed by field name
        on : str, optional
            name of the heatmap panel to draw above, by default the most
            recently added one

        Returns
        -------
        CellGrid
            self, so calls chain
        """
        if isinstance(fields, str):
            fields = [fields]
        cmap = cmap or {}

        heatmaps = [p for p in self._panels if p.kind == 'heatmap']
        if not heatmaps:
            raise ValueError('add a heatmap before annotating its bins')

        if on is None:
            target = heatmaps[-1]
        else:
            matching = [p for p in heatmaps if p.name == on]
            if not matching:
                raise ValueError(
                    f'no heatmap named {on!r}, have {[p.name for p in heatmaps]}')
            target = matching[0]

        for f in fields:
            target.var_fields.append((f, cmap.get(f)))

        return self

    def add_tree(self, tree=None, name='tree', width=WIDTH_TREE):
        """ Add a phylogenetic tree beside the rows

        Parameters
        ----------
        tree : Bio.Phylo.BaseTree.Tree, optional
            tree to draw, by default the one given to the constructor
        name : str, optional
            panel name
        width : float, optional
            relative width of this column

        Returns
        -------
        CellGrid
            self, so calls chain
        """
        tree = tree if tree is not None else self._tree
        if tree is None:
            raise ValueError('no tree given, pass one here or to CellGrid')

        name = self._unique_name(name)

        def draw(ax, _tree=tree):
            return _panels.tree(
                _tree, ax, cell_order=self.cell_order, on_conflict=self._on_conflict)

        return self._add(_Panel('tree', name, width, draw))

    def add_dendrogram(self, key='cell_order', name='dendrogram',
                       width=WIDTH_TREE, **kwargs):
        """ Add the hierarchical clustering behind the ordering

        Reads the linkage :func:`~scgenome.tl.sort_cells` stored, so the
        dendrogram describes the clustering the ordering came from.

        Parameters
        ----------
        key : str, optional
            which stored ordering to draw, by default 'cell_order'
        name : str, optional
            panel name
        width : float, optional
            relative width of this column
        **kwargs :
            passed to :func:`~scgenome.plotting.panels.dendrogram`

        Returns
        -------
        CellGrid
            self, so calls chain
        """
        name = self._unique_name(name)

        def draw(ax, _key=key, _kwargs=kwargs):
            return _panels.dendrogram(
                self.adata, ax, cell_order=self.cell_order, key=_key, **_kwargs)

        return self._add(_Panel('dendrogram', name, width, draw))

    # --- layout ---------------------------------------------------------

    def _columns(self):
        """ Panel columns plus the spacer before the first annotation column """
        columns = []
        spaced = False

        for panel in self._panels:
            if panel.kind == 'obs_annotation' and not spaced:
                columns.append(('spacer', None, WIDTH_SPACER))
                spaced = True
            columns.append(('panel', panel, panel.width))

        return columns

    def plot(self):
        """ Allocate the axes, draw every panel, and collect the legends

        Returns
        -------
        GridResult
        """
        if not self._panels:
            raise ValueError('add at least one panel before plotting')

        fig = self.fig
        if fig is None:
            fig = plt.figure(figsize=self.figsize) if self.figsize else plt.figure()

        columns = self._columns()
        width_ratios = [width for _, _, width in columns]

        n_var_rows = max((len(p.var_fields) for p in self._panels), default=0)
        height_ratios = (
            [HEIGHT_VAR_ANNOTATION] * n_var_rows + [HEIGHT_SPACER] + [WIDTH_HEATMAP])
        main_row = len(height_ratios) - 1

        fig_main, fig_legends = fig.subfigures(
            nrows=2, ncols=1, height_ratios=[5, 1], squeeze=True)
        fig_legends.patch.set_alpha(0.0)

        axes = fig_main.subplots(
            nrows=len(height_ratios), ncols=len(width_ratios),
            width_ratios=width_ratios, height_ratios=height_ratios,
            squeeze=False, gridspec_kw=dict(hspace=0.02, wspace=0.02))

        for ax in axes.flatten():
            ax.set_axis_off()

        result = GridResult(fig=fig, cell_order=self.cell_order)
        result.axes['_grid'] = axes

        specs = []

        for col, (kind, panel, _) in enumerate(columns):
            if kind != 'panel':
                continue

            ax = axes[main_row, col]
            ax.set_axis_on()
            drawn = panel.draw(ax)

            result.axes[panel.name] = ax
            result.panels[panel.name] = drawn
            if drawn.legend is not None:
                specs.append((panel.name, drawn.legend))

            # var annotations stack upward from the row above the heatmap
            for i, (var_field, var_cmap) in enumerate(panel.var_fields):
                var_ax = axes[main_row - 2 - i, col]
                var_ax.set_axis_on()
                var_ax.set_xticks([])
                var_drawn = _panels.var_annotation(
                    self.adata, var_ax, var_field, cmap=var_cmap, style=self.style)

                name = f'{panel.name}:{var_field}'
                result.axes[name] = var_ax
                result.panels[name] = var_drawn
                if var_drawn.legend is not None:
                    specs.append((var_field, var_drawn.legend))

        # Identical legends collapse into one, so two heatmaps of the same
        # palette do not produce the same legend twice
        unique = []
        seen = set()
        for name, spec in specs:
            if spec.key in seen:
                continue
            seen.add(spec.key)
            unique.append((name, spec))

        legend_axes = fig_legends.subplots(
            nrows=1, ncols=max(len(unique), 1), squeeze=False)[0]
        for ax in legend_axes:
            ax.set_axis_off()
            ax.patch.set_alpha(0.0)

        for ax, (name, spec) in zip(legend_axes, unique):
            result.legends[spec.title] = _panels.draw_legend(spec, ax)

        result.axes['_legends'] = legend_axes

        return result
