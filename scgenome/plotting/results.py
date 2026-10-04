""" What the drawing functions return.

A drawing function fills one axes and reports what it drew. It *describes* the
legend it needs as a :class:`LegendSpec` rather than drawing one, which is what
lets a layout collect legends from several of them, drop duplicates, and size a
single legend strip before anything is rendered. :func:`draw_legend` renders a
spec once the layout has decided where it goes.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch


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
    legend : scgenome.pl.LegendSpec, optional
        legend this panel needs, for the layout to draw
    extras : dict
        per function detail, for instance the ordered adata a heatmap drew

    Notes
    -----
    Supports ``result['ax']`` as well as ``result.ax``, and reaches into
    ``extras``, so the dictionaries these functions used to return keep working.
    """

    ax: Axes
    im: Any = None
    legend: Optional[LegendSpec] = None
    extras: dict = field(default_factory=dict)

    # Mapping access so the dicts these replaced keep working
    _LEGACY = ('ax', 'im', 'adata', 'palette_info', 'legend', 'extras')

    def __getitem__(self, key):
        if key in ('ax', 'im', 'legend', 'extras'):
            return getattr(self, key)
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)

    def __contains__(self, key):
        return key in ('ax', 'im', 'legend', 'extras') or key in self.extras

    def keys(self):
        return [k for k in self._LEGACY if k in self]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def draw_legend(spec, ax=None, title=None):
    """ Render a :class:`LegendSpec` into an axes

    Parameters
    ----------
    spec : LegendSpec
        legend to draw
    ax : matplotlib.axes.Axes, optional
        axes to draw into, by default the current axes
    title : str, optional
        override the spec's title

    Returns
    -------
    dict
        the drawn elements, keyed 'legend' for patches or 'cbar' for colorbars
    """
    if ax is None:
        ax = plt.gca()

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
