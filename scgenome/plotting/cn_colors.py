import numpy as np
import collections.abc

from matplotlib.patches import Patch
from numpy import ndarray
from collections import defaultdict


# Total copy number state palette
color_reference = defaultdict(
    lambda: '#D4B9DA',
    {0: '#3182BD', 1: '#9ECAE1', 2: '#CCCCCC', 3: '#FDCC8A', 4: '#FC8D59', 5: '#E34A33',
     6: '#B30000', 7: '#980043', 8: '#DD1C77', 9: '#DF65B0', 10: '#C994C7', 11: '#D4B9DA'}
)


# Allele state names, ordered such that the index of each name is the integer
# code used in layers['allele_state'], see `add_allele_state_layer`
allele_state_names = [
    'A-Hom',
    'A-Gained',
    'Balanced',
    'B-Gained',
    'B-Hom',
]


# Allele state color palette, keyed by state name
allele_state_colors = {
    'A-Hom': '#56941E',
    'A-Gained': '#94C773',
    'Balanced': '#d5d5d4',
    'B-Gained': '#7B52AE',
    'B-Hom': '#471871',
}


# Allele state color palette, keyed by the integer codes of layers['allele_state']
allele_state_code_colors = {
    code: allele_state_colors[name] for code, name in enumerate(allele_state_names)
}


def hex_to_rgb(h):
    if h is None:
        return np.array((0, 0, 0), dtype=int)
    h = h.lstrip('#')
    return np.array(tuple(np.uint8(int(h[i:i+2], 16)) for i in (0, 2 ,4)), dtype=int)


def map_cn_colors(X: ndarray) -> ndarray:
    """ Create an array of colors from an array of copy number states

    Parameters
    ----------
    X : ndarray
        copy number states

    Returns
    -------
    ndarray
        colors with shape X.shape + (3,)
    """
    X_colors = np.zeros(X.shape + (3,), dtype=int)
    X_colors[:] = hex_to_rgb('#ffffff')
    X_colors[X < 0, :] = hex_to_rgb(color_reference[0])
    X_colors[X > max(color_reference.keys()), :] = hex_to_rgb(color_reference[max(color_reference.keys())])
    for state, hex in color_reference.items():
        X_colors[X == state, :] = hex_to_rgb(hex)
    return X_colors


def map_discrete_colors(X: ndarray, palette: dict, default='#ffffff') -> ndarray:
    """ Create an array of colors by looking values up in a palette

    Values are mapped to colors by equality against the palette keys, so the
    resulting colors do not depend on which values happen to be present in
    `X`. Values absent from the palette, including nan, are given `default`.

    Parameters
    ----------
    X : ndarray
        values to map
    palette : dict
        mapping of value to hex color
    default : str, optional
        hex color for values absent from the palette, by default white

    Returns
    -------
    ndarray
        colors with shape X.shape + (3,)
    """
    X_colors = np.zeros(X.shape + (3,), dtype=int)
    X_colors[:] = hex_to_rgb(default)
    for value, hex in palette.items():
        X_colors[X == value, :] = hex_to_rgb(hex)
    return X_colors


def map_allele_state_colors(X: ndarray) -> ndarray:
    """ Create an array of colors from an array of allele specific states

    Bins with no allele specific copy number, encoded as nan by
    `add_allele_state_layer`, are colored white.

    Parameters
    ----------
    X : ndarray
        allele states, integer coded as per `add_allele_state_layer`

    Returns
    -------
    ndarray
        colors with shape X.shape + (3,)
    """
    return map_discrete_colors(X, allele_state_code_colors)


def _patch_legend(ax, labels, colors, title, frameon=True, loc=2, bbox_to_anchor=(0., 1.)):
    """ Display a legend of labelled color patches
    """
    patches = [Patch(facecolor=c, edgecolor=c) for c in colors]

    ncol = min(3, int(len(labels)**(1/2)))

    legend = ax.legend(patches, labels, ncol=ncol,
        frameon=frameon, loc=loc, bbox_to_anchor=bbox_to_anchor,
        facecolor='white', edgecolor='white', fontsize='4',
        title=title, title_fontsize='6')
    legend.set_zorder(level=200)

    return legend


def cn_legend(ax, frameon=True, loc=2, bbox_to_anchor=(0., 1.), title='Copy Number'):
    """ Display a legend for copy number state colors

    Parameters
    ----------
    ax : Axes
        matplotlib Axes on which to show legend
    frameon : bool, optional
        show frame, by default True
    loc : int, optional
        location of the legend, by default 2
    bbox_to_anchor : tuple, optional
        bounding box to which to anchor legend location, by default (0., 1.)
    title : str, optional
        title of the legend, by default 'Copy Number'

    Returns
    -------
    Legend
        legend object
    """
    states = list(color_reference.keys())
    colors = [color_reference[s] for s in states]

    return _patch_legend(
        ax, states, colors, title,
        frameon=frameon, loc=loc, bbox_to_anchor=bbox_to_anchor)


def allele_state_legend(ax, frameon=True, loc=2, bbox_to_anchor=(0., 1.), title='AS CN state'):
    """ Display a legend for allele specific state colors

    Parameters
    ----------
    ax : Axes
        matplotlib Axes on which to show legend
    frameon : bool, optional
        show frame, by default True
    loc : int, optional
        location of the legend, by default 2
    bbox_to_anchor : tuple, optional
        bounding box to which to anchor legend location, by default (0., 1.)
    title : str, optional
        title of the legend, by default 'AS CN state'

    Returns
    -------
    Legend
        legend object
    """
    colors = [allele_state_colors[n] for n in allele_state_names]

    return _patch_legend(
        ax, allele_state_names, colors, title,
        frameon=frameon, loc=loc, bbox_to_anchor=bbox_to_anchor)


def resolve_palette(palette):
    """ Resolve a discrete palette specification

    Parameters
    ----------
    palette : str or dict
        'cn' for the total copy number palette, 'allele_state' for the allele
        specific state palette, or a dict mapping value to hex color

    Returns
    -------
    dict
        'mapper' is a callable from an array of values to an array of colors
        with shape values.shape + (3,), 'legend' is a callable taking an Axes
        and a title and returning a Legend, 'title' is the preferred legend
        title for this palette or None to let the caller choose, and 'levels'
        and 'colors' are the parallel lists a legend labels
    """
    if palette == 'cn':
        states = list(color_reference.keys())
        return {
            'mapper': map_cn_colors,
            'legend': lambda ax, title: cn_legend(ax, title=title),
            'title': None,
            'levels': states,
            'colors': [color_reference[s] for s in states],
        }

    elif palette == 'allele_state':
        return {
            'mapper': map_allele_state_colors,
            'legend': lambda ax, title: allele_state_legend(ax, title=title),
            'title': 'AS CN state',
            'levels': list(allele_state_names),
            'colors': [allele_state_colors[n] for n in allele_state_names],
        }

    elif isinstance(palette, collections.abc.Mapping):
        levels = list(palette.keys())
        colors = [palette[l] for l in levels]
        return {
            'mapper': lambda X: map_discrete_colors(X, palette),
            'legend': lambda ax, title: _patch_legend(ax, levels, colors, title),
            'title': None,
            'levels': levels,
            'colors': colors,
        }

    raise ValueError(f"unknown palette {palette!r}, expected 'cn', 'allele_state' or a dict")


def add_allele_state_layer(adata):
    """ Add a layer representing allelic states for plotting.

    Encodes allele-specific states as integers matching the index of each
    state in `allele_state_names`:
    0 = A-Hom (B==0), 1 = A-Gained (A>B), 2 = Balanced (A==B),
    3 = B-Gained (B>A), 4 = B-Hom (A==0)

    Bins for which either allele has no copy number, for instance those
    excluded by allele specific copy number calling, are encoded as nan.

    Parameters
    ----------
    adata : anndata.AnnData
        annotated data with layers['A'] and layers['B']

    Returns
    -------
    anndata.AnnData
        adata with layers['allele_state'] added

    Reads
    -----
    adata.layers['A'] : allele A copy number states
    adata.layers['B'] : allele B copy number states

    Modifies
    --------
    adata.layers['allele_state'] : integer allele state encoding, nan where
        allele specific copy number is unavailable
    """
    for layer_name in ('A', 'B'):
        if layer_name not in adata.layers:
            raise ValueError(
                f"missing layer {layer_name!r}, available layers are {list(adata.layers.keys())}")

    A = np.asarray(adata.layers['A'], dtype=float)
    B = np.asarray(adata.layers['B'], dtype=float)

    allele_state = np.full(adata.shape, np.nan)
    allele_state[B == 0] = 0
    allele_state[(A != 0) & (B != 0) & (A > B)] = 1
    allele_state[(A != 0) & (B != 0) & (A == B)] = 2
    allele_state[(A != 0) & (B != 0) & (B > A)] = 3
    allele_state[A == 0] = 4

    # Bins with no allele specific copy number have no state
    allele_state[~np.isfinite(A) | ~np.isfinite(B)] = np.nan

    adata.layers['allele_state'] = allele_state

    return adata
