import numpy as np

from matplotlib.patches import Patch
from numpy import ndarray
from collections import defaultdict


# Total copy number state palette
color_reference = defaultdict(
    lambda: '#D4B9DA',
    {0: '#3182BD', 1: '#9ECAE1', 2: '#CCCCCC', 3: '#FDCC8A', 4: '#FC8D59', 5: '#E34A33',
     6: '#B30000', 7: '#980043', 8: '#DD1C77', 9: '#DF65B0', 10: '#C994C7', 11: '#D4B9DA'}
)


# Allele state color palette
allele_state_colors = {
    'A-Hom': '#56941E',
    'A-Gained': '#94C773',
    'Balanced': '#d5d5d4',
    'B-Gained': '#7B52AE',
    'B-Hom': '#471871',
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

    Returns
    -------
    Legend
        legend object
    """
    states = []
    patches = []
    for s, h in color_reference.items():
        states.append(s)
        patches.append(Patch(facecolor=h, edgecolor=h))

    ncol = min(3, int(len(states)**(1/2)))

    legend = ax.legend(patches, states, ncol=ncol,
        frameon=frameon, loc=loc, bbox_to_anchor=bbox_to_anchor,
        facecolor='white', edgecolor='white', fontsize='4',
        title=title, title_fontsize='6')
    legend.set_zorder(level=200)

    return legend


def add_allele_state_layer(adata):
    """ Add a layer representing allelic states for plotting.

    Encodes allele-specific states as integers:
    0 = A-Hom (B==0), 1 = A-Gained (A>B), 2 = Balanced (A==B),
    3 = B-Gained (B>A), 4 = B-Hom (A==0)

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
    adata.layers['allele_state'] : integer allele state encoding
    """
    allele_state = np.zeros(adata.shape)
    allele_state[adata.layers['B'] == 0] = 0
    allele_state[(adata.layers['A'] != 0) & (adata.layers['B'] != 0) & (adata.layers['A'] > adata.layers['B'])] = 1
    allele_state[(adata.layers['A'] != 0) & (adata.layers['B'] != 0) & (adata.layers['A'] == adata.layers['B'])] = 2
    allele_state[(adata.layers['A'] != 0) & (adata.layers['B'] != 0) & (adata.layers['B'] > adata.layers['A'])] = 3
    allele_state[adata.layers['A'] == 0] = 4

    adata.layers['allele_state'] = allele_state

    return adata

