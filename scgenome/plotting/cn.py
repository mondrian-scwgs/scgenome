import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from pandas import DataFrame
from scgenome import refgenome
from scipy.sparse import issparse
from scgenome.plotting import cn_colors
from scgenome.tools.getters import get_obs_data


def genome_axis_plot(data, plot_function, position_columns, **kwargs):
    data = data.merge(refgenome.info.chromosome_info)
    for columns in position_columns:
        data[columns] = data[columns] + data['chromosome_start']

    plot_function(data=data, **kwargs)


def setup_genome_xaxis_lims(ax, chromosome=None, start=None, end=None):
    if chromosome is not None:
        chromosome_start = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_start']
        chromosome_end = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_end']

        if start is not None:
            plot_start = chromosome_start + start
        else:
            plot_start = chromosome_start

        if end is not None:
            plot_end = chromosome_start + end
        else:
            plot_end = chromosome_end

    else:
        plot_start = 0
        plot_end = refgenome.info.chromosome_info['chromosome_end'].max()

    ax.set_xlim((plot_start-0.5, plot_end+0.5))


def setup_genome_xaxis_ticks(ax, chromosome=None, start=None, end=None, major_spacing=2e7, minor_spacing=1e6, chromosome_names=None):
    if chromosome_names is None:
        chromosome_names = {}
    
    if chromosome is not None:
        if major_spacing is None:
            major_spacing = 2e7

        if minor_spacing is None:
            minor_spacing = 1e6

        chromosome_length = refgenome.info.chromosome_info.set_index('chr').loc[
            chromosome, 'chromosome_length']
        chromosome_start = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_start']
        chromosome_end = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_end']

        xticks = np.arange(0, chromosome_length, major_spacing)
        xticklabels = ['{0:d}M'.format(int(x / 1e6)) for x in xticks]
        xminorticks = np.arange(0, chromosome_length, minor_spacing)

        ax.set_xticks(xticks + chromosome_start)
        ax.set_xticklabels(xticklabels)

        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(xminorticks + chromosome_start))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    else:
        ax.set_xticks([0] + refgenome.info.chromosome_info['chromosome_end'].values.tolist())
        ax.set_xticklabels([])

        ax.xaxis.set_minor_locator(
            matplotlib.ticker.FixedLocator(refgenome.info.chromosome_info['chromosome_mid'])
        )
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter([chromosome_names.get(c, c) for c in refgenome.info.chromosomes]))


def setup_squash_yaxis(ax):
    squash_coeff = 0.15
    squash_fwd = lambda a: np.tanh(squash_coeff * a)
    squash_rev = lambda a: np.arctanh(a) / squash_coeff
    ax.set_yscale('function', functions=(squash_fwd, squash_rev))

    yticks = np.array([0, 2, 4, 7, 20])
    ax.set_yticks(yticks)

    # Matplotlib will set problematic ylim if there are large y values
    ylim = ax.get_ylim()
    ax.set_ylim(-0.25, ylim[1])
 

def plot_profile(
        data: DataFrame,
        y,
        hue=None,
        ax=None,
        palette=None,
        chromosome=None,
        start=None,
        end=None,
        squashy=False,
        tick_major_spacing=None,
        tick_minor_spacing=None,
        **kwargs
):
    """Plot scatter points of copy number across the genome or a chromosome.

    Parameters
    ----------
    data : pandas.DataFrame
        copy number data
        observation to plot
    y : str
        field with values for y axis
    hue : str, optional
        field by which to color points, None for no color, by default None
    ax : matplotlib.axes.Axes, optional
        existing axess to plot into, by default None
    palette : str, optional
        color palette passed to sns.scatterplot
    chromosome : str, optional
        single chromosome plot, by default None
    start : int, optional
        start of plotting region
    end : int, optional
        end of plotting region
    squashy : bool, optional
        compress y axis, by default False
    rawy : bool, optional
        raw data on y axis, by default False
    tick_major_spacing : int, optional
        major tick spacing, by default 
    tick_minor_spacing : int, optional
        minor tick spacing, by default 1e6
    **kwargs :
        kwargs for sns.scatterplot

    Returns
    -------
    matplotlib.axes.Axes
        Axes used for plotting

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()
        scgenome.pl.plot_profile(adata[:, adata.var['gc'] > 0], 'gc')

    """

    if ax is None:
        ax = plt.gca()

    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 0

    if 's' not in kwargs:
        kwargs['s'] = 5

    if palette is None and hue is not None:
        palette = cn_colors.color_reference

    genome_axis_plot(
        data,
        sns.scatterplot,
        ('start',),
        x='start',
        y=y,
        hue=hue,
        palette=palette,
        ax=ax,
        **kwargs)

    setup_genome_xaxis_ticks(
        ax,
        chromosome=chromosome,
        start=start,
        end=end,
        major_spacing=tick_major_spacing,
        minor_spacing=tick_minor_spacing,
    )

    setup_genome_xaxis_lims(
        ax,
        chromosome=chromosome,
        start=start,
        end=end,
    )

    if squashy:
        setup_squash_yaxis(ax)

    if chromosome is not None:
        ax.set_xlabel(f'Chromosome {chromosome}')

    else:
        ax.set_xlabel('Chromosome')

    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    return ax


def plot_cn_profile(
        adata: AnnData,
        obs_id: str,
        value_layer_name=None,
        state_layer_name=None,
        ax=None,
        palette=None,
        chromosome=None,
        start=None,
        end=None,
        squashy=False,
        tick_major_spacing=None,
        tick_minor_spacing=None,
        **kwargs
):
    """Plot scatter points of copy number across the genome or a chromosome.

    Parameters
    ----------
    adata : AnnData
        copy number data
    obs_id : str
        observation to plot
    value_layer_name : str, optional
        layer with values for y axis, None for X, by default None
    state_layer_name : str, optional
        layer with states for colors, None for no color by state, by default None
    ax : matplotlib.axes.Axes, optional
        existing axess to plot into, by default None
    palette : str, optional
        color palette passed to sns.scatterplot
    chromosome : str, optional
        single chromosome plot, by default None
    start : int, optional
        start of plotting region
    end : int, optional
        end of plotting region
    squashy : bool, optional
        compress y axis, by default False
    rawy : bool, optional
        raw data on y axis, by default False
    tick_major_spacing : int, optional
        major tick spacing, by default 
    tick_minor_spacing : int, optional
        minor tick spacing, by default 1e6
    **kwargs :
        kwargs for sns.scatterplot

    Returns
    -------
    matplotlib.axes.Axes
        Axes used for plotting

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()
        scgenome.pl.plot_cn_profile(adata, 'SA922-A90554B-R27-C43', value_layer_name='copy', state_layer_name='state')

    """

    if value_layer_name is None:
        value_layer_name = '_X'

    layers = {value_layer_name}

    if state_layer_name is not None:
        layers.add(state_layer_name)

    cn_data = get_obs_data(
        adata,
        obs_id,
        ['chr', 'start'],
        layer_names=layers)

    return plot_profile(
        cn_data,
        y=value_layer_name,
        hue=state_layer_name,
        ax=ax,
        palette=palette,
        chromosome=chromosome,
        start=start,
        end=end,
        squashy=squashy,
        tick_major_spacing=tick_major_spacing,
        tick_minor_spacing=tick_minor_spacing,
        **kwargs)
