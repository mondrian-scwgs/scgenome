"""

usage:
gwplot = GenomeWidePlot(
    data, col_to_plot, plot_func
)
plt.savefig(out_png)

data: dataframe with required cols: chr, start, end
col_to_plot: col in data frame with the data we want to plot
plot_func: function to be plotted


to access the axes object:
> gwplot.ax

(can also be provided when instantiating GenomeWidePlot)


To set x axis to one chromosome only:
> gwplot.set_axis_to_chromosome(chromosome)


Example usage:


import pandas as pd

data = pd.read_csv("small_dataset.csv.gz")

data = data[data['cell_id'] == '130081A-R37-C13']


fig = plt.Figure(figsize=(20,10))
ax = plt.gca()

gwplot = GenomeWidePlot(
    data, 'copy', ax=ax, kind='scatter', hue='state', palette='cn'
)

data['copy'] = data['copy'] + 2
blue_palette = {
            0: '#01529B',
            1: '#01529B',
            2: '#01529B',
            3: '#01529B',
            4: '#01529B',
            5: '#01529B',
            6: '#01529B',
            7: '#01529B',
            8: '#01529B',
            9: '#01529B',
            10: '#01529B',
            11: '#01529B'
}

gwplot = GenomeWidePlot(
    data, 'copy', ax=ax, kind='scatter', hue='state', palette=blue_palette
)


plt.savefig('out.png')

"""
import matplotlib
import numpy as np
import seaborn as sns
from anndata import AnnData
from scgenome import refgenome
from scipy.sparse import issparse
from scgenome.plotting import cn_colors

import matplotlib.pyplot as plt
import matplotlib.units as units
import matplotlib.ticker as ticker
import pandas as pd

refgenome.initialize()


class ChromPos:
    def __init__(self, chrom, pos):
        self.chr = chrom
        self.pos = pos


class ChromPosConverter(units.ConversionInterface):

    @staticmethod
    def convert(value, unit, axis):
        if isinstance(value, ChromPos):
            return value.pos + refgenome.chromosome_starts[value.chr]
        if isinstance(value, pd.Series):
            return value.apply(lambda x: x.pos + refgenome.chromosome_starts[x.chr])
        else:
            return [val.pos + refgenome.chromosome_starts[val.chr] for val in value]

    @staticmethod
    def axisinfo(unit, axis):

        if len(refgenome.plot_chromosomes) == 1:
            chromosome = refgenome.chromosomes[0]
            chromosome_length = refgenome.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_length']
            chromosome_start = refgenome.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_start']
            chromosome_end = refgenome.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_end']
            xticks = np.arange(0, chromosome_length + 2e7, 2e7)
            xticklabels = ['{0:d}M'.format(int(x / 1e6)) for x in xticks]
            xminorticks = np.arange(0, chromosome_length, 1e6)

            label = f'chromosome {refgenome.plot_chromosomes[0]}'
            majfmt = ticker.FixedFormatter(xticklabels)
            majloc = ticker.FixedLocator(xticks + chromosome_start)
            minloc = ticker.FixedLocator(xminorticks + chromosome_start)
            minfmt = ticker.NullFormatter()
            default_limits = [chromosome_start, chromosome_end]
        else:
            positions = refgenome.chromosome_info

            label = 'chromosome'
            majloc = ticker.FixedLocator([0] + positions['chromosome_end'].tolist())
            majfmt = ticker.NullFormatter()
            minloc = ticker.FixedLocator(positions['chromosome_mid'].tolist())
            minfmt = ticker.FixedFormatter(refgenome.plot_chromosomes)
            default_limits = [0, max(positions['chromosome_end'])]

        return units.AxisInfo(
            majloc=majloc,
            majfmt=majfmt,
            minloc=minloc,
            minfmt=minfmt,
            label=label,
            default_limits=default_limits
        )

    @staticmethod
    def default_units(x, axis):
        return x


units.registry[ChromPos] = ChromPosConverter()


class GenomeWidePlot(object):
    def __init__(
            self,
            data,
            plot_function,
            ax=None,
            position_columns=('start',),
            **kwargs
    ):
        self.data = data
        self.kwargs = kwargs
        self.position_columns = position_columns

        if ax is None:
            self.fig = plt.gcf()
            self.ax = plt.gca()
        else:
            self.fig = plt.gcf()
            self.ax = ax

        self.add_chromosome_info()
        plot_function(self.data, ax=ax, **self.kwargs)
        self.ax.spines[['right', 'top']].set_visible(False)

    def add_chromosome_info(self):
        self.data = self.data.merge(refgenome.chromosome_info)
        self.data = self.data[self.data['chr'].isin(refgenome.chromosomes)]
        for columns in self.position_columns:
            self.data[columns] = self.data.apply(lambda x: ChromPos(x.chr, x[columns]), axis=1)

    def set_ylims(self, y_min, y_max):
        self.ax.set_ylim((-0.05 * y_max, y_max))
        self.ax.set_yticks(range(y_min, int(y_max) + 1))
        self.ax.spines['left'].set_bounds(0, y_max)

    def squash_y_axis(self):
        squash_coeff = 0.15
        squash_fwd = lambda a: np.tanh(squash_coeff * a)
        squash_rev = lambda a: np.arctanh(a) / squash_coeff
        self.ax.set_yscale('function', functions=(squash_fwd, squash_rev))

        yticks = np.array([0, 2, 4, 7, 20])
        self.ax.set_yticks(yticks)

        return self.fig

    def annotate_gene(self, gene_name, gene_chr, gene_start, v_locator=0):

        loc = refgenome.chromosome_info.query(f'chr == "{gene_chr}"')['chromosome_start']
        assert len(loc) == 1
        loc = loc.iloc[0] + gene_start

        plt.axvline(loc, color='k', ls=':')
        plt.annotate(gene_name, (loc, 1 + 0.1 * v_locator))

    def chromosome_ticks_to_display(self, chromosomes):
        ticks = [v if v in chromosomes else "" for v in refgenome.plot_chromosomes]
        self.ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(ticks))


def plot_cn_profile(
        adata: AnnData,
        obs_id: str,
        value_layer_name=None,
        state_layer_name=None,
        ax=None,
        max_cn=13,
        s=5,
        squashy=False,
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
    ax : [type], optional
        existing axis to plot into, by default None
    max_cn : int, optional
        max copy number for y axis, by default 13
    chromosome : [type], optional
        single chromosome plot, by default None
    s : int, optional
        size of scatter points, by default 5
    squashy : bool, optional
        compress y axis, by default False
    rawy : bool, optional
        raw data on y axis, by default False

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()
        scgenome.pl.plot_cn_profile(adata, 'SA922-A90554B-R27-C43', value_layer_name='copy', state_layer_name='state')

    TODO: missing return
    """
    cn_data = adata.var.copy()

    if value_layer_name is not None:
        cn_value = adata[[obs_id], :].layers[value_layer_name]
    else:
        cn_value = adata[[obs_id], :].X

    if issparse(cn_value):
        cn_data['value'] = cn_value.toarray()[0]
    else:
        cn_data['value'] = np.array(cn_value)[0]

    # TODO: what if state is sparse
    cn_field_name = None
    if state_layer_name is not None:
        cn_data['state'] = np.array(adata[[obs_id], :].layers[state_layer_name][0])
        cn_field_name = 'state'

    cn_data = cn_data.dropna(subset=['value'])

    def scatterplot(data, y=None, ax=None, **kwargs):
        sns.scatterplot(x='start', y=y, data=data, ax=ax, linewidth=0, legend=False, **kwargs)

    gwp = GenomeWidePlot(
        cn_data,
        scatterplot,
        hue=cn_field_name,
        ax=ax,
        s=s,
        y='value',
        palette=cn_colors.color_reference
    )

    if squashy:
        gwp.squash_y_axis()
    else:
        gwp.set_ylims(0, max_cn)

    return gwp


def plot_var_profile(
        adata,
        value_field_name,
        cn_field_name=None,
        ax=None,
        s=5,
        max_cn=12,
        squashy=False
):
    """Plot scatter points of copy number across the genome or a chromosome.

    Parameters
    ----------
    adata : AnnData
        copy number data
    value_field_name : str, optional
        var field with values for y axis, None for X, by default None
    cn_field_name : str, optional
        var field with states for colors, None for no color by state, by default None
    ax : [type], optional
        existing axis to plot into, by default None
    max_cn : int, optional
        max copy number for y axis, by default 13
    chromosome : [type], optional
        single chromosome plot, by default None
    s : int, optional
        size of scatter points, by default 5
    squashy : bool, optional
        compress y axis, by default False
    rawy : bool, optional
        raw data on y axis, by default False

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()
        scgenome.pl.plot_var_profile(adata[:, adata.var['gc'] > 0], 'gc', rawy=True)

    TODO: missing return
    """

    def scatterplot(data, y=None, ax=None, **kwargs):
        sns.scatterplot(x='start', y=y, data=data, ax=ax, linewidth=0, **kwargs)

    data = adata.var.copy()

    gwp = GenomeWidePlot(
        data,
        scatterplot,
        hue=cn_field_name,
        ax=ax,
        s=s,
        y=value_field_name,
    )

    if squashy:
        gwp.squash_y_axis()
    else:
        gwp.set_ylims(0, max_cn)

    return gwp
