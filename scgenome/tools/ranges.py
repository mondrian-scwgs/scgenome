import pyranges as pr
import numpy as np
import pandas as pd
import anndata as ad

from pyranges import PyRanges
from pandas import DataFrame
from anndata import AnnData
from numpy import ndarray

import scgenome.refgenome
from scgenome._deprecate import renamed_arguments




def dataframe_to_pyranges(data: DataFrame) -> PyRanges:
    """ Convert dataframe with chr, start, end to pyranges object

    Parameters
    ----------
    data : DataFrame
        dataframe with columns 'chr', 'start', 'end'

    Returns
    -------
    PyRanges
        pyranges object
    """
    data = pr.PyRanges(data.rename(columns={
        'chr': 'Chromosome',
        'start': 'Start',
        'end': 'End',
    }))

    return data


def pyranges_to_dataframe(data: PyRanges) -> DataFrame:
    """ Convert pyranges object to dataframe

    Parameters
    ----------
    data : PyRanges
        pyranges object

    Returns
    -------
    DataFrame
        dataframe with columns 'chr', 'start', 'end'
    """
    data = data.as_df().rename(columns={
        'Chromosome': 'chr',
        'Start': 'start',
        'End': 'end',
    })

    return data


def create_bins_pr(bin_size: int, genome=None) -> PyRanges:
    """ Create a regular binning of the genome

    Parameters
    ----------
    bin_size : int
        size of bins
    genome : str or RefGenomeInfo, optional
        genome version, by default uses global setting

    Returns
    -------
    PyRanges
        regular bins tiled across the genome
    """    
    genome_info = scgenome.refgenome.get_genome_info(genome=genome)
    chromsizes = genome_info.chromosome_info[['chr', 'chromosome_length']].rename(
        columns={'chr': 'Chromosome', 'chromosome_length': 'End'}).assign(Start=0)[['Chromosome', 'Start', 'End']]

    chromsizes = pr.PyRanges(chromsizes)

    bins = pr.gf.tile_genome(chromsizes, bin_size)

    return bins


@renamed_arguments(binsize='bin_size')
def create_bins(bin_size: int, genome=None) -> DataFrame:
    """ Create a regular binning of the genome

    Parameters
    ----------
    bin_size : int
        size of bins
    genome : str or RefGenomeInfo, optional
        genome version, by default uses global setting

    Returns
    -------
    DataFrame
        regular bins tiled across the genome
    """
    return pyranges_to_dataframe(create_bins_pr(bin_size=bin_size, genome=genome))


def rebin_agg_df(data: DataFrame, intersect: DataFrame, agg_f: dict) -> DataFrame:
    """ Rebin a dataframe, aggregating across multiple intersecting regions

    Parameters
    ----------
    data : DataFrame
        binned data to rebin and aggregate with columns 'chr', 'start', 'end' and index 'bin'
    intersect : DataFrame
        mapping between previous and target bins with columns 'bin', 'target_bin', 'width'
    agg_f : dict of tuples
        aggregate functions, similar to pandas dataframe groupby .agg

    Returns
    -------
    DataFrame
        rebinned and aggregated data with columns according to 'agg_f' and index 'target_bin'
    """
    data = data.loc[intersect['bin'], :]

    data = data.set_index(intersect.set_index('bin')['target_bin'], append=True)
    data = data.set_index(intersect.set_index('bin')['width'], append=True)
    
    data = data.groupby(level='target_bin').agg(**agg_f)

    return data


def rebin_agg_layer(adata: AnnData, intersect: DataFrame, layer_name: str, agg_f: dict) -> DataFrame:
    """ Rebin an anndata layer, aggregating across multiple intersecting regions

    Parameters
    ----------
    adata : DataFrame
        adata to rebin and aggregate with var columns 'chr', 'start', 'end' and index 'bin'
    intersect : DataFrame
        mapping between previous and target bins with columns 'bin', 'target_bin', 'width'
    layer_name : str
        name of layer to rebin and aggregate
    agg_f : dict of tuples
        aggregate functions, similar to pandas dataframe groupby .agg

    Returns
    -------
    DataFrame
        rebinned and aggregated layer data with columns according to 'agg_f' and index 'target_bin'
    """
    data = adata.to_df(layer=layer_name)

    data = data.loc[:, intersect['bin']].T

    data = data.set_index(intersect.set_index('bin')['target_bin'], append=True)
    data = data.set_index(intersect.set_index('bin')['width'], append=True)

    data = data.groupby(level='target_bin').agg(agg_f)
    
    return data.T


class NoIntersectionError(Exception):
    pass


def intersect_regions(a: DataFrame, b: DataFrame) -> DataFrame:
    """ Compute intersection between two sets of regions

    Parameters
    ----------
    a : DataFrame
        region data with columns 'chr', 'start', 'end', ...
    b : DataFrame
        region data with columns 'chr', 'start', 'end', ...

    Returns
    -------
    DataFrame
        intersecting regions with columns 'chr', 'start', 'end' and columns from 'a' and 'b'
    """
    a = scgenome.tools.ranges.dataframe_to_pyranges(a)
    b = scgenome.tools.ranges.dataframe_to_pyranges(b)

    intersect_1 = a.intersect(b)
    intersect_2 = b.intersect(a)

    if intersect_1.empty or intersect_2.empty:
        raise NoIntersectionError()

    intersect = pd.merge(
        scgenome.tools.ranges.pyranges_to_dataframe(intersect_1),
        scgenome.tools.ranges.pyranges_to_dataframe(intersect_2),
        on=['chr', 'start', 'end'])

    return intersect


def intersect_positions(a: DataFrame, b: DataFrame):
    """ Compute a position by cell matrix from a bin by cell matrix and a set of positions

    a position by cell matrix from a bin by cell matrix and a set of positions representing locations of heterozygous SNPs.

    Parameters
    ----------
    a : DataFrame
        region data with columns 'chr', 'start', 'end', ...
    b : DataFrame
        region data with columns 'chr', 'position', ...
    """

    b['start'] = b['position']
    b['end'] = b['position']

    intersect = intersect_regions(a, b)
    intersect = intersect.drop(['start', 'end'], axis=1)

    return intersect


def assign_bins(
        positions: DataFrame,
        bins: DataFrame,
        chr_col: str = 'chromosome',
        position_col: str = 'position') -> pd.Series:
    """ Find the bin containing each of a set of genomic positions

    Bins and positions are both 1-based, and a bin contains positions `start`
    to `end` inclusive, as in DLP HMMCopy bins.

    Parameters
    ----------
    positions : DataFrame
        genomic positions, with chromosome and position columns named by
        `chr_col` and `position_col`, for instance the `var` of an SNV AnnData
        from `scgenome.pp.read_snv_genotyping`
    bins : DataFrame
        bins with columns 'chr', 'start', 'end', indexed by bin ID, for
        instance the `var` of a copy number AnnData
    chr_col : str, optional
        chromosome column of `positions`, by default 'chromosome'
    position_col : str, optional
        position column of `positions`, by default 'position'

    Returns
    -------
    pandas.Series
        ID of the bin containing each position, indexed like `positions`,
        missing for positions outside every bin

    Examples
    --------
    Assign SNVs to the copy number bins of the same cells:

    >>> snv_adata.var['cn_bin'] = scgenome.tl.assign_bins(snv_adata.var, cn_adata.var)
    """
    for name, data, columns in [
            ('positions', positions, [chr_col, position_col]),
            ('bins', bins, ['chr', 'start', 'end'])]:
        missing = [c for c in columns if c not in data.columns]
        if missing:
            raise ValueError(f"assign_bins: {name} missing columns {missing}, has {list(data.columns)}")

    chromosomes = positions[chr_col].astype(str).values
    coords = positions[position_col].values
    bin_chromosomes = bins['chr'].astype(str).values
    assigned = np.full(len(positions), None, dtype=object)

    for chromosome in np.unique(chromosomes):
        idx = np.flatnonzero(chromosomes == chromosome)
        chr_bins = bins[bin_chromosomes == chromosome].sort_values('start')
        if len(chr_bins) == 0:
            continue

        # Last bin starting at or before each position, then check it reaches the position
        bin_idx = np.searchsorted(chr_bins['start'].values, coords[idx], side='right') - 1
        in_bin = (bin_idx >= 0)
        in_bin[in_bin] = coords[idx[in_bin]] <= chr_bins['end'].values[bin_idx[in_bin]]

        assigned[idx[in_bin]] = chr_bins.index.values[bin_idx[in_bin]]

    return pd.Series(assigned, index=positions.index, name=bins.index.name)

def rebin(adata: AnnData, target_bins: DataFrame, outer_join: bool=False, agg_X=None, agg_var=None, agg_layers=()) -> AnnData:
    """ Rebin an AnnData and aggregate across multiple intersecting regions

    Parameters
    ----------
    adata : AnnData
        data to rebin and aggregate
    target_bins : DataFrame
        target bins with columns 'chr', 'start', 'end'
    outer_join : bool, optional
        whether to include target bins with no overlap in adata, by default False
    agg_X : callable, optional
        aggregate function for X, by default None
    agg_var : dict, optional
        aggregate functions for var, by default None
    agg_layers : dict, optional
        aggregate functions for each layer, by default ()

    Returns
    -------
    AnnData
        rebinned and aggregated data
    """
    bins = adata.var.rename_axis('bin').reset_index()[['chr', 'start', 'end', 'bin']]

    target_bins = target_bins.copy()
    target_bins['target_bin'] = (
        target_bins['chr'].astype(str) + ':' +
        target_bins['start'].astype(str) + '-' +
        target_bins['end'].astype(str))

    intersect = intersect_regions(bins, target_bins[['chr', 'start', 'end', 'target_bin']])
    intersect['width'] = intersect['end'] - intersect['start'] + 1

    if agg_var is not None:
        var = rebin_agg_df(adata.var, intersect, agg_var)
    
    else:
        var = intersect[['target_bin']].drop_duplicates().set_index('target_bin')

    var = target_bins.set_index('target_bin').merge(
        var, left_index=True, right_index=True, how=('inner', 'left')[outer_join])

    var = var.sort_values(['chr', 'start'])

    if agg_X is not None:
        X = rebin_agg_layer(adata, intersect, None, agg_X)
        X = X.reindex(columns=var.index)

    else:
        X = None

    layer_data = {}
    for layer_name in agg_layers:
        layer_data[layer_name] = rebin_agg_layer(adata, intersect, layer_name, agg_layers[layer_name])
        layer_data[layer_name] = layer_data[layer_name].reindex(columns=var.index)

    adata = ad.AnnData(
        X,
        obs=adata.obs,
        var=var,
        layers=layer_data,
    )

    return adata


def rebin_regular(adata: AnnData, bin_size: int, outer_join: bool=False, agg_X=None, agg_var=None, agg_layers=()) -> AnnData:
    """ Rebin an AnnData and aggregate across multiple intersecting regions

    Parameters
    ----------
    adata : AnnData
        data to rebin and aggregate
    bin_size : int
        width of target bins
    outer_join : bool, optional
        whether to include target bins with no overlap in adata, by default False
    agg_X : callable, optional
        aggregate function for X, by default None
    agg_var : dict, optional
        aggregate functions for var, by default None
    agg_layers : dict, optional
        aggregate functions for each layer, by default ()

    Returns
    -------
    AnnData
        rebinned and aggregated data
    """
    target_bins = create_bins(bin_size)
    target_bins['start'] = target_bins['start'] + 1

    return rebin(adata, target_bins, outer_join=outer_join, agg_X=agg_X, agg_var=agg_var, agg_layers=agg_layers)


def weighted_mean(values: ndarray, widths: ndarray) -> float:
    """ Compute weighted mean

    Parameters
    ----------
    values : ndarray
        values to compute mean 
    widths : ndarray
        weights corresponding to each value

    Returns
    -------
    float
        weighted mean
    """
    weighted_mean = (widths * values).sum() / widths.sum()
    return weighted_mean


def bin_width_weighted_mean(df: DataFrame) -> float:
    """ Convenience function to allow computing of weighted mean with rebin aggregate functions

    Parameters
    ----------
    df : DataFrame
        group of data for which to compute mean, must have width in index

    Returns
    -------
    float
        weighted mean
    """
    values = df.values
    widths = df.index.get_level_values('width').values
    return weighted_mean(values, widths)


