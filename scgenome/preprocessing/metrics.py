import numpy as np
import pandas as pd

from anndata import AnnData
from scgenome._validate import validate_adata
# tools.cluster imports preprocessing, so refer to it as a module, resolved at call time
from scgenome.tools import cluster as _cluster


def calculate_local_baf_variance(adata: AnnData, window_size: int = 4) -> AnnData:
    """ Compute local BAF variance using a sliding window on pseudobulk BAF.

    Sums allele counts over all cells to give a pseudobulk BAF per bin, then
    computes the variance of BAF in forward and reverse rolling windows of
    `window_size` bins, taking the smaller of the two. Bins are ordered by
    chr then start, and windows are not broken at chromosome boundaries.
    A bin with low local variance has a consistent pseudobulk BAF with its
    neighbours, which is useful for flagging noisy bins.

    Parameters
    ----------
    adata : anndata.AnnData
        copy number data with allele count layers
    window_size : int, optional
        number of bins in each rolling window, by default 4

    Returns
    -------
    anndata.AnnData
        adata with var['local_baf_variance'] added

    Reads
    -----
    adata.layers['alleleB'] : allele B counts
    adata.layers['totalcounts'] : total allele counts
    adata.var['chr'] : chromosome per bin
    adata.var['start'] : start position per bin

    Modifies
    --------
    adata.var['local_baf_variance'] : local BAF variance per bin, nan for bins
        with no allele counts
    """
    validate_adata(
        adata, require_layers=['alleleB', 'totalcounts'], require_var=['chr', 'start', 'end'],
        caller='calculate_local_baf_variance')

    agg_layers = {
        'alleleB': np.nansum,
        'totalcounts': np.nansum,
    }

    baf = _cluster.aggregate_pseudobulk(adata, agg_layers=agg_layers)
    with np.errstate(divide='ignore', invalid='ignore'):
        baf['BAF'] = baf['alleleB'] / baf['totalcounts']

    baf = baf[['chr', 'start', 'end', 'BAF']].sort_values(['chr', 'start']).dropna()

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
    fwd = baf['BAF'].rolling(window=indexer, min_periods=1).var()
    rev = baf['BAF'].iloc[::-1].rolling(window=indexer, min_periods=1).var().iloc[::-1]
    baf['local_baf_variance'] = np.minimum(
        np.nan_to_num(fwd.values, nan=np.inf),
        np.nan_to_num(rev.values, nan=np.inf))

    adata.var['local_baf_variance'] = baf['local_baf_variance']

    return adata
