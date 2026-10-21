import numpy as np
import pandas as pd

from anndata import AnnData
from scgenome._validate import validate_adata


def compute_cn_changepoint_matrix(adata: AnnData) -> pd.DataFrame:
    """ Compute copy number changepoints in each chromosome and in each cell.

    For each chromosome, computes bins where adjacent integer copy number
    states differ (transition points). Bins with no GC content, and bins with
    mean copy number of 10 or more across cells, are removed first.

    Parameters
    ----------
    adata : anndata.AnnData
        copy number data with layers['state'] and layers['copy'], and var
        columns chr, start, gc

    Returns
    -------
    pandas.DataFrame
        boolean changepoint matrix, index=bin IDs, columns=cell IDs. A bin is
        True for a cell if its state differs from that of the next retained
        bin on the same chromosome; the last retained bin of each chromosome
        is omitted.

    Reads
    -----
    adata.layers['state'] : integer copy number states
    adata.layers['copy'] : continuous copy number (for filtering)
    adata.var['gc'] : GC content per bin
    adata.var['chr'] : chromosome per bin
    adata.var['start'] : start position per bin

    Modifies
    --------
    nothing, returns a new DataFrame
    """
    validate_adata(
        adata, require_layers=['state', 'copy'], require_var=['chr', 'start', 'gc'],
        caller='compute_cn_changepoint_matrix')

    # Remove poor gc regions
    adata_filtered = adata[:, adata.var['gc'] > 0]

    # Filter high copy regions
    mean_bin_copy = np.array(adata_filtered.layers['copy'].mean(axis=0))
    adata_filtered = adata_filtered[:, mean_bin_copy < 10]

    changepoint_matrix = []

    for _, df in adata_filtered.var.groupby('chr', observed=True):
        chr_adata = adata_filtered[:, df.index]

        # CN transitions / changepoints using diff, sort first
        chr_adata = chr_adata[:, chr_adata.var.sort_values(['chr', 'start']).index]
        is_changepoint = (np.diff(chr_adata.layers['state'], axis=1) != 0)

        # Create a changepoint matrix
        is_changepoint_df = pd.DataFrame(
            is_changepoint,
            index=chr_adata.obs.index,
            columns=chr_adata.var.index[:-1],
        ).T
        changepoint_matrix.append(is_changepoint_df)

    changepoint_matrix = pd.concat(changepoint_matrix)

    return changepoint_matrix
