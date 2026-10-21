import numpy as np
import pandas as pd

from anndata import AnnData
from pandas import DataFrame
from scgenome._validate import validate_adata


def add_cn_bin(adata: AnnData, bins: DataFrame) -> AnnData:
    """ Assign each SNV to the copy number bin containing it

    Bins and SNV positions are both 1-based, and a bin contains positions
    `start` to `end` inclusive, as in DLP HMMCopy bins.

    Parameters
    ----------
    adata : anndata.AnnData
        SNV data, as from `scgenome.pp.read_snv_genotyping`, with var columns
        chromosome and position
    bins : pandas.DataFrame
        copy number bins with columns chr, start, end, indexed by bin ID, for
        instance the `var` of a copy number AnnData

    Returns
    -------
    anndata.AnnData
        adata with var['cn_bin'] added

    Raises
    ------
    ValueError
        if any SNV falls outside every bin

    Reads
    -----
    adata.var['chromosome'] : chromosome of each SNV
    adata.var['position'] : position of each SNV

    Modifies
    --------
    adata.var['cn_bin'] : ID of the bin containing each SNV
    """
    validate_adata(adata, require_var=['chromosome', 'position'], caller='add_cn_bin')
    missing = [c for c in ['chr', 'start', 'end'] if c not in bins.columns]
    if missing:
        raise ValueError(f"add_cn_bin: bins missing columns {missing}, has {list(bins.columns)}")

    snv_chromosome = adata.var['chromosome'].astype(str).values
    snv_position = adata.var['position'].values
    bin_chromosome = bins['chr'].astype(str).values

    cn_bin = np.full(adata.shape[1], None, dtype=object)

    for chromosome in np.unique(snv_chromosome):
        snv_idx = np.flatnonzero(snv_chromosome == chromosome)
        chr_bins = bins[bin_chromosome == chromosome].sort_values('start')
        if len(chr_bins) == 0:
            continue

        # Last bin starting at or before each position, then check it reaches the position
        bin_idx = np.searchsorted(chr_bins['start'].values, snv_position[snv_idx], side='right') - 1
        in_bin = (bin_idx >= 0)
        in_bin[in_bin] = snv_position[snv_idx[in_bin]] <= chr_bins['end'].values[bin_idx[in_bin]]

        cn_bin[snv_idx[in_bin]] = chr_bins.index.values[bin_idx[in_bin]]

    is_missing = pd.isnull(cn_bin)
    if is_missing.any():
        raise ValueError(
            f"add_cn_bin: {is_missing.sum()} SNVs are not in any bin, "
            f"for instance {list(adata.var.index[is_missing][:5])}")

    adata.var['cn_bin'] = cn_bin

    return adata

