

import importlib.resources

import anndata as ad
import pandas as pd
from anndata import AnnData
from pandas import DataFrame


def OV2295_HMMCopy_reduced() -> AnnData:
    """ DLP data from the OV2295 ovarian cell lines.

    Returns
    -------
    AnnData
        HMMCopy data, reduced size
    """

    adata_filename = importlib.resources.files('scgenome').joinpath('datasets/data/OV2295_HMMCopy_reduced.h5ad')
    return ad.read_h5ad(adata_filename)


def OV_051_Medicc2_reduced() -> AnnData:
    """ DLP data from the OV2295 ovarian cell lines.

    Returns
    -------
    AnnData
        HMMCopy data, reduced size
    """

    adata_filename = importlib.resources.files('scgenome').joinpath('datasets/data/OV_051_Medicc2_reduced.h5ad')
    return ad.read_h5ad(adata_filename)


def OV081_Signals_reduced() -> AnnData:
    """ Allele specific DLP data from the SPECTRUM-OV-081 ovarian tumour.

    Two samples from one patient, processed with signals, so both total and
    allele specific copy number are available. Unlike
    :func:`OV2295_HMMCopy_reduced` this carries the allele layers (``A``,
    ``B``, ``BAF``, ``alleleA``, ``alleleB``, ``totalcounts``) needed for
    allele specific plotting.

    Reduced to 100 cells sampled from the 8 largest clusters, at the native
    500 kb bin resolution. Pairs with :func:`OV081_breakpoints`, which holds
    the somatic rearrangements called on the same aliquot.

    Returns
    -------
    AnnData
        signals allele specific copy number data, reduced size
    """

    adata_filename = importlib.resources.files('scgenome').joinpath('datasets/data/OV081_Signals_reduced.h5ad')
    return ad.read_h5ad(adata_filename)


def OV081_breakpoints() -> DataFrame:
    """ Somatic rearrangement breakpoints for the SPECTRUM-OV-081 ovarian tumour.

    Called on the same aliquot as :func:`OV081_Signals_reduced`, so the
    breakpoints can be drawn against that copy number data. Unfiltered, so
    read support columns are included and callers can apply their own
    thresholds.

    Returns
    -------
    DataFrame
        breakpoints with columns ``chromosome_1``, ``strand_1``,
        ``position_1``, ``chromosome_2``, ``strand_2``, ``position_2``,
        ``rearrangement_type``, ``num_unique_reads``, ``template_length_min``
    """

    breakpoints_filename = importlib.resources.files('scgenome').joinpath('datasets/data/OV081_breakpoints.csv.gz')
    return pd.read_csv(
        breakpoints_filename, sep='\t',
        dtype={'chromosome_1': str, 'chromosome_2': str})
