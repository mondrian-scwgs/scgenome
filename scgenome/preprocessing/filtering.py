import logging
import numpy as np
import scipy.stats

from anndata import AnnData
from scgenome._validate import validate_adata



_default_filters = (
    'filter_quality',
    'filter_reads',
    'filter_copy_state_diff',
    'filter_is_s_phase',
)

def calculate_filter_metrics(
        adata: AnnData,
        quality_score_threshold=0.75,
        read_count_threshold=500000,
        copy_state_diff_threshold=1.,
    ) -> AnnData:
    """ Calculate additional filtering metrics to be used by other filtering methods.

    Parameters
    ----------
    adata : AnnData
        copy number data on which to calculate filter metrics
    quality_score_threshold : float, optional
        The minimum quality to set to keep, by default 0.75
    read_count_threshold : int, optional
        The minimum total mapped reads from hmmcopy to set for keeping, by default 500000
    copy_state_diff_threshold : float, optional
        Minimum copy-state difference threshold to set to keep, by default 1.

    Returns
    -------
    AnnData
        input AnnData with filter columns added to obs

    Reads
    -----
    adata.obs['quality'] : quality score per cell (optional)
    adata.obs['total_mapped_reads_hmmcopy'] : total reads per cell (optional)
    adata.obs['is_s_phase'] : S-phase indicator per cell (optional)
    adata.layers['copy'] : continuous copy number values
    adata.layers['state'] : integer copy number states

    Modifies
    --------
    adata.obs['filter_quality'] : True if cell passes quality threshold
    adata.obs['filter_reads'] : True if cell passes read count threshold
    adata.obs['filter_copy_state_diff'] : True if cell passes copy-state diff threshold
    adata.obs['filter_is_s_phase'] : True if cell is not in S-phase (optional)
    adata.obsm['copy_state_diff'] : absolute difference between copy and state
    adata.obsm['copy_state_diff_mean'] : mean copy-state difference per cell
    """
    validate_adata(
        adata,
        require_layers=['copy', 'state'],
        caller='calculate_filter_metrics',
    )

    # Filter Quality and Filter Reads
    if 'quality' in adata.obs.columns:
        adata.obs['filter_quality'] = (adata.obs['quality'] > quality_score_threshold)
    else:
        logging.warning("quality is not in AnnData.obs. Skipping filter_quality")
    
    if 'total_mapped_reads_hmmcopy' in adata.obs.columns:
        adata.obs['filter_reads'] = (adata.obs['total_mapped_reads_hmmcopy'] > read_count_threshold)
    else:
        logging.warning("total_mapped_reads_hmmcopy is not in AnnData.obs. Skipping total_mapped_reads_hmmcopy")
    
    # Copy State Difference Filter
    adata.obsm['copy_state_diff'] = np.absolute(adata.layers['copy'] - adata.layers['state'])
    adata.obsm['copy_state_diff_mean'] = np.nanmean(adata.obsm['copy_state_diff'], axis=1)

    adata.obs['filter_copy_state_diff'] = (adata.obsm['copy_state_diff_mean'] < copy_state_diff_threshold)

    # Filter s phase column
    if 'is_s_phase' in adata.obs.columns:
        adata.obs['filter_is_s_phase'] = ~(adata.obs['is_s_phase'].fillna(False))
    else:
        logging.warning("No is_s_phase in AnnData.obs. Skipping filter_is_s_phase")

    return adata


def filter_cells(
        adata: AnnData,
        filters = _default_filters,
    ) -> AnnData:
    """
    Filter poor quality cells based on the filters provided.

    Parameters
    -------
    adata : AnnData
        AnnData to perform operation with
    filters : list, optional
        Filters to apply. Keeps cells where filters are true, by default _default_filters

    Returns
    -------
    AnnData
        filtered copy number data (subset of input cells)

    Reads
    -----
    adata.obs[filters] : boolean filter columns (from calculate_filter_metrics)

    Modifies
    --------
    Subsets adata to cells passing all filter criteria.
    """
    validate_adata(adata, caller='filter_cells')

    # Ensure cnfilter.calculate_filter_metrics has been called
    for filter_option in filters:
        if filter_option not in adata.obs.columns:
            logging.warning(
                f"WARNING: {filter_option} is not found! "
                "Skipping. Are you sure `scgenome.pp.calculate_filter_metrics` has been called?"
            )
            continue

        adata = adata[adata.obs[filter_option]]

    return adata


