""" Build the bundled OV081 allele-specific example dataset.

The source data are a full SPECTRUM-OV-081 signals result (1236 cells x 6087
bins, ~500 MB) and the matching somatic breakpoint calls from the same
aliquot. Neither is small enough to ship, so this script subsets them down to
something that can live in the repo and be executed on every docs build.

Run from the repository root, with the source files alongside::

    python scripts/make_OV081_signals_reduced.py \
        --adata signals_SPECTRUM-OV-081.h5 \
        --breakpoints SHAH_H000340_T03_03_DLP01_somatic_breakpoints.csv

This is not part of the package or the test suite. It is checked in so the
bundled dataset can be regenerated, and so the choices below (which cells,
which columns, which dtypes) are reviewable rather than lost in a notebook.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

# Cells are sampled from the largest tumour clusters only. Singleton clusters
# and the unassigned (-1) cluster carry no signal at this size and make the
# heatmap look like noise.
N_CELLS = 100
N_CLUSTERS = 8
MIN_CLUSTER_SIZE = 8

# This patient's data contains a substantial normal cell population, plus
# clusters that are near-diploid with essentially no loss of heterozygosity.
# Both are uninteresting for copy number examples and, being the two largest
# clusters, would dominate a size-ordered sample. Mean cluster LOH separates
# them decisively: the normal-like clusters sit near 0.002, every tumour
# cluster above 0.16. Filtering on the phenotype rather than on a hardcoded
# list of cluster ids keeps this honest if the source data is ever recalled.
MIN_CLUSTER_LOH = 0.05

# obs is trimmed from ~100 columns to the ones the documentation actually
# plots or annotates by. The rest are sequencing-run bookkeeping (fastqscreen
# counts, primer sequences, plate positions) that only inflate the file.
OBS_COLUMNS = [
    # provenance
    'brief_cell_id',
    'patient_id',
    'sample_id',
    'library_id',
    # clustering / ordering, used by the heatmap pages
    'cluster_id',
    'cluster_size',
    'cell_order',
    # QC, used for cell selection and annotation bars
    'quality',
    'mad_hmmcopy',
    'coverage_depth',
    'total_mapped_reads',
    'total_reads',
    # copy number summaries
    'mean_copy',
    'ploidy',
    'state_mode',
    'multiplier',
    'breakpoints',
    'n_wgd',
    'is_wgd',
    'fraction_loh',
    'mean_allele_diff',
]

# Breakpoint columns used by plot_rearrangement_arcs plus the two fields the
# docs filter on. The source table has ~90 columns of caller internals.
BREAKPOINT_COLUMNS = [
    'chromosome_1', 'strand_1', 'position_1',
    'chromosome_2', 'strand_2', 'position_2',
    'rearrangement_type', 'num_unique_reads', 'template_length_min',
]

# The breakpoint rows are deliberately NOT filtered by read support. The full
# 11k calls gzip to ~150 KB, and shipping them unfiltered lets the docs show
# the support filtering as part of the example rather than hiding it here.


def select_cells(adata, rng):
    """ Sample cells from the largest tumour clusters, proportional to cluster size. """

    obs = adata.obs

    # Drop cells that would show up as noise rather than as biology. S phase
    # cells have replication-driven copy number artefacts, and doublets and
    # outliers are exactly what a reader would filter out first.
    keep = (
        (obs['cluster_id'].astype(str) != '-1') &
        (obs['is_normal'].astype(str) != 'True') &
        (obs['is_doublet'].astype(str) == 'No') &
        (obs['is_outlier'].astype(str) == '0') &
        (~obs['is_s_phase'].astype(bool)))
    obs = obs[keep]

    # Restrict to clusters that look like tumour, then take the largest.
    cluster_loh = obs.groupby('cluster_id', observed=True)['fraction_loh'].mean()
    tumour_clusters = cluster_loh[cluster_loh > MIN_CLUSTER_LOH].index

    sizes = obs.loc[obs['cluster_id'].isin(tumour_clusters), 'cluster_id'].value_counts()
    sizes = sizes[sizes >= MIN_CLUSTER_SIZE].head(N_CLUSTERS)

    total = sizes.sum()
    selected = []

    for cluster_id, size in sizes.items():
        cluster_cells = obs.index[obs['cluster_id'] == cluster_id]

        # At least 2 per cluster so every cluster survives as a visible band,
        # otherwise proportional to how big the cluster is in the full data.
        n = max(2, int(round(N_CELLS * size / total)))
        n = min(n, len(cluster_cells))

        selected.extend(rng.choice(cluster_cells, size=n, replace=False))

    return list(selected)


def reduce_adata(adata, rng):
    """ Subset to a small cell set and shrink the dtypes. """

    adata = adata[select_cells(adata, rng)].copy()

    adata.obs = adata.obs[OBS_COLUMNS].copy()

    # cell_order arrives as a categorical of stringified floats over all 1236
    # cells. Re-rank it over the surviving cells so it is a contiguous integer
    # order, which is what the plotting code expects.
    cell_order = adata.obs['cell_order'].astype(float)
    adata.obs['cell_order'] = cell_order.rank(method='first').astype(int) - 1

    # cluster_size described the full dataset; recompute against this subset so
    # it agrees with what is actually here.
    adata.obs['cluster_id'] = adata.obs['cluster_id'].cat.remove_unused_categories()
    adata.obs['cluster_size'] = adata.obs.groupby('cluster_id', observed=True)['cluster_id'].transform('size')

    for column in ('sample_id', 'library_id', 'patient_id', 'is_wgd'):
        adata.obs[column] = adata.obs[column].cat.remove_unused_categories()

    # The outlier calls were made on the full dataset and every flagged cell
    # has been dropped, so the recorded parameters no longer describe anything
    # present here.
    del adata.uns['outliers']

    # Reads and integer states are small non-negative integers; the float
    # layers do not carry anywhere near float64 of real precision.
    adata.X = adata.X.astype(np.int32)
    adata.layers['state'] = adata.layers['state'].astype(np.int8)

    for layer in ('copy', 'A', 'B', 'BAF'):
        adata.layers[layer] = adata.layers[layer].astype(np.float32)

    # Allele and total counts are integral in the source data despite being
    # stored as floats. NaN means "no allele data for this bin", so they cannot
    # become an integer dtype without losing that distinction.
    for layer in ('alleleA', 'alleleB', 'totalcounts'):
        adata.layers[layer] = adata.layers[layer].astype(np.float32)

    adata.uns['genome'] = 'hg19'

    return adata


def reduce_breakpoints(path):
    """ Load the somatic breakpoint calls and trim to the columns of interest. """

    breakpoints = pd.read_csv(
        path, sep='\t', low_memory=True,
        dtype={'chromosome_1': str, 'chromosome_2': str})

    return breakpoints[BREAKPOINT_COLUMNS].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adata', required=True, help='full signals h5ad')
    parser.add_argument('--breakpoints', required=True, help='somatic breakpoint calls (tsv)')
    parser.add_argument(
        '--outdir', default='scgenome/datasets/data',
        help='where to write the bundled files')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    rng = np.random.default_rng(args.seed)

    adata = reduce_adata(ad.read_h5ad(args.adata), rng)
    adata_path = outdir / 'OV081_Signals_reduced.h5ad'
    adata.write_h5ad(adata_path, compression='gzip')

    breakpoints = reduce_breakpoints(args.breakpoints)
    breakpoints_path = outdir / 'OV081_breakpoints.csv.gz'
    breakpoints.to_csv(breakpoints_path, sep='\t', index=False, compression='gzip')

    print(adata)
    print(f'\n{adata_path}: {adata_path.stat().st_size / 1e6:.1f} MB')
    print(f'{breakpoints_path}: {breakpoints_path.stat().st_size / 1e6:.1f} MB '
          f'({len(breakpoints)} breakpoints)')


if __name__ == '__main__':
    main()
