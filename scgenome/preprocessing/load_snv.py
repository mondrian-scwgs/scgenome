import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

from anndata import AnnData

from scgenome._deprecate import renamed_arguments


@renamed_arguments(filename='genotyping_filename')
def read_snv_genotyping(genotyping_filename: str) -> AnnData:
    """ Read SNV genotyping into an AnnData

    Parameters
    ----------
    genotyping_filename : str
        SNV genotyping filename

    Returns
    -------
    AnnData
        SNV counts in layers['alt_count'] and layers['ref_count'], with X unset
        Cells and variants retain their first-occurrence order. Missing pairs
        have zero counts, and duplicate cell-variant rows are summed.
    """

    data = pd.read_csv(genotyping_filename, dtype={
        'chromosome': 'category',
        'ref': 'category',
        'alt': 'category',
        'cell_id': 'category'})

    obs = data[['cell_id']].drop_duplicates().set_index('cell_id')
    variant_columns = ['chromosome', 'position', 'ref', 'alt']
    var = data[variant_columns].drop_duplicates()
    variant_index = pd.MultiIndex.from_frame(var)
    cell_coordinates = obs.index.get_indexer(data['cell_id'])
    variant_coordinates = variant_index.get_indexer(
        pd.MultiIndex.from_frame(data[variant_columns]))

    layers = {
        count_column: csr_matrix(
            (pd.to_numeric(data[count_column]).to_numpy(), (cell_coordinates, variant_coordinates)),
            shape=(len(obs), len(var)))
        for count_column in ['alt_count', 'ref_count']
    }

    obs.index = obs.index.astype(str)
    var['snv_id'] = (
        var['chromosome'].astype(str) + '-' +
        var['position'].astype(str) + ':' +
        var['ref'].astype(str) + '>' +
        var['alt'].astype(str))
    var = var.set_index('snv_id')

    adata = ad.AnnData(
        X=None,
        obs=obs,
        var=var,
        layers=layers,
    )

    return adata
