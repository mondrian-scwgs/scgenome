"""Tests for assigning SNVs to bins and long form to sparse matrix conversion."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scgenome
from scgenome.utils import create_sparse_matrices


@pytest.fixture
def bins():
    return pd.DataFrame({
        'chr': ['1', '1', '1', '2'],
        'start': [1, 101, 201, 1],
        'end': [100, 200, 300, 100],
    }, index=pd.Index(['1:1-100', '1:101-200', '1:201-300', '2:1-100'], name='bin'))


def snv_adata(chromosomes, positions):
    var = pd.DataFrame({'chromosome': chromosomes, 'position': positions})
    var.index = [f'snv{i}' for i in range(len(var))]
    return ad.AnnData(obs=pd.DataFrame(index=['cell1']), var=var)


def test_add_cn_bin_includes_both_bin_ends(bins):
    adata = snv_adata(['1', '1', '1', '1', '2'], [1, 100, 101, 300, 50])

    scgenome.tl.add_cn_bin(adata, bins)

    assert list(adata.var['cn_bin']) == ['1:1-100', '1:1-100', '1:101-200', '1:201-300', '2:1-100']


def test_add_cn_bin_ignores_bin_order_and_categoricals(bins):
    adata = snv_adata(pd.Categorical(['2', '1']), [10, 150])
    shuffled = bins.iloc[::-1].assign(chr=lambda df: pd.Categorical(df['chr']))

    scgenome.tl.add_cn_bin(adata, shuffled)

    assert list(adata.var['cn_bin']) == ['2:1-100', '1:101-200']


@pytest.mark.parametrize('chromosome,position', [('1', 301), ('3', 10), ('1', 0)])
def test_add_cn_bin_raises_outside_bins(bins, chromosome, position):
    adata = snv_adata(['1', chromosome], [50, position])

    with pytest.raises(ValueError, match='1 SNVs are not in any bin'):
        scgenome.tl.add_cn_bin(adata, bins)


def test_create_sparse_matrices():
    data = pd.DataFrame({
        'cell_id': ['b', 'a', 'b', 'c'],
        'snv_id': ['y', 'x', 'x', 'y'],
        'alt_count': [1, 0, 2, 3],
        'ref_count': [4, 5, 0, 6],
    })
    original = data.copy()

    matrices, index, columns = create_sparse_matrices(data, 'cell_id', 'snv_id', ['alt_count', 'ref_count'])

    assert list(index) == ['a', 'b', 'c'] and index.name == 'cell_id'
    assert list(columns) == ['x', 'y'] and columns.name == 'snv_id'
    np.testing.assert_array_equal(matrices['alt_count'].toarray(), [[0, 0], [2, 1], [0, 3]])
    np.testing.assert_array_equal(matrices['ref_count'].toarray(), [[5, 0], [0, 4], [0, 6]])
    assert matrices['alt_count'].nnz == 3 and matrices['ref_count'].nnz == 3
    pd.testing.assert_frame_equal(data, original)
