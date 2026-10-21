"""Tests for long form to sparse matrix conversion."""

import numpy as np
import pandas as pd

from scgenome.utils import create_sparse_matrices


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
