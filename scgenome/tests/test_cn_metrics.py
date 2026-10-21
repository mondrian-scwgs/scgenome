"""Tests for copy number changepoints and local BAF variance."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scgenome


def cn_adata(state, chrs, starts, gc=None, copy=None, layers=None):
    state = np.array(state, dtype=float)
    var = pd.DataFrame({'chr': chrs, 'start': starts})
    var['end'] = var['start'] + 99
    var['gc'] = 0.4 if gc is None else gc
    var.index = [f'{c}:{s}-{e}' for c, s, e in zip(var['chr'], var['start'], var['end'])]
    obs = pd.DataFrame(index=[f'cell{i}' for i in range(state.shape[0])])
    layers = dict(layers or {})
    layers['state'] = state
    layers['copy'] = state if copy is None else np.array(copy, dtype=float)
    return ad.AnnData(obs=obs, var=var, layers=layers)


def test_changepoints_within_chromosomes_only():
    adata = cn_adata(
        [[2, 2, 3, 3, 1],
         [2, 2, 2, 2, 2]],
        chrs=['1', '1', '1', '2', '2'], starts=[1, 101, 201, 1, 101])

    cp = scgenome.tl.compute_cn_changepoint_matrix(adata)

    # last bin of each chromosome has no next bin
    assert list(cp.index) == ['1:1-100', '1:101-200', '2:1-100']
    assert list(cp.columns) == ['cell0', 'cell1']
    np.testing.assert_array_equal(cp.values, [[False, False], [True, False], [True, False]])


def test_changepoints_sort_by_start_and_skip_filtered_bins():
    # bins out of order, one with no gc and one with high copy, both spanning a change
    adata = cn_adata(
        [[3, 2, 5, 9, 3]],
        chrs=['1'] * 5, starts=[301, 1, 101, 201, 401],
        gc=[0.4, 0.4, -1, 0.4, 0.4], copy=[[3, 2, 5, 20, 3]])

    cp = scgenome.tl.compute_cn_changepoint_matrix(adata)

    # remaining bins in order: 1 (2), 301 (3), 401 (3)
    assert list(cp.index) == ['1:1-100', '1:301-400']
    np.testing.assert_array_equal(cp['cell0'].values, [True, False])


def test_changepoints_requires_layers():
    adata = cn_adata([[2, 2]], chrs=['1', '1'], starts=[1, 101])
    del adata.layers['copy']
    with pytest.raises(ValueError, match='copy'):
        scgenome.tl.compute_cn_changepoint_matrix(adata)


def test_local_baf_variance():
    # pseudobulk BAF across both cells: 0.5, 0.5, 0.5, 0.5, 0.0, missing
    allele_b = np.array([[1, 2, 1, 3, 0, 0], [1, 0, 1, 1, 0, 0]], dtype=float)
    total = np.array([[2, 2, 2, 6, 3, 0], [2, 2, 2, 2, 1, 0]], dtype=float)
    adata = cn_adata(
        np.full((2, 6), 2), chrs=['1'] * 6, starts=[1, 101, 201, 301, 401, 501],
        layers={'alleleB': allele_b, 'totalcounts': total})

    result = scgenome.pp.calculate_local_baf_variance(adata)

    assert result is adata
    lbv = adata.var['local_baf_variance']
    # the first four bins each have a window of identical BAF on one side
    np.testing.assert_allclose(lbv.iloc[:4], 0.)
    # bin 5 is alone in its forward window, so only the backward window, which spans the drop, counts
    np.testing.assert_allclose(lbv.iloc[4], np.var([0.5, 0.5, 0.5, 0.], ddof=1))
    # no allele counts, no BAF
    assert np.isnan(lbv.iloc[5])
