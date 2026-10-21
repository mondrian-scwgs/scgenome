import unittest
import numpy as np
import pandas as pd
import pytest
from scgenome.tools.ranges import intersect_regions, intersect_positions, assign_position_bins, NoIntersectionError


class TestRanges(unittest.TestCase):

    def test_intersect_regions(self):
        # Create sample dataframes
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'chr': ['chr1', 'chr1', 'chr2'],
            'start': [100, 200, 300],
            'end': [150, 250, 350]
        })

        df2 = pd.DataFrame({
            'id': [4, 5, 6],
            'chr': ['chr1', 'chr1', 'chr2'],
            'start': [120, 220, 320],
            'end': [170, 270, 370]
        })

        # Expected result
        expected_result = pd.DataFrame({
            'chr': ['chr1', 'chr1', 'chr2'],
            'start': [120, 220, 320],
            'end': [150, 250, 350],
            'id_x': [1, 2, 3],
            'id_y': [4, 5, 6]
        })

        # Test intersect_regions function
        result = intersect_regions(df1, df2)

        # Dont test categories or column order
        result['chr'] = result['chr'].astype(str)
        result = result[expected_result.columns]

        pd.testing.assert_frame_equal(result, expected_result)


    def test_intersect_regions_no_intersection(self):
        # Create sample dataframes with no intersection
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'chr': ['chr1', 'chr1', 'chr2'],
            'start': [100, 200, 300],
            'end': [150, 250, 350]
        })

        df2 = pd.DataFrame({
            'id': [4, 5, 6],
            'chr': ['chr3', 'chr3', 'chr4'],
            'start': [120, 220, 320],
            'end': [170, 270, 370]
        })

        # Test intersect_regions function raises NoIntersectionError
        with self.assertRaises(NoIntersectionError):
            intersect_regions(df1, df2)


    def test_intersect_positions(self):
        # Create sample dataframes
        regions = pd.DataFrame({
            'id': [1, 2, 3],
            'chr': ['chr1', 'chr1', 'chr2'],
            'start': [100, 200, 300],
            'end': [150, 250, 350]
        })

        positions = pd.DataFrame({
            'id': [4, 5, 6, 7],
            'chr': ['chr1', 'chr1', 'chr2', 'chr2'],
            'position': [120, 180, 320, 360]
        })

        # Expected result
        expected_result = pd.DataFrame({
            'chr': ['chr1', 'chr2'],
            'position': [120, 320],
            'id_x': [1, 3],
            'id_y': [4, 6]
        })

        # Test intersect_positions function
        result = intersect_positions(regions, positions)

       # Dont test categories or column order
        result['chr'] = result['chr'].astype(str)
        result = result[expected_result.columns]

        pd.testing.assert_frame_equal(result, expected_result)


    def test_intersect_positions_no_intersection(self):
        # Create sample dataframes with no intersection
        regions = pd.DataFrame({
            'id': [1, 2, 3],
            'chr': ['chr1', 'chr1', 'chr2'],
            'start': [100, 200, 300],
            'end': [150, 250, 350]
        })

        positions = pd.DataFrame({
            'id': [4, 5, 6, 7],
            'chr': ['chr3', 'chr3', 'chr4', 'chr4'],
            'position': [120, 180, 320, 360]
        })

        # Test intersect_positions function raises NoIntersectionError
        with self.assertRaises(NoIntersectionError):
            intersect_positions(regions, positions)



if __name__ == '__main__':
    unittest.main()

@pytest.fixture
def bins():
    return pd.DataFrame({
        'chr': ['1', '1', '1', '2'],
        'start': [1, 101, 201, 1],
        'end': [100, 200, 300, 100],
    }, index=pd.Index(['1:1-100', '1:101-200', '1:201-300', '2:1-100'], name='bin'))


def positions(chromosomes, coords, **columns):
    return pd.DataFrame({'chromosome': chromosomes, 'position': coords, **columns},
                        index=[f'snv{i}' for i in range(len(coords))])


def test_assign_position_bins_includes_both_bin_ends(bins):
    snvs = positions(['1', '1', '1', '1', '2'], [1, 100, 101, 300, 50])

    assigned = assign_position_bins(snvs, bins)

    assert list(assigned) == ['1:1-100', '1:1-100', '1:101-200', '1:201-300', '2:1-100']
    assert assigned.index.equals(snvs.index)
    assert assigned.name == 'bin'


def test_assign_position_bins_ignores_bin_order_and_categoricals(bins):
    shuffled = bins.iloc[::-1].assign(chr=lambda df: pd.Categorical(df['chr']))

    assigned = assign_position_bins(positions(pd.Categorical(['2', '1']), [10, 150]), shuffled)

    assert list(assigned) == ['2:1-100', '1:101-200']


def test_assign_position_bins_missing_outside_bins(bins):
    # past the last bin, unknown chromosome, before the first bin
    assigned = assign_position_bins(positions(['1', '1', '3', '1'], [50, 301, 10, 0]), bins)

    assert assigned.iloc[0] == '1:1-100'
    assert assigned.iloc[1:].isnull().all()


def test_assign_position_bins_other_column_names(bins):
    snps = pd.DataFrame({'chr': ['1', '2'], 'coord': [150, 1]})

    assigned = assign_position_bins(snps, bins, chr_col='chr', position_col='coord')

    assert list(assigned) == ['1:101-200', '2:1-100']


def test_assign_position_bins_requires_columns(bins):
    with pytest.raises(ValueError, match="positions missing columns \\['chromosome'\\]"):
        assign_position_bins(pd.DataFrame({'chr': ['1'], 'position': [1]}), bins)
