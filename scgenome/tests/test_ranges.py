import unittest
import pandas as pd
from scgenome.tools.ranges import intersect_regions, intersect_positions, NoIntersectionError


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