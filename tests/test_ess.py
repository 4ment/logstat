import unittest

import numpy as np

from logstat.ess import effective_sample_size, effective_sample_size_one_dim


class TestEffectiveSampleSizeOneDim(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.states = np.arange(1.0, 11.0)

    def test_effective_sample_size_one_dim_positive(self):
        # Test with max_lag 2, which should have a valid effective sample size
        result = effective_sample_size_one_dim(self.states, 2)
        expected = 10
        self.assertEqual(result, expected)

        # Test with max_lag 4
        result = effective_sample_size_one_dim(self.states, 4)
        expected = 2
        self.assertEqual(result, expected)

    def test_effective_sample_size_one_dim_zero_lag(self):
        # Test with max_lag 0, should raise an error
        with self.assertRaises(AssertionError):
            effective_sample_size_one_dim(self.states, 0)

    def test_effective_sample_size_one_dim_negative_lag(self):
        # Test with a negative lag, should raise an error
        with self.assertRaises(AssertionError):
            effective_sample_size_one_dim(self.states, -1)

    def test_effective_sample_size_one_dim_invalid_data(self):
        # Test with invalid data, such as an empty array
        with self.assertRaises(AssertionError):
            effective_sample_size_one_dim(np.array([]), 2)

    def test_effective_sample_size_one_dim_identical_data(self):
        # Test with identical values
        result = effective_sample_size_one_dim(np.zeros(10), 5)
        self.assertEqual(result, 0.0)

    def test_effective_sample_size_2d(self):
        # Test effective_sample_size on 2D data
        states_2d = np.resize(self.states, (2, len(self.states))).T
        states_2d[:, 1] += 10
        result = effective_sample_size(states_2d)
        np.testing.assert_array_equal(result, np.array([3.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
