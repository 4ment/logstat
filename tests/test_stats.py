import unittest

import numpy as np

from logstat.stats import (
    compute_stats,
    higher_posterior_density,
    higher_posterior_density_one_dim,
)


class TestHigherPosteriorDensity(unittest.TestCase):
    def setUp(self):
        # Set up some sample data for testing
        self.samples_1d = np.arange(1.0, 11.0)
        self.samples_2d = np.resize(self.samples_1d, (2, 10)).T

    def test_hpd_one_dim(self):
        # Test HPD with 50% proportion
        result = higher_posterior_density_one_dim(0.5, self.samples_1d)
        expected = np.array([1.0, 5.0])
        np.testing.assert_array_equal(result, expected)

        # Test HPD with 90% proportion
        result = higher_posterior_density_one_dim(0.9, self.samples_1d)
        expected = np.array([1.0, 9.0])
        np.testing.assert_array_equal(result, expected)

        # Test invalid proportion (should raise AssertionError)
        with self.assertRaises(AssertionError):
            higher_posterior_density_one_dim(1.1, self.samples_1d)

    def test_hpd_2d(self):
        # Test HPD with 50% proportion on 2D data
        result = higher_posterior_density(0.5, self.samples_2d)
        expected = np.array([[1.0, 1.0], [5.0, 5.0]])
        np.testing.assert_array_equal(result, expected)

        # Test with sorted=False (checks if function sorts before applying HPD)
        unsorted_samples_2d = np.array(
            [
                [5, 7, 3, 9, 10, 6, 2, 1, 4, 8],
                [100, 80, 60, 40, 20, 10, 30, 50, 70, 90],
            ],
            dtype=np.double,
        ).T
        expected = np.array([[1.0, 10.0], [5.0, 50.0]])
        result = higher_posterior_density(0.5, unsorted_samples_2d, is_sorted=False)
        np.testing.assert_array_equal(result, expected)

        # Test invalid proportion (should raise AssertionError)
        with self.assertRaises(AssertionError):
            higher_posterior_density(0.0, self.samples_2d)

    def test_compute_stats(self):
        # Test with sample 2D data and a proportion of 0.5
        result, header = compute_stats(self.samples_2d, 0.5)
        self.assertEqual(header, ['ESS', 'mean', 'median', '25.00%', '75.00%', 'stdev'])

        # Check if output dimensions are correct
        self.assertEqual(result.shape, (2, 6))  # 2 samples, 6 stats

        # Check median even
        np.testing.assert_array_equal(result[:, 2], np.median(self.samples_2d, axis=0))

        # Check median odd
        samples = self.samples_2d.copy()
        samples.resize((11, 2))
        result, _ = compute_stats(samples, 0.5)
        np.testing.assert_array_equal(result[:, 2], np.median(samples, axis=0))


if __name__ == "__main__":
    unittest.main()
