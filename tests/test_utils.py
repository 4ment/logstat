import numpy as np

from logstat.utils import pad_list


def test_pad_list_with_missing_values():
    """Test pad_list when x has fewer elements than expected."""

    class Args:
        burnin = [0.1, 0.2]
        logfiles = ['file1.csv', 'file2.csv', 'file3.csv']

    args = Args()

    result = pad_list(args.burnin, len(args.logfiles))
    expected = [0.1, 0.2, 0.2]  # Padding with the last x value

    assert np.array_equal(result, expected), "Expected x with padding."


def test_pad_list_empty_x():
    """Test pad_list when x is None."""

    class Args:
        burnin = None
        logfiles = ['file1.csv', 'file2.csv']

    args = Args()

    result = pad_list(args.burnin, len(args.logfiles))
    expected = [0.0, 0.0]  # Default padding with zeroes

    assert np.array_equal(result, expected), "Expected zero padding for x."
