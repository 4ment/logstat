import numpy as np

from logstat.utils import pad_list, tabulate


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


def test_tabulate():
    """Test tabulate."""
    data = [
        ["Alice", "25", "Engineer"],
        ["Bob", "30", "Doctor"],
        ["Charlie", "35", "Artist"],
    ]
    header = ["Name", "Age", "Occupation"]
    row_names = ["1", "22", "333"]

    expected_output = (
        "    |    Name | Age | Occupation\n"
        "----+---------+-----+-----------\n"
        "1   |   Alice |  25 |   Engineer\n"
        "22  |     Bob |  30 |     Doctor\n"
        "333 | Charlie |  35 |     Artist"
    )
    result = tabulate(data, header, row_names)

    assert (
        result.strip() == expected_output.strip()
    ), "Tabulate function output is incorrect"
