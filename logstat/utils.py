from io import StringIO

import numpy as np


def formatter(x):
    """Format a floating point number.

    Scientific notation is used if |x| < 0.1 or |x| >= 100000.0

    Args:
        x (float): a floating point number.

    Returns:
        string: string representation of number.

    Examples:
    >>> formatter(10000000)
    '1.000e+07'
    >>> formatter(-0.15)
    '-0.15'
    >>> formatter(10.0)
    '10'
    """
    if abs(x) < 0.1 or abs(x) >= 100000.0:
        return f'{x:,.3e}'
    else:
        return f'{x:,.3f}'.rstrip('0').rstrip('.')


def pad_list(x, size, default=0.0):
    """Expand list.

    Args:
        x (list of float): list
        size (int): expected length of x.

    Returns:
        list of float: x list
    """
    if x is None:
        x = [default]

    return np.pad(
        x,
        (0, size - len(x)),
        constant_values=x[-1],
    )


def tabulate(data, header, variables):
    """Tabulate data."""
    table_header = np.expand_dims(np.concatenate([np.array(['']), header]), axis=0)
    table_body = np.concatenate([np.expand_dims(variables, axis=1), data], axis=1)
    table = np.concatenate([table_header, table_body], axis=0)
    table_max = np.apply_along_axis(lambda x: max(map(len, x)), 0, table)

    string = StringIO()
    string.write(f"{table[0, 0]:{table_max[0]}}")
    for col in range(1, table.shape[1]):
        string.write(f" |{table[0, col]:{'>'}{table_max[col]+1}}")
    string.write('\n')

    separator = np.full(sum(table_max) + 3 * (len(table_max) - 1), '-')
    separator[np.cumsum(table_max + 3)[:-1] - 2] = '+'
    string.write(''.join(separator) + '\n')

    for row in range(1, table.shape[0]):
        string.write(f"{table[row, 0]:{table_max[0]}}")
        for col in range(1, table.shape[1]):
            string.write(f" |{table[row, col]:{'>'}{table_max[col]+1}}")
        string.write('\n')

    return string.getvalue()
