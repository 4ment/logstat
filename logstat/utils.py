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
