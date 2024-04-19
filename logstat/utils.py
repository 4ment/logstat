def formatter(x):
    """Format a floating point number.

    Scientific notation is used if |x| < 0.1 or |x| >= 100000.0

    Args:
        x (float): a floating point number.

    Returns:
        string: string representation of number.
    """
    if abs(x) < 0.1 or abs(x) >= 100000.0:
        return f'{x:,.3e}'
    else:
        return f'{x:,.3f}'
