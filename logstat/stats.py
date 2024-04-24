import sys

import numpy as np

from logstat.ess import effective_sample_size


def higher_posterior_density_one_dim(proportion, x_sorted):
    """Higher posterior density (HPD) interval of N samples.

    Args:
        proportion (float): interval (0 < proportion < 1).
        x_sorted (array_like): array containing samples.

    Returns:
        ndarray: HPD interval with shape [2].
    """
    assert 0.0 < proportion < 1.0

    min_range = sys.float_info.max
    hpd_index = 0

    diff = int(round(proportion * len(x_sorted)))
    for i in range(len(x_sorted) - diff + 1):
        range_ = abs(x_sorted[i + diff - 1] - x_sorted[i])
        if range_ < min_range:
            min_range = range_
            hpd_index = i

    return np.array([x_sorted[hpd_index], x_sorted[hpd_index + diff - 1]])


def higher_posterior_density(proportion, samples, is_sorted=True):
    """Higher posterior density (HPD) intervals of K variables with N samples.

    Args:
        proportion (float): interval (0 < proportion < 1).
        x_sorted (array_like): array containing samples of shape [K,N].
        sorted (bool, optional): If True samples are already sorted. Defaults to True.

    Returns:
        ndarray: HPD intervals with shape [2,K].
    """

    def fn(x):
        return higher_posterior_density_one_dim(proportion, x)

    if not is_sorted:
        samples = np.sort(samples, axis=0)

    return np.apply_along_axis(fn, 0, samples)


def compute_stats(data, proportion):
    """Compute statistics from MCMC samples

    Statistics include:
      - Effecive sample size (ESS)
      - Mean and median
      - Higher posterior density interval
      - Standard deviation

    Args:
        data (ndarray): samples with shape [N,K]
        proportion (float): interval of higher posterior density (0 < hpd < 1)

    Returns:
        ndarray: statistics with shape [K,6]
        list: statistic names
    """
    mean_ = np.mean(data, axis=0, keepdims=True)
    stdev = np.std(data, axis=0, keepdims=True)
    sorted_data = np.sort(data, axis=0)
    # avoid sorting data again in quantile
    # median = np.quantile(data, np.array([0.5]), axis=0)
    if data.shape[0] % 2 == 0:
        median = (
            sorted_data[int(data.shape[0] / 2) - 1, :]
            + sorted_data[int(data.shape[0] / 2), :]
        ) / 2
    else:
        median = sorted_data[int(data.shape[0] / 2), :]
    median = np.expand_dims(median, axis=0)
    hpd = higher_posterior_density(proportion, sorted_data, True)
    ess = effective_sample_size(data)
    ess = np.expand_dims(ess, axis=0)
    lower_hpd = (1.0 - proportion) / 2 * 100
    upper_hpd = 100.0 - lower_hpd
    header = [
        'ESS',
        'mean',
        'median',
        f"{lower_hpd:.2f}%".rstrip('0').rstrip('.'),
        f"{upper_hpd:.2f}%".rstrip('0').rstrip('.'),
        'stdev',
    ]
    return np.concatenate((ess, mean_, median, hpd, stdev)).T, header
