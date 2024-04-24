import numpy as np


def effective_sample_size_one_dim(states, max_lag):
    """Effective sample size of N samples for one variable.

    Args:
        states (ndarray): samples with shape [N].
        max_lag (int): maximum lag (max_lag > 0).

    Returns:
        float: Effective sample size.
    """
    assert max_lag > 0
    assert len(states) > max_lag

    samples = len(states)
    gamma_stat = np.zeros(max_lag)
    var_stat = 0.0

    for lag in range(max_lag):
        gamma_stat[lag] = np.mean(states[: samples - lag] * states[lag:])

        if lag == 0:
            var_stat = gamma_stat[0]
        elif lag % 2 == 0:
            s = gamma_stat[lag - 1] + gamma_stat[lag]
            if s > 0:
                var_stat += 2.0 * s
            else:
                break

    if var_stat > 0:
        return round(samples * gamma_stat[0] / var_stat)
    else:
        return 0.0


def effective_sample_size(states):
    """Effective sample size of K variables with N samples.

    Args:
        states (ndarray): samples with shape [N,K].

    Returns:
        ndarray: effective sample size.
    """
    max_lag = min(states.shape[0] - 1, 2000)

    states = states - np.mean(states, axis=0)

    def fn(states):
        return effective_sample_size_one_dim(states, max_lag)

    return np.apply_along_axis(fn, 0, states)
