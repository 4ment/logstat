import argparse

import numpy as np
import pandas as pd

from logstat.ess import effective_sample_size
from logstat.utils import formatter


def create_arg_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='logstat',
        description="""Command line interface for calculating statistics from log files""",
    )
    parser.add_argument('logfiles', metavar='N', nargs='+', help="""log file""")
    parser.add_argument(
        '--hpd',
        type=float,
        default=0.95,
        help="""interval of higher posterior distribution (HPD) [default: %(default)s]""",
    )
    parser.add_argument(
        '--burnin',
        type=float,
        action='append',
        help="""burnin [default: %(default)s]""",
    )
    parser.add_argument(
        '--sep', default='\t', help="""delimiter in log file [default: tab-delimited]"""
    )
    parser.add_argument(
        '--comment',
        default='#',
        help="""comment at the beginning of the log file [default: %(default)s]""",
    )
    parser.add_argument('--include', action='append', help="""include column name""")
    parser.add_argument('--exclude', action='append', help="""exclude column name""")

    return parser


def compute_stats(data, hpd):
    """Compute statistics from MCMC samples

    Args:
        data (ndarray): samples with shape [N,K]
        hpd (float): interval of higher posterior distribution (0 < hpd < 1)

    Returns:
        ndarray: statistics with shape [K,6]
    """
    mean_ = np.mean(data, axis=0, keepdims=True)
    stdev = np.std(data, axis=0, keepdims=True)
    lower_hpd = (1.0 - hpd) / 2
    upper_hpd = 1.0 - lower_hpd
    q = np.quantile(data, np.array([0.5, lower_hpd, upper_hpd]), axis=0)
    ess = effective_sample_size(data)
    ess = np.expand_dims(ess, axis=0)
    header = ['ESS', 'mean', 'median', '0.025', '0.975', 'stdev']
    return np.concatenate((ess, mean_, q, stdev)).T, header


def print_data(data, index, columns):
    """Print tabulated stats."""
    with pd.option_context(
        'display.max_rows',
        None,
        'display.max_columns',
        None,
        'display.width',
        0,
    ):
        print(
            pd.DataFrame(
                data,
                index=index,
                columns=columns,
            )
        )


def main():
    """Main function."""
    parser = create_arg_parser()
    args = parser.parse_args()

    all_data = []
    all_variables = []

    burnins = np.pad(
        args.burnin,
        (0, len(args.logfiles) - len(args.burnin)),
        constant_values=args.burnin[-1],
    )

    for idx, f in enumerate(args.logfiles):
        df = pd.read_csv(f, sep=args.sep, comment=args.comment)

        excluded = ['state']
        if args.exclude is not None:
            excluded.extend(args.exclude)
            df.drop(inplace=True, columns=excluded)
        elif args.include is not None:
            df = df.filter(args.include)
        else:
            df.drop(inplace=True, columns=excluded)

        data = df.to_numpy()

        if args.burnin is not None:
            start = int(burnins[idx] * data.shape[0])
            data = data[start:, :]

        data, stats_header = compute_stats(data, args.hpd)

        # convert ndarray of floats to ndarray of strings
        ess = list(map(lambda x: [f'{int(x)}'], data[:, 0]))
        stats = [list(map(formatter, data[:, i])) for i in range(1, data.shape[1])]

        all_data.append(np.concatenate((np.array(ess), np.array(stats).T), axis=1))
        all_variables.append(df.columns)

    if len(all_data) == 1:
        print_data(all_data[0], all_variables[0], stats_header)
    else:
        common_variables = np.array(
            list(
                set(all_variables[0].to_list()).intersection(
                    *list(map(lambda x: set(x.to_list()), all_variables[1:]))
                )
            )
        )
        stats_count = all_data[0].shape[1]

        if len(common_variables) > 0:
            # preserve the original order of the parameters
            indices = []
            common_variables_list = common_variables.tolist()
            first_variables_list = all_variables[0].to_list()
            for x in first_variables_list:
                if x in common_variables_list:
                    indices.append(first_variables_list.index(x))
            indices = np.array(indices)
            common_variables = all_variables[0][indices]
            common_data = [d[indices, :] for d in all_data]

            variables = np.expand_dims(common_variables, 1)
            variables = np.concatenate(
                (variables, np.full((len(common_variables), len(all_data) - 1), '')),
                axis=1,
            ).reshape(-1)
            final_data = np.concatenate(common_data, axis=1).reshape(-1, stats_count)

            print_data(final_data, variables, stats_header)
        else:
            variables = np.expand_dims(all_variables[0], 1)
            variables = np.concatenate(
                (variables, np.full((len(all_variables), len(all_data) - 1), '')),
                axis=1,
            ).reshape(-1)
            final_data = np.concatenate(all_data, axis=1).reshape(-1, stats_count)
            print_data(final_data, variables, stats_header)


if __name__ == "__main__":
    main()
