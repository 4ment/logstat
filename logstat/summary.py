import numpy as np
import pandas as pd

from logstat.stats import compute_stats
from logstat.utils import formatter, pad_list


def run(args):
    """Main function."""

    all_data = []
    all_variables = []

    burnins = pad_list(args.burnin, len(args.logfiles))

    for idx, f in enumerate(args.logfiles):
        df = pd.read_csv(f, sep=args.sep, comment=args.comment)

        excluded = [args.state]

        if args.exclude is not None:
            excluded.extend(args.exclude)
            df.drop(inplace=True, columns=excluded)
        elif args.include is not None:
            df = df.filter(args.include)
        else:
            df.drop(inplace=True, columns=excluded)

        if args.burnin is not None:
            start = int(burnins[idx] * df.shape[0])
            df = df[start:]

        data = df.to_numpy()

        data, stats_header = compute_stats(data, args.hpd)

        # convert ndarray of floats to ndarray of strings
        ess = list(map(lambda x: [f'{int(x)}'], data[:, 0]))
        stats = [list(map(formatter, data[:, i])) for i in range(1, data.shape[1])]

        all_data.append(np.concatenate((np.array(ess), np.array(stats).T), axis=1))
        all_variables.append(df.columns)

    if len(all_data) == 1:
        final_data = all_data[0]
        variables = all_variables[0]
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
        else:
            variables = np.expand_dims(all_variables[0], 1)
            variables = np.concatenate(
                (variables, np.full((len(all_variables), len(all_data) - 1), '')),
                axis=1,
            ).reshape(-1)
            final_data = np.concatenate(all_data, axis=1).reshape(-1, stats_count)

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
                final_data,
                index=variables,
                columns=stats_header,
            )
        )
