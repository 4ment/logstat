import click
import numpy as np
import polars as pl

from logstat.options import common_options
from logstat.stats import compute_stats
from logstat.utils import formatter, pad_list, tabulate


@click.command()
@common_options
@click.option(
    '--hpd',
    type=float,
    default=0.95,
    help="""interval of higher posterior density (HPD) [default: %(default)s]""",
)
def run(logfiles, burnin, sep, skip_rows, comment, state, include, exclude, hpd):
    """Summarize posterior sample."""

    all_data = []
    all_variables = []
    burnins = pad_list(burnin, len(logfiles))
    skip_rows = pad_list(skip_rows, len(logfiles), default=0)

    for idx, f in enumerate(logfiles):
        df = pl.read_csv(
            f, separator=sep, comment_prefix=comment, skip_rows=skip_rows[idx]
        )
        excluded = [state]

        if len(exclude) > 0:
            excluded.extend(exclude)
            df = df.drop(excluded)
        elif len(include) > 0:
            df = df.select(include)
        else:
            df = df.drop(excluded)

        if burnin is not None:
            start = int(burnins[idx] * df.shape[0])
            df = df[start:]

        data = df.to_numpy()

        data, stats_header = compute_stats(data, hpd)

        # convert ndarray of floats to ndarray of strings
        ess = list(map(lambda x: [f'{int(x)}'], data[:, 0]))
        stats = [list(map(formatter, data[:, i])) for i in range(1, data.shape[1])]

        all_data.append(np.concatenate((np.array(ess), np.array(stats).T), axis=1))
        all_variables.append(np.array(df.columns))

    if len(all_data) == 1:
        final_data = all_data[0]
        variables = all_variables[0]
    else:
        common_variables = np.array(
            list(set(all_variables[0]).intersection(*list(map(set, all_variables[1:]))))
        )
        stats_count = all_data[0].shape[1]

        if len(common_variables) > 0:
            # preserve the original order of the parameters
            indices = np.array(
                [
                    idx
                    for idx, var in enumerate(all_variables[0])
                    if var in common_variables
                ]
            )
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

    print()
    print(tabulate(final_data, stats_header, variables))
