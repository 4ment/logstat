import click


def common_options(fn):
    fn = click.argument(
        'logfiles',
        metavar='LOGFILE',
        nargs=-1,
        type=click.Path(exists=True),
    )(fn)

    fn = click.option(
        '--burnin',
        type=float,
        multiple=True,
        show_default=True,
        help="""burnin""",
    )((fn))
    fn = click.option(
        '--sep', default='\t', help="""delimiter in log file [default: tab-delimited]"""
    )(fn)
    fn = click.option(
        '--skip_rows',
        type=int,
        multiple=True,
        show_default=True,
        help="""start reading log file after skip_rows lines [default: 0]""",
    )(fn)
    fn = click.option(
        '--comment',
        default='#',
        show_default=True,
        help="""comment at the beginning of the log file""",
    )(fn)
    fn = click.option(
        '--state',
        default='state',
        show_default=True,
        help="""state number identifier""",
    )(fn)
    fn = click.option('--include', multiple=True, help="""include column name""")(fn)
    fn = click.option('--exclude', multiple=True, help="""exclude column name""")(fn)
    return fn
