import argparse
import sys
from typing import Sequence

from logstat import summary


def create_arg_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='logstat',
        description="""Command line interface for analyzing log files.""",
    )
    parser.add_argument('logfiles', metavar='N', nargs='+', help="""log file""")
    parser.add_argument(
        '--hpd',
        type=float,
        default=0.95,
        help="""interval of higher posterior density (HPD) [default: %(default)s]""",
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
    parser.add_argument(
        '--state',
        default='state',
        help="""state number identifier [default: %(default)s]""",
    )
    parser.add_argument('--include', action='append', help="""include column name""")
    parser.add_argument('--exclude', action='append', help="""exclude column name""")

    return parser


def main(argv: Sequence[str] | None = None):
    """Main function."""
    parser = create_arg_parser()
    args = parser.parse_args(argv)

    return summary.run(args)


def console_main():
    """This serves as CLI entry point."""
    sys.exit(main())
