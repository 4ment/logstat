import sys

import click

from logstat import summary
from logstat._version import __version__


@click.group()
@click.version_option(__version__)
@click.pass_context
def cli(ctx):
    """Command line interface for analyzing log files"""
    pass


cli.add_command(summary.run, name="summarize")


def console_main():
    """This serves as CLI entry point."""
    sys.exit(cli())
