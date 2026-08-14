import click
from atria_core.logger import get_logger

import atria_datasets

logger = get_logger(__name__)


@click.group()
def cli() -> None:
    """Atria Datasets command line interface."""


@cli.command("list")
def list_datasets() -> None:
    """Print the available lazily imported dataset factories."""
    logger.info("Available datasets:\n%s", "\n".join(atria_datasets.__all__))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
