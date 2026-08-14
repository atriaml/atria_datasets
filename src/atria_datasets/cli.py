import click
from atria_core.logger import get_logger

import atria_datasets

logger = get_logger(__name__)


@click.group()
def cli() -> None:
    """Atria Datasets command line interface."""


@cli.command("list")
def list_datasets() -> None:
    """Print the names registered with the dataset registry."""
    logger.info(
        "Available datasets:\n%s", "\n".join(sorted(atria_datasets.datasets.list()))
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
