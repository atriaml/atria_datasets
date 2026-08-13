from pathlib import Path

import click
from atria_core.logger import get_logger
from atria_core.registry import RegistryStore

from atria_datasets.registry import datasets

logger = get_logger(__name__)

_REGISTRY_PATH = Path(__file__).parent / "registry" / "registry.json"


@click.group()
def cli() -> None:
    """Atria Datasets command line interface."""


@cli.group()
def registry() -> None:
    """Inspect and build the dataset registry."""


@registry.command("print")
def registry_print() -> None:
    """Print the currently loaded dataset registry."""
    logger.info(f"Dataset registry loaded successfully:\n{datasets.list()}")


@registry.command("build")
def registry_build() -> None:
    """Dump the dataset registry to registry.json."""
    # importing atria_datasets (this module's parent package) already ran
    # every @dataset_configs.register(...) -- just dump what's in the registry now.
    RegistryStore.dump(path=_REGISTRY_PATH, data=datasets.to_dict())
    logger.info(f"Wrote {len(datasets.list())} dataset(s) to {_REGISTRY_PATH}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
