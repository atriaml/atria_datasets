"""Small end-to-end dataset preparation example."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from atria_core.datasets import Cacher, FileStorageType
from atria_core.logger import get_logger
from atria_core.visualizers import visualize

import atria_datasets

logger = get_logger(__name__)


def prepare_dataset(
    name: str,
    output_dir: str = "./test",
    visualize_samples: bool = True,
    **dataset_kwargs: Any,
) -> None:
    """Load and cache a dataset, then inspect the first sample of each split."""
    logger.info("Loading dataset %s...", name)
    dataset = atria_datasets.load_dataset(name, **dataset_kwargs)
    # TODO: remove the ignore after atria-core's Cacher annotation is updated
    # from Dataset[Any, T_Sample] to Dataset[T_Sample, Any].
    cached = Cacher(FileStorageType.DELTALAKE).cache(dataset)  # type: ignore[type-var]
    logger.info("Cached dataset:\n%s", cached)

    for split, split_iterator in cached.split_iterators.items():
        samples: Any = split_iterator
        sample_count = len(samples)
        print(f"{name}[{split.value}]: {sample_count} samples")
        if not visualize_samples or sample_count == 0:
            continue

        sample = samples[0].load()
        sample_dir = Path(output_dir) / name / split.value
        sample_dir.mkdir(parents=True, exist_ok=True)
        visualize(sample, output_dir=str(sample_dir))


def main() -> None:
    """Parse a dataset factory name and run its preparation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(atria_datasets.__all__))
    args = parser.parse_args()
    prepare_dataset(args.name)


if __name__ == "__main__":
    main()
