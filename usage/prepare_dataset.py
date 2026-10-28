"""Small end-to-end dataset preparation example."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from atria_core.datasets import DatasetBuilder, FileStorageType
from atria_core.logger import get_logger
from atria_core.visualizers import visualize

import atria_datasets

logger = get_logger(__name__)


def benchmark_dataset(dataset: Any, n: int = 10000) -> None:
    """Iterate over all splits, time each sample load, and log min/max/avg."""
    import time

    for split, split_iterator in dataset.split_iterators.items():
        samples: Any = split_iterator
        count = min(len(samples), n)
        if count == 0:
            continue
        times: list[float] = []
        for i in range(count):
            t0 = time.perf_counter()
            samples[i].load()
            times.append(time.perf_counter() - t0)
        logger.info(
            f"[{split.value}] n={count}"
            f"  min={min(times):.3f}s"
            f"  max={max(times):.3f}s"
            f"  avg={sum(times) / len(times):.3f}s"
        )


def prepare_dataset(
    name: str,
    output_dir: str = "./outputs/",
    enable_caching: bool = False,
    visualize_samples: bool = True,
    benchmark: bool = False,
    **dataset_kwargs: Any,
) -> None:
    """Load and cache a dataset, then inspect the first sample of each split."""
    dataset = DatasetBuilder().load(name, **dataset_kwargs)
    if enable_caching:
        dataset = dataset.cache(FileStorageType.DELTALAKE, store_images_to_files=True)
    dataset = dataset.build()

    logger.info("Cached dataset:\n%s", dataset)

    for split, split_iterator in dataset.split_iterators.items():
        samples: Any = split_iterator
        if not visualize_samples or len(samples) == 0:
            continue
        sample = samples[0].load()
        sample_dir = Path(output_dir) / name / split.value
        sample_dir.mkdir(parents=True, exist_ok=True)
        visualize(sample, output_dir=str(sample_dir))
        logger.info(f"First sample of split `{split}`:\n {sample}")

    if benchmark:
        benchmark_dataset(dataset)


def main() -> None:
    """Parse a dataset name and optional config overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(atria_datasets.datasets.list()))
    parser.add_argument(
        "config",
        nargs="*",
        metavar="KEY=VALUE",
        help="Dataset config overrides, with values parsed as JSON when possible.",
    )
    args = parser.parse_args()
    dataset_kwargs: dict[str, Any] = {}
    for item in args.config:
        key, separator, value = item.partition("=")
        if not separator or not key:
            parser.error(f"expected KEY=VALUE, got {item!r}")
        if value.lower() in {"true", "false"}:
            dataset_kwargs[key] = value.lower() == "true"
        elif re.fullmatch(r"-?\d+", value):
            dataset_kwargs[key] = int(value)
        else:
            dataset_kwargs[key] = value
    prepare_dataset(args.name, **dataset_kwargs)


if __name__ == "__main__":
    main()
