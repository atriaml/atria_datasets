"""CLI entry point for preparing any registered dataset: builds it, caches
every split to disk (msgpack shards), and visualizes the first sample of
each split. Every dataset goes through the exact same pipeline regardless
of which one you pick -- only the registry name and config kwargs change.

Usage:
    python usage/prepare_dataset.py gnhk
    python usage/prepare_dataset.py scadsai_german_handwriting --level=word
    python usage/prepare_dataset.py fhswf_german_handwriting --output_dir=./out --no-visualize_samples

Any kwarg not consumed by `prepare_dataset` itself (output_dir,
visualize_samples) is forwarded to the dataset's config, e.g. `--level=word`
above is forwarded to `ScaDSAIConfig(level="word")`.
"""

from __future__ import annotations

from typing import Any

import fire
from atria_core.datasets import Cacher, FileStorageType
from atria_core.logger import get_logger
from atria_core.visualizers import visualize

import atria_datasets  # noqa: F401  (populates the registry on import)
from atria_datasets.registry import datasets

logger = get_logger(__name__)


def prepare_dataset(
    name: str,
    output_dir: str = "./test",
    visualize_samples: bool = True,
    **config_kwargs: Any,
) -> None:
    """Build, cache, and inspect a registered dataset.

    Args:
        name: Registry key of the dataset, e.g. "gnhk".
        output_dir: Directory to write sample visualizations under.
        visualize_samples: If set, visualize the first sample of each split.
        **config_kwargs: Extra kwargs forwarded to the dataset's config,
            e.g. --level=word for scadsai_german_handwriting.
    """
    logger.info(f"Loading dataset {name}...")
    available = sorted(datasets.list())
    if name not in available:
        raise SystemExit(
            f"Unknown dataset {name!r}. Available datasets: {', '.join(available)}"
        )

    config = datasets.get(name)(**config_kwargs)
    dataset = config.build_module()

    cached = Cacher(FileStorageType.DELTALAKE, store_artifacts=False).cache(dataset)
    print("cached", cached)

    for split, split_iterator in cached.split_iterators.items():
        print(f"{name}[{split.value}]: {len(split_iterator)} samples")

        sample = split_iterator[0].load()
        print(sample)

        if visualize_samples:
            visualize(sample, output_dir=f"{output_dir}/{name}/{split.value}")


if __name__ == "__main__":
    fire.Fire(prepare_dataset)
