"""Configs for the Hugging Face layout-analysis datasets."""

from __future__ import annotations

from atria_core.datasets import HuggingfaceDatasetConfig


class LayoutDatasetConfig(HuggingfaceDatasetConfig):
    """Per-split sample caps for a Hugging Face layout-analysis dataset.

    A cap of `None` keeps every sample in that split.
    """

    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
