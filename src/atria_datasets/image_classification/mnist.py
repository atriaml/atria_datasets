"""MNIST handwritten digits from the Hugging Face hub."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from atria_core.datasets._hf_dataset import HuggingfaceDataset
from atria_core.types._data_instance._image_instance import ImageInstance
from atria_core.types._generic._annotations import ClassificationAnnotation
from atria_core.types._generic._image import Image


class _MNISTTransform:
    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    def __call__(self, sample: dict[str, Any]) -> ImageInstance:
        return ImageInstance(
            # Deterministic, so a sample keeps its id across runs: a hash of
            # its pixels and label.
            sample_id=hashlib.sha1(
                sample["image"].tobytes() + bytes([sample["label"]])
            ).hexdigest()[:16],
            image=Image(content=sample["image"]),
        ).add_annotation(
            ClassificationAnnotation(
                label_value=sample["label"],
                label_name=self._labels[sample["label"]],
            )
        )


class MNIST(HuggingfaceDataset[ImageInstance]):
    """MNIST: 28x28 grayscale handwritten digits in 10 classes."""

    __module_name__ = "mnist"
    __hf_repo__ = "ylecun/mnist"
    __hf_config_name__ = "mnist"

    def _build_input_transform(self) -> Callable[[Any], ImageInstance]:
        return _MNISTTransform(labels=self.metadata.dataset_labels.classification or [])
