"""GNHK (GoodNotes Handwriting Kollection) dataset."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from atria_core.datasets import Cacher, Dataset, DatasetConfig, FileStorageType
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    OCRLevel,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._annotations import OCRAnnotation
from atria_core.types._generic._image import Image
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets

_DATA_URLS = [
    UrlSpec(
        url="https://www.kaggle.com/api/v1/datasets/download/evandu/gnhk-dataset",
        url_ext=".zip",
    )
]

_DESCRIPTION = (
    "GNHK (GoodNotes Handwriting Kollection) dataset containing English "
    "handwritten text images with word-level polygon annotations."
)

_HOMEPAGE = "https://www.kaggle.com/datasets/evandu/gnhk-dataset"
_LICENSE = "CC BY 4.0"


def _parse_manifest(manifest_path: Path) -> dict[str, OCRAnnotation]:
    annotations = {}

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            texts = []
            bboxes = []

            for item in data["annotations"]["texts"]:
                texts.append(item["text"])

                polygon = item["polygon"]

                xs = [point["x"] for point in polygon]
                ys = [point["y"] for point in polygon]

                bboxes.append([min(xs), min(ys), max(xs), max(ys)])

            annotations[data["source-ref"]] = OCRAnnotation(
                level=OCRLevel.word,
                texts=np.asarray(texts, dtype=object),
                bboxes=np.asarray(bboxes, dtype=np.float64),
            )

    return annotations


@datasets.register("gnhk")
@pydantic_dataclass(frozen=True)
class GNHKConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> GNHK:
        return GNHK(self, **kwargs)


class SplitIterator(Sequence[tuple[Path, OCRAnnotation]]):
    def __init__(self, data_dir: str, split: DatasetSplitType):
        root = Path(data_dir)

        split_dir = root / "gnhk-dataset" / "gnhk" / f"{split.value}_data" / split.value

        manifest_path = split_dir / f"{split.value}.manifest"

        annotations = _parse_manifest(manifest_path)

        self.samples = []

        for image_name, annotation in annotations.items():
            image_path = split_dir / image_name

            if image_path.exists():
                self.samples.append((image_path, annotation))

    def __getitem__(self, index: int) -> tuple[Path, OCRAnnotation]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class InputTransform:
    def __call__(
        self, sample: tuple[Path, OCRAnnotation]
    ) -> SinglePageDocumentInstance:
        image_path, annotation = sample

        return SinglePageDocumentInstance(
            sample_id=str(uuid.uuid4()), visual=Image(file_path=str(image_path))
        ).add_annotation(annotation)


class GNHK(Dataset[GNHKConfig, SinglePageDocumentInstance]):
    def _download_urls(self) -> list[str]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION, homepage=_HOMEPAGE, license=_LICENSE
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(data_dir=data_dir, split=split)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()

