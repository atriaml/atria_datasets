"""ScaDS.AI German Line- and Word-Level Handwriting Dataset."""

from __future__ import annotations

import csv
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
    SinglePageDocumentInstance,
)
from atria_core.types._generic._annotations import OCRAnnotation, OCRLevel
from atria_core.types._generic._image import Image
from atria_core.visualizers import visualize
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets

_DATA_URLS = [
    UrlSpec(
        url="https://zenodo.org/records/18301532/files/scadsai_german_handwriting_line_word_level_v01.tar.gz?download=1",
        url_ext=".tar.gz",
    )
]

_DESCRIPTION = (
    "ScaDS.AI German Line- and Word-Level Handwriting Dataset "
    "containing handwritten line and word images with transcription."
)

_HOMEPAGE = "https://zenodo.org/records/18301532"
_LICENSE = "CC BY 4.0"


def _parse_annotations(csv_path: Path, level: OCRLevel) -> dict[str, OCRAnnotation]:
    annotations = {}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_key = row["line_file"] if level == OCRLevel.line else row["word_file"]

            bbox = None

            if all(
                key in row and row[key] not in ("", None)
                for key in ["x", "y", "width", "height"]
            ):
                x = float(row["x"])
                y = float(row["y"])
                w = float(row["width"])
                h = float(row["height"])

                bbox = np.asarray([[x, y, x + w, y + h]], dtype=np.float64)

            annotations[image_key] = OCRAnnotation(
                level=level, texts=np.asarray([row["text"]], dtype=object), bboxes=bbox
            )

    return annotations


@datasets.register("scadsai_german_handwriting")
@pydantic_dataclass(frozen=True)
class ScaDSAIConfig(DatasetConfig):
    level: OCRLevel = OCRLevel.line

    def build_module(self, **kwargs: Any) -> ScaDSAI:
        return ScaDSAI(self, **kwargs)


class SplitIterator(Sequence[tuple[Path, OCRAnnotation]]):
    def __init__(self, data_dir: str, level: OCRLevel):
        root = Path(data_dir) / "scadsai_german_handwriting_line_word_level_v01"

        image_dir = root / "images" / level.value

        csv_path = root / "ground_truth" / "csv" / f"{level}_annotations.csv"

        ocr_level = OCRLevel.line if level == "line" else OCRLevel.word

        annotations = _parse_annotations(csv_path, ocr_level)

        self.samples: list[tuple[Path, OCRAnnotation]] = []

        for image_name, annotation in annotations.items():
            image_path = image_dir / image_name

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


class ScaDSAI(Dataset[ScaDSAIConfig, SinglePageDocumentInstance]):
    def _download_urls(self) -> list[str]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION, homepage=_HOMEPAGE, license=_LICENSE
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(data_dir=data_dir, level=self.config.level)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()
