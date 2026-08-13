"""ScaDS.AI German Line- and Word-Level Handwriting Dataset."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._annotations import TranscriptionAnnotation
from atria_core.types._generic._elements import OCRLevel
from atria_core.types._generic._image import Image
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import dataset_configs

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


def _row_text(row: dict[str, str], ocr_level: OCRLevel) -> str:
    keys = (f"{ocr_level.name}_text", "text", "transcription", "gt", "label")
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    raise ValueError(f"Missing transcription field for {ocr_level.name} sample: {row}")


@dataset_configs.register(name="scadsai_german_handwriting")
@pydantic_dataclass(frozen=True)
class ScaDSAIConfig(DatasetConfig):
    level: OCRLevel = OCRLevel.line

    def build_module(self, **kwargs: Any) -> ScaDSAI:
        return ScaDSAI(config=self, **kwargs)


class SplitIterator(Sequence[tuple[Path, str]]):
    def __init__(self, data_dir: str, ocr_level: OCRLevel):
        root = Path(data_dir) / "scadsai_german_handwriting_line_word_level_v01"

        image_dir = root / "images"
        csv_dir = root / "ground_truth" / "csv"
        if ocr_level not in {OCRLevel.word, OCRLevel.line, OCRLevel.page}:
            raise ValueError(f"Unsupported OCR level: {ocr_level}")

        image_dir = image_dir / f"{ocr_level.name}s"
        csv_path = csv_dir / f"{ocr_level.name}_annotations.csv"
        self.samples: list[tuple[Path, str]] = []

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_key = (
                    row["line_file"] if ocr_level == OCRLevel.line else row["word_file"]
                )
                text = _row_text(row=row, ocr_level=ocr_level)

                image_path = image_dir / image_key

                if not image_path.exists():
                    continue

                self.samples.append((image_path, text))

    def __getitem__(self, index: int) -> tuple[Path, str]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class InputTransform:
    def __init__(self, *, ocr_level: OCRLevel) -> None:
        self.ocr_level = ocr_level

    def __call__(self, sample: tuple[Path, str]) -> SinglePageDocumentInstance:
        image_path, text = sample

        return SinglePageDocumentInstance(
            sample_id=image_path.stem, visual=Image(file_path=str(image_path))
        ).add_annotation(
            annotation=TranscriptionAnnotation(text=text, level=self.ocr_level)
        )


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
        return SplitIterator(data_dir=data_dir, ocr_level=self.config.level)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform(ocr_level=self.config.level)
