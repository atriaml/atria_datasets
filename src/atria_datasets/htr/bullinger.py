from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._elements import OCRLevel
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._common import TextAnnotationTransform
from atria_datasets.registry import datasets
from atria_datasets.utils import require_manual_path

_HOMEPAGE = "https://tc11.cvc.uab.es/datasets/BullingerDB_1"
_TSV = {
    DatasetSplitType.train: "train/train_frequent.tsv",
    DatasetSplitType.validation: "valid/valid_frequent.tsv",
    DatasetSplitType.test: "test/test_frequent.tsv",
}


@datasets.register("bullinger")
@pydantic_dataclass(frozen=True)
class BullingerConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> Bullinger:
        return Bullinger(self, **kwargs)


class Bullinger(Dataset[BullingerConfig, SinglePageDocumentInstance]):
    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        root = require_manual_path(
            data_dir,
            "bullinger",
            homepage=_HOMEPAGE,
            instructions="Download the ground-truth TSV files and line images using the upstream bullinger-htr instructions.",
        )
        return {"bullinger": root}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="BullingerDB historical line images with writer-disjoint ground-truth splits.",
            homepage=_HOMEPAGE,
            license="CC BY-NC-ND 3.0",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_TSV)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> list[tuple[Path, str]]:
        root = Path(data_dir) / "bullinger"
        samples: list[tuple[Path, str]] = []
        with (root / _TSV[split]).open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                fields = [field for field in row if field]
                if len(fields) >= 2:
                    image_path = root / fields[0]
                    if image_path.exists():
                        samples.append((image_path, fields[-1]))
        return samples

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return TextAnnotationTransform(level=OCRLevel.line)
