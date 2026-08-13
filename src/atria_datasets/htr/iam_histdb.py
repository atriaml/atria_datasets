from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._elements import OCRLevel
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._common import TextAnnotationTransform, TextFileIterator
from atria_datasets.utils import require_manual_path

_HOMEPAGE = "https://fki.tic.heia-fr.ch/databases/iam-historical-document-database"

from atria_datasets.registry import dataset_configs


@dataset_configs.register(name="iam_histdb")
@pydantic_dataclass(frozen=True)
class IAMHistDBConfig(DatasetConfig):
    collection: Literal["washington", "parzival", "saint_gall"] = "washington"

    def build_module(self, **kwargs: Any) -> IAMHistDB:
        return IAMHistDB(config=self, **kwargs)


class IAMHistDB(Dataset[IAMHistDBConfig, SinglePageDocumentInstance]):
    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        root = require_manual_path(
            data_dir=data_dir,
            expected_path=f"iam_histdb/{self.config.collection}",
            homepage=_HOMEPAGE,
            instructions="Extract the registered dataset, preserving paired line-image and .txt transcription names.",
        )
        return {self.config.collection: root}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=f"IAM Historical Document Database: {self.config.collection} line images and transcriptions.",
            homepage=_HOMEPAGE,
            license="Non-commercial research and teaching only",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> TextFileIterator:
        return TextFileIterator(
            root=Path(data_dir) / "iam_histdb" / self.config.collection
        )

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return TextAnnotationTransform(level=OCRLevel.line)
