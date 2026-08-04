from __future__ import annotations

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

from atria_datasets.htr._common import TextSidecarIterator, TextSidecarTransform
from atria_datasets.registry import datasets
from atria_datasets.utils import require_manual_path

_HOMEPAGE = "https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/"


@datasets.register("cvl")
@pydantic_dataclass(frozen=True)
class CVLConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> CVL:
        return CVL(self, **kwargs)


class CVL(Dataset[CVLConfig, SinglePageDocumentInstance]):
    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        root = require_manual_path(
            data_dir,
            "cvl",
            homepage=_HOMEPAGE,
            instructions="Download CVL 1.1 under its non-commercial terms. Export each desired word crop with a same-stem UTF-8 .txt transcription using the official XML parser.",
        )
        return {"cvl": root}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="CVL English/German handwriting word crops exported from its official XML annotations.",
            homepage=_HOMEPAGE,
            license="CC BY-NC 3.0",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> TextSidecarIterator:
        return TextSidecarIterator(Path(data_dir) / "cvl")

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return TextSidecarTransform(level=OCRLevel.word)
