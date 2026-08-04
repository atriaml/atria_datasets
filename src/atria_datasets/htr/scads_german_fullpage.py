from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._elements import OCRLevel
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._common import TextAnnotationTransform, TextFileIterator
from atria_datasets.registry import datasets


@datasets.register("scadsai_german_fullpage")
@pydantic_dataclass(frozen=True)
class ScaDSAIFullPageConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> ScaDSAIFullPage:
        return ScaDSAIFullPage(self, **kwargs)


class ScaDSAIFullPage(Dataset[ScaDSAIFullPageConfig, SinglePageDocumentInstance]):
    def _download_urls(self) -> list[UrlSpec]:
        return [
            UrlSpec(
                url="https://zenodo.org/records/18283705/files/scadsai_german_handwriting_full_page_v01.tar.gz",
                url_ext=".tar.gz",
            )
        ]

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="77 German handwritten full pages paired with UTF-8 transcriptions; intended for evaluation.",
            homepage="https://zenodo.org/records/18283705",
            license="CC BY 4.0",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.test]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> TextFileIterator:
        return TextFileIterator(data_dir)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return TextAnnotationTransform(level=OCRLevel.page)
