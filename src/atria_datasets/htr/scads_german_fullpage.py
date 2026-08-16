from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets import Dataset, UrlSpec
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    OCRLevel,
    SinglePageDocumentInstance,
)

from atria_datasets.htr._common import TextAnnotationTransform, TextFileIterator


class ScaDSAIFullPage(Dataset[SinglePageDocumentInstance]):
    """ScaDS.AI German handwritten full pages and transcriptions."""

    __module_name__ = "scadsai_german_fullpage"

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
        return TextFileIterator(root=data_dir)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return TextAnnotationTransform(level=OCRLevel.page)
