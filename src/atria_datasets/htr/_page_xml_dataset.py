"""Base implementation for archive-backed PAGE-XML datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)

from atria_datasets.htr._common import PageXMLIterator, PageXMLTransform


class PageXMLDataset(Dataset[DatasetConfig, SinglePageDocumentInstance]):
    urls: ClassVar[list[Any]] = []
    description: ClassVar[str]
    homepage: ClassVar[str]
    license_name: ClassVar[str]
    split_aliases: ClassVar[dict[DatasetSplitType, tuple[str, ...]]] = {
        DatasetSplitType.train: ()
    }

    def _download_urls(self) -> list[Any]:
        return self.urls

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=self.description,
            homepage=self.homepage,
            license=self.license_name,
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(self.split_aliases)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> PageXMLIterator:
        return PageXMLIterator(
            root=Path(data_dir), split_aliases=self.split_aliases[split]
        )

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return PageXMLTransform()
