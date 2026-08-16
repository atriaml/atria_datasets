"""Base implementation for archive-backed PAGE-XML datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from atria_core.datasets import Dataset, DatasetConfig, UrlSpec
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)

from atria_datasets.htr._common import PageXMLIterator, PageXMLTransform

T_PageXMLConfig = TypeVar("T_PageXMLConfig", bound=DatasetConfig, default=DatasetConfig)


class PageXMLDataset[T_PageXMLConfig: DatasetConfig = DatasetConfig](
    Dataset[SinglePageDocumentInstance, T_PageXMLConfig]
):
    """Dataset of scanned pages paired with PAGE-XML ground truth, downloaded
    as archives and read straight off the extracted directory tree.

    Subclasses supply the archive URLs and the directory-name fragments that
    identify each split.
    """

    __abstract__ = True

    urls: ClassVar[list[UrlSpec]] = []
    description: ClassVar[str]
    homepage: ClassVar[str]
    license_name: ClassVar[str]
    split_aliases: ClassVar[dict[DatasetSplitType, tuple[str, ...]]] = {
        DatasetSplitType.train: ()
    }

    def _download_urls(self) -> list[UrlSpec]:
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
