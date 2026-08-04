from __future__ import annotations

from typing import Any

from atria_core.datasets import DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import DatasetSplitType
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._page_xml_dataset import PageXMLDataset
from atria_datasets.registry import datasets


@datasets.register("icdar2017_escher")
@pydantic_dataclass(frozen=True)
class ICDAR2017EscherConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> ICDAR2017Escher:
        return ICDAR2017Escher(self, **kwargs)


class ICDAR2017Escher(PageXMLDataset):
    # Train-A and Test-A are the fully line-annotated HTR benchmark subsets.
    urls = [
        UrlSpec(
            url="https://zenodo.org/records/835489/files/Train-A.tbz2",
            # Canonical spelling understood by Atria's archive extractor.
            url_ext=".tar.bz2",
        ),
        UrlSpec(
            url="https://zenodo.org/records/835489/files/Test-A.tgz", url_ext=".tgz"
        ),
    ]
    description = "Fully annotated Train-A/Test-A subsets of the ICDAR2017 READ HTR benchmark, primarily Alfred Escher letters."
    homepage = "https://zenodo.org/records/835489"
    license_name = "CC BY 4.0"
    split_aliases = {
        DatasetSplitType.train: ("train-a", "train_a"),
        DatasetSplitType.test: ("test-a", "test_a"),
    }
