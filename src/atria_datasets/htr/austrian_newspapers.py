from __future__ import annotations

from typing import Any

from atria_core.datasets import DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import DatasetSplitType
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._page_xml_dataset import PageXMLDataset
from atria_datasets.registry import datasets


@datasets.register(name="austrian_newspapers")
@pydantic_dataclass(frozen=True)
class AustrianNewspapersConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> AustrianNewspapers:
        return AustrianNewspapers(config=self, **kwargs)


class AustrianNewspapers(PageXMLDataset):
    urls = [
        UrlSpec(
            url="https://zenodo.org/records/3387369/files/TrainingSet_ONB_Newseye_GT_M1%2B.zip",
            url_ext=".zip",
        ),
        UrlSpec(
            url="https://zenodo.org/records/3387369/files/ValidationSet_ONB_Newseye_GT_M1%2B.zip",
            url_ext=".zip",
        ),
    ]
    description = "Open NewsEye/READ OCR ground truth from Austrian newspapers; historical Fraktur print, not handwriting."
    homepage = "https://zenodo.org/records/3387369"
    license_name = "CC BY 4.0"
    split_aliases = {
        DatasetSplitType.train: ("train",),
        DatasetSplitType.validation: ("valid", "val"),
    }
