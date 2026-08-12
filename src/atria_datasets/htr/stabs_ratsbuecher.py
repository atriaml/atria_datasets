from __future__ import annotations

from typing import Any

from atria_core.datasets import DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._page_xml_dataset import PageXMLDataset
from atria_datasets.registry import datasets


@datasets.register(name="stabs_ratsbuecher_o10")
@pydantic_dataclass(frozen=True)
class StABSRatsbuecherConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> StABSRatsbuecher:
        return StABSRatsbuecher(config=self, **kwargs)


class StABSRatsbuecher(PageXMLDataset):
    urls = [
        UrlSpec(
            url="https://zenodo.org/records/5153263/files/StABS_Ratsbuch_O_10.zip",
            url_ext=".zip",
        )
    ]
    description = (
        "Staatsarchiv Basel-Stadt Ratsbuch O 10 / Urfehdenbuch X PAGE-XML ground truth."
    )
    homepage = "https://zenodo.org/records/5153263"
    license_name = "CC BY-NC-SA 4.0"
