from __future__ import annotations

from typing import Any

from atria_core.datasets import DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._page_xml_dataset import PageXMLDataset
from atria_datasets.registry import datasets


@datasets.register(name="read_konzilsprotokolle")
@pydantic_dataclass(frozen=True)
class KonzilsprotokolleConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> Konzilsprotokolle:
        return Konzilsprotokolle(config=self, **kwargs)


class Konzilsprotokolle(PageXMLDataset):
    urls = [
        UrlSpec(
            url="https://zenodo.org/records/215383/files/german_konzilsprotokolle.tar.gz",
            url_ext=".tar.gz",
        )
    ]
    description = "READ German Konzilsprotokolle: 8,770 historical handwritten lines with PAGE-XML."
    homepage = "https://zenodo.org/records/215383"
    license_name = "CC BY 4.0"
