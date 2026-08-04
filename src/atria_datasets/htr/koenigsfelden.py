from __future__ import annotations

from typing import Any, Literal

from atria_core.datasets import DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._page_xml_dataset import PageXMLDataset
from atria_datasets.registry import datasets


@datasets.register("koenigsfelden")
@pydantic_dataclass(frozen=True)
class KoenigsfeldenConfig(DatasetConfig):
    # Zenodo's digitized_documents.zip explicitly excludes the cartularies
    # described by kbs_pageXML.zip. U-17 is therefore the only collection in
    # this release whose PAGE annotations can be paired with the bundled JPGs.
    collection: Literal["kbs", "u17"] = "u17"

    def build_module(self, **kwargs: Any) -> Koenigsfelden:
        return Koenigsfelden(self, **kwargs)


class Koenigsfelden(PageXMLDataset):
    description = "Charters and records of Königsfelden Abbey and Bailiwick, with PAGE-XML ground truth."
    homepage = "https://zenodo.org/records/5179361"
    license_name = "CC BY 4.0"

    def _download_urls(self) -> list[UrlSpec]:
        if self.config.collection == "kbs":
            raise ValueError(
                "The Koenigsfelden Zenodo release does not bundle the cartulary "
                "images required by kbs_pageXML.zip. Use collection='u17' (the "
                "default), whose images are included in digitized_documents.zip."
            )
        page_xml = f"{self.config.collection}_pageXML.zip"
        return [
            UrlSpec(
                url=f"https://zenodo.org/records/5179361/files/{page_xml}",
                url_ext=".zip",
            ),
            UrlSpec(
                url="https://zenodo.org/records/5179361/files/digitized_documents.zip",
                url_ext=".zip",
            ),
        ]
