from __future__ import annotations

from typing import Literal

from atria_core.datasets import DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import DatasetSplitType
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._page_xml_dataset import PageXMLDataset


@pydantic_dataclass(frozen=True)
class KoenigsfeldenConfig(DatasetConfig):
    """Params for the Königsfelden PAGE-XML dataset."""

    # Zenodo's digitized_documents.zip explicitly excludes the cartularies
    # described by kbs_pageXML.zip. U-17 is therefore the only collection in
    # this release whose PAGE annotations can be paired with the bundled JPGs.
    collection: Literal["kbs", "u17"] = "u17"


class Koenigsfelden(PageXMLDataset[KoenigsfeldenConfig]):
    """Königsfelden Abbey and Bailiwick records with PAGE-XML."""

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


def koenigsfelden(
    collection: Literal["kbs", "u17"] = "u17",
    data_dir: str | None = None,
    access_token: str | None = None,
    split: DatasetSplitType | None = None,
) -> Koenigsfelden:
    """Build the Königsfelden PAGE-XML dataset."""
    return Koenigsfelden(
        config=KoenigsfeldenConfig(collection=collection),
        dataset_dir_name="koenigsfelden",
        data_dir=data_dir,
        access_token=access_token,
        split=split,
    )
