from __future__ import annotations

from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import DatasetSplitType

from atria_datasets.htr._page_xml_dataset import PageXMLDataset


class StABSRatsbuecher(PageXMLDataset):
    """Staatsarchiv Basel-Stadt Ratsbuch O 10 / Urfehdenbuch X: handwritten
    council records with PAGE-XML ground truth."""

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


def stabs_ratsbuecher_o10(
    data_dir: str | None = None,
    access_token: str | None = None,
    split: DatasetSplitType | None = None,
) -> StABSRatsbuecher:
    """Build the Staatsarchiv Basel-Stadt Ratsbuch O 10 dataset.

    Args:
        data_dir: Where to read and write data.
        access_token: Credential for datasets behind authentication.
        split: Build only this split, instead of every available one.
    """
    return StABSRatsbuecher(
        dataset_dir_name="stabs_ratsbuecher_o10",
        data_dir=data_dir,
        access_token=access_token,
        split=split,
    )
