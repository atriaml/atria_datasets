from __future__ import annotations

from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import DatasetSplitType

from atria_datasets.htr._page_xml_dataset import PageXMLDataset


class Konzilsprotokolle(PageXMLDataset):
    """READ German Konzilsprotokolle: historical handwritten council minutes
    with PAGE-XML ground truth."""

    urls = [
        UrlSpec(
            url="https://zenodo.org/records/215383/files/german_konzilsprotokolle.tar.gz",
            url_ext=".tar.gz",
        )
    ]
    description = "READ German Konzilsprotokolle: 8,770 historical handwritten lines with PAGE-XML."
    homepage = "https://zenodo.org/records/215383"
    license_name = "CC BY 4.0"


def read_konzilsprotokolle(
    data_dir: str | None = None,
    access_token: str | None = None,
    split: DatasetSplitType | None = None,
) -> Konzilsprotokolle:
    """Build the READ German Konzilsprotokolle dataset.

    Args:
        data_dir: Where to read and write data.
        access_token: Credential for datasets behind authentication.
        split: Build only this split, instead of every available one.
    """
    return Konzilsprotokolle(
        dataset_dir_name="read_konzilsprotokolle",
        data_dir=data_dir,
        access_token=access_token,
        split=split,
    )
