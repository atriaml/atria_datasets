from __future__ import annotations

from atria_core.datasets._download._download_manager import UrlSpec

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
