from __future__ import annotations

from atria_core.datasets import UrlSpec

from atria_datasets.htr._page_xml_dataset import PageXMLDataset


class StABSRatsbuecher(PageXMLDataset):
    """Staatsarchiv Basel-Stadt Ratsbuch O 10 / Urfehdenbuch X: handwritten
    council records with PAGE-XML ground truth."""

    __module_name__ = "stabs_ratsbuecher_o10"

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
