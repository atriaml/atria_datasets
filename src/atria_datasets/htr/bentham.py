from __future__ import annotations

from atria_core.datasets import UrlSpec
from atria_core.types import DatasetSplitType

from atria_datasets.htr._page_xml_dataset import PageXMLDataset


class Bentham(PageXMLDataset):
    """Bentham handwritten manuscript pages with PAGE-XML ground truth, from
    the ImageCLEF 2016 release."""

    __module_name__ = "bentham"

    urls = [
        UrlSpec(url=f"https://zenodo.org/records/52994/files/{name}", url_ext=".zip")
        for name in (
            "pages_train_jpg.zip",
            "pages_train_xml.zip",
            "pages_devel_jpg_1.zip",
            "pages_devel_jpg_2.zip",
            "pages_devel_xml.zip",
            "pages_test_jpg.zip",
            "pages_test_xml.zip",
        )
    ]
    description = "Bentham handwritten manuscript pages and PAGE-XML from the ImageCLEF 2016 release."
    homepage = "https://zenodo.org/records/52994"
    license_name = "CC BY-NC-SA 4.0"
    split_aliases = {
        DatasetSplitType.train: ("train",),
        DatasetSplitType.validation: ("devel", "valid"),
        DatasetSplitType.test: ("test",),
    }
