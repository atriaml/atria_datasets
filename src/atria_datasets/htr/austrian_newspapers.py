from __future__ import annotations

from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import DatasetSplitType

from atria_datasets.htr._page_xml_dataset import PageXMLDataset


class AustrianNewspapers(PageXMLDataset):
    """NewsEye/READ OCR ground truth from Austrian newspapers: historical
    Fraktur print with PAGE-XML transcriptions."""

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


def austrian_newspapers(
    data_dir: str | None = None,
    access_token: str | None = None,
    split: DatasetSplitType | None = None,
) -> AustrianNewspapers:
    """Build the Austrian newspapers OCR ground-truth dataset."""
    return AustrianNewspapers(
        dataset_dir_name="austrian_newspapers",
        data_dir=data_dir,
        access_token=access_token,
        split=split,
    )
