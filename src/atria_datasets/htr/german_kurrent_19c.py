from __future__ import annotations

from typing import Literal

from atria_core.datasets import DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import DatasetSplitType
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._common import PageXMLIterator
from atria_datasets.htr._page_xml_dataset import PageXMLDataset

_DTA_MARKERS = ("libelt", "hufeland", "erbkam", "auerbach")


@pydantic_dataclass(frozen=True)
class GermanKurrent19CConfig(DatasetConfig):
    """Params for the German Kurrent 19th-century dataset."""

    license_subset: Literal["cc_by_only", "all"] = "cc_by_only"


class GermanKurrent19C(PageXMLDataset[GermanKurrent19CConfig]):
    """German Kurrent pages and lines from 19th-century sources."""

    __config__ = GermanKurrent19CConfig

    urls = [
        UrlSpec(
            url="https://zenodo.org/records/17252677/files/data.zip", url_ext=".zip"
        )
    ]
    description = (
        "German Kurrent handwritten pages and lines from multiple 19th-century sources."
    )
    homepage = "https://zenodo.org/records/17252677"
    license_name = "Mixed: CC BY 4.0 and CC BY-NC-SA 4.0; see selected configuration"
    split_aliases = {
        DatasetSplitType.train: ("trainingset",),
        DatasetSplitType.validation: ("validationset",),
        DatasetSplitType.test: ("testset",),
    }

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> PageXMLIterator:
        iterator = super()._build_split_iterator(split=split, data_dir=data_dir)
        if self.config.license_subset == "cc_by_only":
            iterator.samples = [
                sample
                for sample in iterator.samples
                if any(marker in str(sample[1]).lower() for marker in _DTA_MARKERS)
            ]
        return iterator

    def _metadata(self):
        metadata = super()._metadata()
        if self.config.license_subset == "cc_by_only":
            return type(metadata)(
                description=f"{self.description} Deutsches Textarchiv sources only.",
                homepage=self.homepage,
                license="CC BY 4.0",
            )
        return metadata


def german_kurrent_19c(
    license_subset: Literal["cc_by_only", "all"] = "cc_by_only",
    data_dir: str | None = None,
    access_token: str | None = None,
    split: DatasetSplitType | None = None,
) -> GermanKurrent19C:
    """Build the German Kurrent 19th-century dataset."""
    return GermanKurrent19C(
        config=GermanKurrent19CConfig(license_subset=license_subset),
        dataset_dir_name="german_kurrent_19c",
        data_dir=data_dir,
        access_token=access_token,
        split=split,
    )
