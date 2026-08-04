from __future__ import annotations

from typing import Any, Literal

from atria_core.datasets import DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import DatasetSplitType
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._common import PageXMLIterator
from atria_datasets.htr._page_xml_dataset import PageXMLDataset
from atria_datasets.registry import datasets

_DTA_MARKERS = ("libelt", "hufeland", "erbkam", "auerbach")


@datasets.register("german_kurrent_19c")
@pydantic_dataclass(frozen=True)
class GermanKurrent19CConfig(DatasetConfig):
    license_subset: Literal["cc_by_only", "all"] = "cc_by_only"

    def build_module(self, **kwargs: Any) -> GermanKurrent19C:
        return GermanKurrent19C(self, **kwargs)


class GermanKurrent19C(PageXMLDataset):
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
        iterator = super()._build_split_iterator(split, data_dir)
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
