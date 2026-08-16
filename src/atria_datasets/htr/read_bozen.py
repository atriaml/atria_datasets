"""READ Bozen / Ratsprotokolle (ICFHR2016) dataset."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, overload

from atria_core.datasets import Dataset
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._image import Image

from atria_datasets.parsers import parse_page_xml

_DATA_URLS = [
    UrlSpec(
        url="https://zenodo.org/records/1164045/files/Train-And-Val-ICFHR-2016.tgz",
        url_ext=".tgz",
    ),
    UrlSpec(
        url="https://zenodo.org/records/1164045/files/Test-ICFHR-2016.tgz",
        url_ext=".tgz",
    ),
]

_DESCRIPTION = (
    "READ Bozen (Ratsprotokolle) dataset from the ICFHR2016 Handwritten "
    "Text Recognition competition: council-meeting minutes (1470-1805) "
    "from the city of Bozen, in Early Modern German, with line-level "
    "PAGE-XML ground truth."
)
_HOMEPAGE = "https://zenodo.org/records/1164045"
_LICENSE = "CC BY 4.0"

# (relative path to the PAGE-XML dir, whether images are co-located there).
# Confirmed against the real archives: Training/Validation keep each
# split's images alongside its PAGE-XML under "page/"; the separately
# published Test archive keeps images one level up instead.
_SPLIT_LAYOUT: dict[DatasetSplitType, tuple[str, bool]] = {
    DatasetSplitType.train: ("PublicData/Training/page", True),
    DatasetSplitType.validation: ("PublicData/Validation/page", True),
    DatasetSplitType.test: ("Test-ICFHR-2016/page", False),
}


class SplitIterator(Sequence[tuple[Path, Path]]):
    """Each sample is (image_path, page_xml_path)."""

    def __init__(self, data_dir: str, split: DatasetSplitType):
        page_rel, images_colocated = _SPLIT_LAYOUT[split]

        # The download manager's extraction wrapper directory name isn't
        # guaranteed to match the archive's own internal top-level folder
        # (e.g. gnhk.py ends up double-nested "gnhk-dataset/gnhk", while
        # scads_german.py doesn't) -- search for the known internal path
        # instead of assuming a fixed depth under data_dir.
        matches = list(Path(data_dir).rglob(page_rel))
        if not matches:
            raise FileNotFoundError(
                f"Could not find {page_rel!r} under {data_dir} for split {split.value}"
            )
        page_dir = matches[0]
        image_dir = page_dir if images_colocated else page_dir.parent

        self.samples: list[tuple[Path, Path]] = []
        for xml_path in sorted(page_dir.glob("*.xml")):
            image_path = image_dir / f"{xml_path.stem}.JPG"
            if image_path.exists():
                self.samples.append((image_path, xml_path))

    @overload
    def __getitem__(self, index: int) -> tuple[Path, Path]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[Path, Path]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> tuple[Path, Path] | Sequence[tuple[Path, Path]]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class InputTransform:
    def __call__(self, sample: tuple[Path, Path]) -> SinglePageDocumentInstance:
        image_path, xml_path = sample

        annotation = parse_page_xml(xml_path=xml_path)

        return SinglePageDocumentInstance(
            sample_id=image_path.stem, visual=Image(file_path=str(image_path))
        ).add_annotation(annotation=annotation)


class ReadBozen(Dataset[SinglePageDocumentInstance]):
    """READ Bozen council minutes with line-level PAGE-XML."""

    __module_name__ = "read_bozen"

    def _download_urls(self) -> list[UrlSpec]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION, homepage=_HOMEPAGE, license=_LICENSE
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [
            DatasetSplitType.train,
            DatasetSplitType.validation,
            DatasetSplitType.test,
        ]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(data_dir=data_dir, split=split)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()
