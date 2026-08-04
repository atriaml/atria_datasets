from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._annotations import TranscriptionAnnotation
from atria_core.types._generic._elements import OCRLevel
from atria_core.types._generic._image import Image
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets

_HOMEPAGE = "https://zenodo.org/records/1492267"
_ARCHIVE_NAME = "cvl-database-1-1"
_URLS = [UrlSpec(url=f"{_HOMEPAGE}/files/{_ARCHIVE_NAME}.zip", url_ext=".zip")]


@datasets.register("cvl")
@pydantic_dataclass(frozen=True)
class CVLConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> CVL:
        return CVL(self, **kwargs)


class CVLWordIterator(Sequence[tuple[Path, str]]):
    """Load the official word crops whose filenames contain their transcript."""

    def __init__(self, data_dir: str, split: DatasetSplitType) -> None:
        root = Path(data_dir)
        relative_split_dir = Path(f"{split.value}set") / "words"
        candidates = (
            root / _ARCHIVE_NAME / _ARCHIVE_NAME / relative_split_dir,
            root / _ARCHIVE_NAME / relative_split_dir,
            root / relative_split_dir,
        )
        split_dir = next((path for path in candidates if path.is_dir()), candidates[0])

        self.samples: list[tuple[Path, str]] = []
        for image_path in sorted(split_dir.rglob("*.tif")):
            # <writer>-<text>-<line>-<word>-<transcription>.tif. Limiting the
            # split preserves hyphens that are part of the transcription.
            name_parts = image_path.stem.split("-", 4)
            if len(name_parts) == 5 and name_parts[4]:
                self.samples.append((image_path, name_parts[4]))

    def __getitem__(self, index: int) -> tuple[Path, str]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class CVLWordTransform:
    def __call__(self, sample: tuple[Path, str]) -> SinglePageDocumentInstance:
        image_path, transcription = sample
        return SinglePageDocumentInstance(
            sample_id=image_path.stem, visual=Image(file_path=str(image_path))
        ).add_annotation(
            TranscriptionAnnotation(text=transcription, level=OCRLevel.word)
        )


class CVL(Dataset[CVLConfig, SinglePageDocumentInstance]):
    def _download_urls(self) -> list[UrlSpec]:
        return _URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="CVL 1.1 English/German handwriting word crops using the official train/test split.",
            homepage=_HOMEPAGE,
            license="CC BY-NC 4.0",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> CVLWordIterator:
        return CVLWordIterator(data_dir, split)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return CVLWordTransform()
