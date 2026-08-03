"""ScaDS.AI German Full-Page Handwriting dataset."""

from __future__ import annotations

import uuid
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
from atria_core.types._generic._image import Image
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets

_DATA_URLS = [
    UrlSpec(
        url="https://zenodo.org/records/18301532/files/scadsai_german_handwriting_line_word_level_v01.tar.gz?download=1",
        url_ext=".tar.gz",
    )
]

_DESCRIPTION = (
    "ScaDS.AI German Full-Page Handwriting dataset containing 77 handwritten "
    "German Wikipedia excerpts with plain text ground truth."
)

_HOMEPAGE = "https://zenodo.org/records/18283705"
_LICENSE = "CC BY 4.0"


@datasets.register("scadsai_german_fullpage")
@pydantic_dataclass(frozen=True)
class ScaDSAIGermanFullPageConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> ScaDSAIGermanFullPage:
        return ScaDSAIGermanFullPage(self, **kwargs)


class SplitIterator(Sequence[tuple[Path, Path]]):
    def __init__(self, data_dir: str, split: DatasetSplitType):
        root = Path(data_dir)

        image_dir = root / "images"
        text_dir = root / "ground_truth"
        print("image_dir", image_dir)

        self.samples: list[tuple[Path, Path]] = []

        for image_path in sorted(image_dir.glob("*.jpg")):
            text_path = text_dir / f"{image_path.stem}.txt"

            if text_path.exists():
                self.samples.append((image_path, text_path))

    def __getitem__(self, index: int) -> tuple[Path, Path]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class InputTransform:
    def __call__(self, sample: tuple[Path, Path]) -> SinglePageDocumentInstance:
        image_path, text_path = sample

        with open(text_path, encoding="utf-8") as f:
            text = f.read().strip()

        return SinglePageDocumentInstance(
            sample_id=str(uuid.uuid4()), visual=Image(file_path=str(image_path))
        ).add_annotation(TranscriptionAnnotation(text=text))


class ScaDSAIGermanFullPage(
    Dataset[ScaDSAIGermanFullPageConfig, SinglePageDocumentInstance]
):
    def _download_urls(self) -> list[UrlSpec]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION, homepage=_HOMEPAGE, license=_LICENSE
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(data_dir=data_dir, split=split)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()


dataset = ScaDSAIGermanFullPageConfig().build_module()

train = dataset.split_iterator(DatasetSplitType.train)

print(len(train))
print(train[0])

# cached = Cacher(FileStorageType.MSGPACK).cache(dataset)

# cached_train = cached.split_iterator(DatasetSplitType.train)

print(len(train))
print(train[0].load()._annotations)
