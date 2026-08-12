from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import (
    ClassificationAnnotation,
    DatasetLabels,
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


@datasets.register(name="cvl")
@pydantic_dataclass(frozen=True)
class CVLConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> CVL:
        return CVL(config=self, **kwargs)


def _dataset_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    candidates = (
        root / _ARCHIVE_NAME / _ARCHIVE_NAME,
        root / _ARCHIVE_NAME,
        root,
    )
    return next(
        (path for path in candidates if (path / "trainset" / "words").is_dir()),
        candidates[0],
    )


def _writer_ids(root: Path) -> list[str]:
    return sorted(
        {
            writer_dir.name
            for split_name in ("trainset", "testset")
            for writer_dir in (root / split_name / "words").iterdir()
            if writer_dir.is_dir() and writer_dir.name.isdigit()
        }
    )


@dataclass(frozen=True)
class CVLWordSample:
    image_path: Path
    transcription: str
    writer_id: str
    writer_label: int


class CVLWordIterator(Sequence[CVLWordSample]):
    """Load the official word crops whose filenames contain their transcript."""

    def __init__(self, data_dir: str, split: DatasetSplitType) -> None:
        root = _dataset_root(data_dir=data_dir)
        split_dir = root / f"{split.value}set" / "words"
        writer_labels = {
            writer_id: label
            for label, writer_id in enumerate(_writer_ids(root=root))
        }

        self.samples: list[CVLWordSample] = []
        for image_path in sorted(split_dir.rglob("*.tif")):
            # <writer>-<text>-<line>-<word>-<transcription>.tif. Limiting the
            # split preserves hyphens that are part of the transcription.
            name_parts = image_path.stem.split("-", 4)
            if (
                len(name_parts) == 5
                and all(part.isdigit() for part in name_parts[:4])
                and name_parts[4]
                and name_parts[0] in writer_labels
            ):
                self.samples.append(
                    CVLWordSample(
                        image_path=image_path,
                        transcription=name_parts[4],
                        writer_id=name_parts[0],
                        writer_label=writer_labels[name_parts[0]],
                    )
                )

    def __getitem__(self, index: int) -> CVLWordSample:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class CVLWordTransform:
    def __call__(self, sample: CVLWordSample) -> SinglePageDocumentInstance:
        instance = SinglePageDocumentInstance(
            sample_id=sample.image_path.stem,
            visual=Image(file_path=str(sample.image_path)),
        ).add_annotation(
            annotation=TranscriptionAnnotation(
                text=sample.transcription, level=OCRLevel.word
            )
        )
        return instance.add_annotation(
            annotation=ClassificationAnnotation(
                label_value=sample.writer_label,
                label_name=sample.writer_id,
            )
        )


class CVL(Dataset[CVLConfig, SinglePageDocumentInstance]):
    def _download_urls(self) -> list[UrlSpec]:
        return _URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=(
                "CVL 1.1 English/German handwriting word crops with writer "
                "identities and the official train/test split."
            ),
            homepage=_HOMEPAGE,
            license="CC BY-NC 3.0",
            citation=(
                "Florian Kleber, Stefan Fiel, Markus Diem, and Robert "
                "Sablatnig. CVL-Database: An Off-line Database for Writer "
                "Retrieval, Writer Identification and Word Spotting. "
                "ICDAR 2013, pp. 560-564."
            ),
            dataset_labels=DatasetLabels(
                classification=_writer_ids(root=_dataset_root(data_dir=self.data_dir))
            ),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> CVLWordIterator:
        return CVLWordIterator(data_dir=data_dir, split=split)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return CVLWordTransform()
