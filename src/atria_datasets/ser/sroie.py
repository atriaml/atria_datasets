"""SROIE receipt information extraction dataset."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from PIL import Image as PILImage

from atria_core.datasets import Dataset, UrlSpec
from atria_core.types import (
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    ElementArray,
    EntityLabelingAnnotation,
    Image,
    SinglePageDocumentInstance,
)

_HOMEPAGE = "https://rrc.cvc.uab.es/?ch=13"
_URL = "https://drive.google.com/file/d/1ZyxAw1d-9UvhgNLGRvsJK4gBCMf0VpGD/view?usp=sharing"
_LABELS = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]
_SPLIT_DIRS = {
    DatasetSplitType.train: "train",
    DatasetSplitType.test: "test",
}


@dataclass(frozen=True)
class _SROIERecord:
    image_path: Path
    annotation_path: Path


class _SROIESplit(Sequence[_SROIERecord]):
    def __init__(self, data_dir: str | Path, split: DatasetSplitType) -> None:
        split_dir = Path(data_dir) / "sroie" / "sroie" / _SPLIT_DIRS[split]
        image_dir = split_dir / "images"
        annotation_dir = split_dir / "tagged"
        self.samples = [
            _SROIERecord(
                image_path=image_dir / filename,
                annotation_path=annotation_dir / f"{Path(filename).stem}.json",
            )
            for filename in sorted(os.listdir(image_dir))
        ]

    @overload
    def __getitem__(self, index: int) -> _SROIERecord: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_SROIERecord]: ...

    def __getitem__(self, index: int | slice) -> _SROIERecord | Sequence[_SROIERecord]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class _SROIETransform:
    def __call__(self, sample: _SROIERecord) -> SinglePageDocumentInstance:
        with sample.annotation_path.open(encoding="utf8") as stream:
            raw_annotation = json.load(stream)

        with PILImage.open(sample.image_path) as image:
            width, height = image.size

        words: list[str] = []
        boxes: list[list[float]] = []
        label_names: list[str] = []
        for word, box, label in zip(
            raw_annotation["words"],
            raw_annotation["bbox"],
            raw_annotation["labels"],
            strict=True,
        ):
            word = word.strip()
            if not word:
                continue
            words.append(word)
            boxes.append(
                [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
            )
            label_names.append(label)

        return SinglePageDocumentInstance(
            sample_id=sample.image_path.name,
            visual=Image(file_path=str(sample.image_path)),
            content=DocumentContent(
                elements=ElementArray.from_words(
                    texts=words, bboxes=boxes, normalized=True
                )
            ),
        ).add_annotation(
            EntityLabelingAnnotation(
                word_label_values=[_LABELS.index(label) for label in label_names],
                word_label_names=label_names,
            )
        )


class SROIE(Dataset[SinglePageDocumentInstance]):
    """SROIE receipts with word boxes and company/date/address/total tags."""

    __module_name__ = "sroie"

    def _download_urls(self) -> list[UrlSpec]:
        return [
            UrlSpec(
                url=_URL,
                url_ext=".zip",
                rel_output_file_path="sroie.zip",
            )
        ]

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="SROIE receipts dataset with semantic entity labels.",
            homepage=_HOMEPAGE,
            license="Apache-2.0",
            citation="",
            dataset_labels=DatasetLabels(ser=_LABELS),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLIT_DIRS)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _SROIESplit:
        return _SROIESplit(data_dir=data_dir, split=split)

    def _build_input_transform(
        self,
    ) -> Callable[[Any], SinglePageDocumentInstance]:
        return _SROIETransform()
