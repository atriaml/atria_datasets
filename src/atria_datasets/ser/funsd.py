"""FUNSD form understanding dataset."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from atria_core.datasets import Dataset, DatasetConfig, UrlSpec
from atria_core.types import (
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    EntityLabelingAnnotation,
    Image,
    SinglePageDocumentInstance,
)

from atria_datasets.ser._legacy_content import content_from_words_and_segments
from atria_datasets.ser.utilities import _sorted_indices_in_reading_order

_HOMEPAGE = "https://guillaumejaume.github.io/FUNSD/"
_URLS = [UrlSpec(url=f"{_HOMEPAGE}dataset.zip", url_ext=".zip")]
_LABELS = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]
_SPLIT_DIRS = {
    DatasetSplitType.train: "training_data",
    DatasetSplitType.test: "testing_data",
}


class FUNSDConfig(DatasetConfig):
    """Options for converting the FUNSD form annotations."""

    apply_reading_order_correction: bool = True


@dataclass(frozen=True)
class _FUNSDRecord:
    image_path: Path
    annotation_path: Path


class _FUNSDSplit(Sequence[_FUNSDRecord]):
    def __init__(self, data_dir: str | Path, split: DatasetSplitType) -> None:
        split_dir = Path(data_dir) / "dataset" / "dataset" / _SPLIT_DIRS[split]
        annotations = split_dir / "annotations"
        self.samples = [
            _FUNSDRecord(
                image_path=split_dir / "images" / filename,
                annotation_path=annotations / f"{Path(filename).stem}.json",
            )
            for filename in sorted(os.listdir(split_dir / "images"))
        ]

    @overload
    def __getitem__(self, index: int) -> _FUNSDRecord: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_FUNSDRecord]: ...

    def __getitem__(self, index: int | slice) -> _FUNSDRecord | Sequence[_FUNSDRecord]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class _FUNSDTransform:
    def __init__(self, config: FUNSDConfig) -> None:
        self._config = config

    def __call__(self, sample: _FUNSDRecord) -> SinglePageDocumentInstance:
        with sample.annotation_path.open(encoding="utf8") as stream:
            annotation = json.load(stream)

        from PIL import Image as PILImage

        with PILImage.open(sample.image_path) as image:
            width, height = image.size

        words: list[str] = []
        boxes: list[list[float]] = []
        labels: list[str] = []
        segment_ids: list[int] = []
        for item in annotation["form"]:
            item_words = [word for word in item["words"] if word["text"].strip()]
            if not item_words:
                continue
            segment_id = max(segment_ids, default=-1) + 1
            kind = item["label"].upper()
            for index, word in enumerate(item_words):
                words.append(word["text"])
                x0, y0, x1, y1 = word["box"]
                boxes.append(
                    [
                        x0 / width,
                        y0 / height,
                        x1 / width,
                        y1 / height,
                    ]
                )
                labels.append(
                    "O" if kind == "OTHER" else f"{'B' if index == 0 else 'I'}-{kind}"
                )
                segment_ids.append(segment_id)

        if self._config.apply_reading_order_correction:
            order = _sorted_indices_in_reading_order(
                [tuple(box) for box in boxes]
            )
            words = [words[index] for index in order]
            boxes = [boxes[index] for index in order]
            labels = [labels[index] for index in order]
            segment_ids = [segment_ids[index] for index in order]

        content = content_from_words_and_segments(words, boxes, segment_ids)
        instance = SinglePageDocumentInstance(
            sample_id=sample.image_path.name,
            visual=Image(file_path=str(sample.image_path)),
            content=content,
        )
        return instance.add_annotation(
            EntityLabelingAnnotation(
                word_label_values=[_LABELS.index(label) for label in labels],
                word_label_names=labels,
            )
        )


class FUNSD(Dataset[SinglePageDocumentInstance, FUNSDConfig]):
    """FUNSD forms with word boxes and BIO semantic entity labels."""

    __module_name__ = "funsd"

    def _download_urls(self) -> list[UrlSpec]:
        return _URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="Forms annotated for named entity recognition and document understanding.",
            homepage=_HOMEPAGE,
            license="Apache-2.0",
            dataset_labels=DatasetLabels(ser=_LABELS),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLIT_DIRS)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _FUNSDSplit:
        return _FUNSDSplit(data_dir=data_dir, split=split)

    def _build_input_transform(
        self,
    ) -> Callable[[Any], SinglePageDocumentInstance]:
        return _FUNSDTransform(config=self.config)
