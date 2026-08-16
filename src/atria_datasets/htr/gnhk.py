"""GNHK (GoodNotes Handwriting Kollection) dataset."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast, overload

import numpy as np
from atria_core.datasets import Dataset, UrlSpec
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    Image,
    OCRAnnotation,
    SinglePageDocumentInstance,
)
from numpy.typing import NDArray

from atria_datasets.htr._common import get_image_size

_DATA_URLS = [
    UrlSpec(
        url="https://www.kaggle.com/api/v1/datasets/download/evandu/gnhk-dataset",
        url_ext=".zip",
    )
]

_DESCRIPTION = (
    "GNHK (GoodNotes Handwriting Kollection) dataset containing English "
    "handwritten text images with word-level polygon annotations."
)

_HOMEPAGE = "https://www.kaggle.com/datasets/evandu/gnhk-dataset"
_LICENSE = "CC BY 4.0"


def _parse_manifest(
    manifest_path: Path,
) -> dict[str, list[tuple[str, NDArray[np.float64], NDArray[np.float64]]]]:
    """image name -> list of (text, absolute-pixel bbox (4,), polygon (P, 2)).

    GNHK's SageMaker-Ground-Truth-style JSONL has no line-grouping field --
    every word in `annotations.texts` is independent, so there is no
    hierarchy to preserve beyond a flat word level."""
    annotations = {}

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(s=line)

            words = []

            for item in data["annotations"]["texts"]:
                polygon = np.asarray(
                    [[point["x"], point["y"]] for point in item["polygon"]],
                    dtype=np.float64,
                )
                xs, ys = polygon[:, 0], polygon[:, 1]
                bbox = np.asarray(
                    [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64
                )
                words.append((item["text"], bbox, polygon))

            annotations[data["source-ref"]] = words

    return annotations


class SplitIterator(Sequence[tuple[Path, OCRAnnotation]]):
    def __init__(self, data_dir: str, split: DatasetSplitType):
        root = Path(data_dir)

        split_dir = root / "gnhk-dataset" / "gnhk" / f"{split.value}_data" / split.value

        manifest_path = split_dir / f"{split.value}.manifest"

        annotations = _parse_manifest(manifest_path=manifest_path)
        self.samples = [
            (split_dir / image_name, words)
            for image_name, words in annotations.items()
            if (split_dir / image_name).exists() and words
        ]
        self._annotation_cache: dict[int, OCRAnnotation] = {}

    @overload
    def __getitem__(self, index: int) -> tuple[Path, OCRAnnotation]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[Path, OCRAnnotation]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> tuple[Path, OCRAnnotation] | Sequence[tuple[Path, OCRAnnotation]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self.samples)

        image_path, words = self.samples[index]
        annotation = self._annotation_cache.get(index)
        if annotation is None:
            width, height = get_image_size(image_path=image_path, exif_rotation=True)

            texts = [text for text, _, _ in words]
            bboxes = np.stack([bbox for _, bbox, _ in words]) / np.array([
                width,
                height,
                width,
                height,
            ])
            polygons = [polygon / np.array([width, height]) for _, _, polygon in words]
            annotation = cast(
                OCRAnnotation,
                OCRAnnotation.from_words(
                    texts=texts, bboxes=bboxes, segmentations=polygons
                ),
            )
            self._annotation_cache[index] = annotation

        return image_path, annotation

    def __len__(self) -> int:
        return len(self.samples)


class InputTransform:
    def __call__(
        self, sample: tuple[Path, OCRAnnotation]
    ) -> SinglePageDocumentInstance:
        image_path, annotation = sample

        return SinglePageDocumentInstance(
            sample_id=image_path.stem, visual=Image(file_path=str(image_path))
        ).add_annotation(annotation=annotation)


class GNHK(Dataset[SinglePageDocumentInstance]):
    """GoodNotes handwriting images with word-level polygons."""

    __module_name__ = "gnhk"

    def _download_urls(self) -> list[UrlSpec]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION, homepage=_HOMEPAGE, license=_LICENSE
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(data_dir=data_dir, split=split)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()
