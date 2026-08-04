"""IAM Handwriting Database (offline), supplied manually by the user."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, overload

import numpy as np
from atria_core.datasets import Dataset, DatasetConfig
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._annotations import OCRAnnotation
from atria_core.types._generic._elements import OCRLevel
from atria_core.types._generic._image import Image
from PIL import Image as PILImage
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.parsers import IAMRecord, parse_iam_ascii, parse_iam_split
from atria_datasets.registry import datasets
from atria_datasets.utils import require_manual_path

_HOMEPAGE = "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"
_AACHEN_SPLITS = {
    DatasetSplitType.train: "train.uttlist",
    DatasetSplitType.validation: "validation.uttlist",
    DatasetSplitType.test: "test.uttlist",
}


@datasets.register("iam")
@pydantic_dataclass(frozen=True)
class IAMConfig(DatasetConfig):
    include_bad_segmentations: bool = False

    def build_module(self, **kwargs: Any) -> IAM:
        return IAM(self, **kwargs)


def _bbox(record: IAMRecord, width: int, height: int) -> np.ndarray:
    return np.clip(
        np.asarray(
            [record.x, record.y, record.x + record.width, record.y + record.height],
            dtype=np.float64,
        )
        / np.asarray([width, height, width, height]),
        0.0,
        1.0,
    )


def _build_annotation(
    image_path: Path, lines: list[IAMRecord], words_by_line: dict[str, list[IAMRecord]]
) -> OCRAnnotation:
    width, height = PILImage.open(image_path).size
    ids = [0]
    parent_ids = [-1]
    levels = [OCRLevel.page.value]
    bboxes = [np.asarray([0.0, 0.0, 1.0, 1.0])]
    texts = ["\n".join(line.text for line in lines)]
    for line in lines:
        line_index = len(ids)
        ids.append(line_index)
        parent_ids.append(0)
        levels.append(OCRLevel.line.value)
        bboxes.append(_bbox(line, width, height))
        texts.append(line.text)
        for word in words_by_line.get(line.sample_id, []):
            ids.append(len(ids))
            parent_ids.append(line_index)
            levels.append(OCRLevel.word.value)
            bboxes.append(_bbox(word, width, height))
            texts.append(word.text)
    annotation = OCRAnnotation(
        ids=np.asarray(ids),
        parent_ids=np.asarray(parent_ids),
        levels=np.asarray(levels),
        bboxes=np.asarray(bboxes, dtype=np.float64),
        texts=np.asarray(texts, dtype=object),
    )
    annotation.validate_hierarchy()
    return annotation


class IAMSplitIterator(Sequence[tuple[Path, OCRAnnotation]]):
    def __init__(
        self, root: Path, split: DatasetSplitType, include_bad_segmentations: bool
    ) -> None:
        lines = parse_iam_ascii(root / "ascii" / "lines.txt")
        words = parse_iam_ascii(root / "ascii" / "words.txt")
        form_ids = parse_iam_split(root / "splits" / _AACHEN_SPLITS[split])
        lines_by_form: dict[str, list[IAMRecord]] = defaultdict(list)
        words_by_line: dict[str, list[IAMRecord]] = defaultdict(list)
        for record in lines.values():
            if include_bad_segmentations or record.segmentation_status == "ok":
                lines_by_form[record.form_id].append(record)
        for record in words.values():
            if include_bad_segmentations or record.segmentation_status == "ok":
                words_by_line[record.line_id].append(record)

        self.samples: list[tuple[Path, OCRAnnotation]] = []
        for form_id in sorted(form_ids):
            form_lines = sorted(
                lines_by_form.get(form_id, []), key=lambda item: item.sample_id
            )
            image_path = root / "forms" / f"{form_id}.png"
            if image_path.exists() and form_lines:
                annotation = _build_annotation(image_path, form_lines, words_by_line)
                self.samples.append((image_path, annotation))

    @overload
    def __getitem__(self, index: int) -> tuple[Path, OCRAnnotation]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[Path, OCRAnnotation]]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class IAMInputTransform:
    def __call__(
        self, sample: tuple[Path, OCRAnnotation]
    ) -> SinglePageDocumentInstance:
        image_path, annotation = sample
        return SinglePageDocumentInstance(
            sample_id=image_path.stem, visual=Image(file_path=str(image_path))
        ).add_annotation(annotation)


class IAM(Dataset[IAMConfig, SinglePageDocumentInstance]):
    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        root = require_manual_path(
            data_dir,
            "iam",
            homepage=_HOMEPAGE,
            instructions=(
                "Extract forms/ and ascii/ there, and extract the OpenSLR 56 "
                "Aachen split archive so iam/splits/*.uttlist exists."
            ),
        )
        return {"iam": root}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=(
                "IAM offline English handwriting database with page, line, and "
                "word annotations and the Aachen split."
            ),
            homepage=_HOMEPAGE,
            license="CC BY-NC-SA 4.0",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_AACHEN_SPLITS)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> IAMSplitIterator:
        return IAMSplitIterator(
            Path(data_dir) / "iam", split, self.config.include_bad_segmentations
        )

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return IAMInputTransform()
