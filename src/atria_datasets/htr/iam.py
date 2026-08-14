"""IAM Handwriting Database (offline), supplied manually by the user."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np
from atria_core.datasets import Dataset, DatasetConfig
from atria_core.types import (
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._annotations import OCRAnnotation
from atria_core.types._generic._elements import OCRLevel
from atria_core.types._generic._image import Image

from atria_datasets.htr._common import get_image_size
from atria_datasets.parsers import (
    IAMRecord,
    parse_iam_ascii,
    parse_iam_forms,
    parse_iam_split,
)
from atria_datasets.utils import require_manual_path

_HOMEPAGE = "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"
_AACHEN_SPLITS = {
    DatasetSplitType.train: "train.uttlist",
    DatasetSplitType.validation: "validation.uttlist",
    DatasetSplitType.test: "test.uttlist",
}


def _iam_root(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    for candidate in (data_dir / "iam", data_dir):
        if (candidate / "ascii" / "forms.txt").is_file() and (
            candidate / "forms"
        ).is_dir():
            return candidate
    return require_manual_path(
        data_dir=data_dir,
        expected_path="iam",
        homepage=_HOMEPAGE,
        instructions=(
            "Extract forms/ and ascii/ there, and extract the OpenSLR 56 "
            "Aachen split archive so iam/splits/*.uttlist exists."
        ),
    )


class IAMConfig(DatasetConfig):
    """Params for the IAM offline handwriting database.

    Attributes:
        include_bad_segmentations: Keep lines and words whose segmentation the
            corpus marks as unreliable.
        crop_to_handwriting: Crop each page to the ground-truth handwriting
            extent instead of keeping the full scanned form.
    """

    include_bad_segmentations: bool = False
    crop_to_handwriting: bool = True


def _bbox(record: IAMRecord, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = crop_box
    width, height = right - left, bottom - top
    return np.clip(
        np.asarray(
            [
                record.x - left,
                record.y - top,
                record.x + record.width - left,
                record.y + record.height - top,
            ],
            dtype=np.float64,
        )
        / np.asarray([width, height, width, height]),
        0.0,
        1.0,
    )


def _build_annotation(
    lines: list[IAMRecord],
    words_by_line: dict[str, list[IAMRecord]],
    crop_box: tuple[int, int, int, int],
) -> OCRAnnotation:
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
        bboxes.append(_bbox(record=line, crop_box=crop_box))
        texts.append(line.text)
        for word in words_by_line.get(line.sample_id, []):
            ids.append(len(ids))
            parent_ids.append(line_index)
            levels.append(OCRLevel.word.value)
            bboxes.append(_bbox(record=word, crop_box=crop_box))
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


def _handwriting_crop(
    lines: list[IAMRecord], width: int, height: int
) -> tuple[int, int, int, int]:
    return (
        max(0, min(line.x for line in lines)),
        max(0, min(line.y for line in lines)),
        min(width, max(line.x + line.width for line in lines)),
        min(height, max(line.y + line.height for line in lines)),
    )


@dataclass(frozen=True)
class IAMFormSample:
    image_path: Path
    annotation: OCRAnnotation
    writer_id: str
    writer_label: int
    crop_box: tuple[int, int, int, int] | None


class IAMSplitIterator(Sequence[IAMFormSample]):
    def __init__(
        self,
        root: Path,
        split: DatasetSplitType,
        include_bad_segmentations: bool,
        crop_to_handwriting: bool,
    ) -> None:
        forms = parse_iam_forms(path=root / "ascii" / "forms.txt")
        lines = parse_iam_ascii(path=root / "ascii" / "lines.txt")
        words = parse_iam_ascii(path=root / "ascii" / "words.txt")
        form_ids = parse_iam_split(path=root / "splits" / _AACHEN_SPLITS[split])
        writer_ids = sorted({form.writer_id for form in forms.values()})
        writer_labels = {writer_id: label for label, writer_id in enumerate(writer_ids)}
        image_paths = {path.stem: path for path in (root / "forms").rglob("*.png")}
        lines_by_form: dict[str, list[IAMRecord]] = defaultdict(list)
        words_by_line: dict[str, list[IAMRecord]] = defaultdict(list)
        for record in lines.values():
            if include_bad_segmentations or record.segmentation_status == "ok":
                lines_by_form[record.form_id].append(record)
        for record in words.values():
            if include_bad_segmentations or record.segmentation_status == "ok":
                words_by_line[record.line_id].append(record)

        self.samples: list[IAMFormSample] = []
        for form_id in sorted(form_ids):
            form_lines = sorted(
                lines_by_form.get(form_id, []), key=lambda item: item.sample_id
            )
            image_path = image_paths.get(form_id)
            form = forms.get(form_id)
            if image_path is not None and form is not None and form_lines:
                width, height = get_image_size(image_path=image_path)
                annotation_box = (
                    _handwriting_crop(lines=form_lines, width=width, height=height)
                    if crop_to_handwriting
                    else (0, 0, width, height)
                )
                annotation = _build_annotation(
                    lines=form_lines,
                    words_by_line=words_by_line,
                    crop_box=annotation_box,
                )
                self.samples.append(
                    IAMFormSample(
                        image_path=image_path,
                        annotation=annotation,
                        writer_id=form.writer_id,
                        writer_label=writer_labels[form.writer_id],
                        crop_box=annotation_box if crop_to_handwriting else None,
                    )
                )

    @overload
    def __getitem__(self, index: int) -> IAMFormSample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[IAMFormSample]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class IAMInputTransform:
    def __call__(self, sample: IAMFormSample) -> SinglePageDocumentInstance:
        instance = SinglePageDocumentInstance(
            sample_id=sample.image_path.stem,
            visual=Image(file_path=str(sample.image_path), crop_box=sample.crop_box),
        ).add_annotation(annotation=sample.annotation)
        return instance.add_annotation(
            annotation=ClassificationAnnotation(
                label_value=sample.writer_label, label_name=sample.writer_id
            )
        )


class IAM(Dataset[SinglePageDocumentInstance, IAMConfig]):
    """IAM offline English handwriting database, supplied manually by the user."""

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        root = _iam_root(data_dir=data_dir)
        return {"iam": root}

    def _metadata(self) -> DatasetMetadata:
        forms = parse_iam_forms(
            path=_iam_root(data_dir=self.data_dir) / "ascii" / "forms.txt"
        )
        return DatasetMetadata(
            description=(
                "IAM offline English handwriting database cropped to the "
                "ground-truth handwriting extent, with writer identities, page, "
                "line, and word annotations, and the Aachen split."
            ),
            homepage=_HOMEPAGE,
            license="CC BY-NC-SA 4.0",
            dataset_labels=DatasetLabels(
                classification=sorted({form.writer_id for form in forms.values()})
            ),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_AACHEN_SPLITS)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> IAMSplitIterator:
        return IAMSplitIterator(
            root=_iam_root(data_dir=data_dir),
            split=split,
            include_bad_segmentations=self.config.include_bad_segmentations,
            crop_to_handwriting=self.config.crop_to_handwriting,
        )

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return IAMInputTransform()
