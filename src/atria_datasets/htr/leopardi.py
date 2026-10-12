from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, overload

from lxml import etree
import numpy as np
from atria_core.datasets import Dataset, UrlSpec
from atria_core.types import (
    DatasetSplitType,
    DatasetMetadata,
    Image,
    OCRAnnotation,
    OCRLevel,
    SinglePageDocumentInstance,
)

from atria_datasets.htr._common import get_image_size


class LeopardiIterator(Sequence[tuple[Path, Path]]):
    def __init__(self, root: str, split: DatasetSplitType):
        split_name = "val" if split == DatasetSplitType.validation else split.value
        self.samples = []
        for xml_path in sorted(Path(root).rglob("xml/*.xml")):
            form = etree.parse(str(xml_path)).getroot()
            image_name = form.get("filename")
            if not image_name:
                continue
            image_path = xml_path.parent.parent / "pages" / split_name / image_name
            if image_path.is_file():
                self.samples.append((image_path, xml_path))

    @overload
    def __getitem__(self, index: int) -> tuple[Path, Path]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[Path, Path]]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class LeopardiTransform:
    def __call__(self, sample: tuple[Path, Path]) -> SinglePageDocumentInstance:
        image_path, xml_path = sample
        form = etree.parse(str(xml_path)).getroot()
        lines = form.findall("./handwritten-part/line")
        width, height = get_image_size(image_path)
        line_texts = [line.get("text", "") for line in lines]
        line_boxes = [
            [
                float(line.get("x", "0")),
                float(line.get("y", "0")),
                float(line.get("x", "0")) + float(line.get("w", "0")),
                float(line.get("y", "0")) + float(line.get("h", "0")),
            ]
            for line in lines
        ]
        annotation = OCRAnnotation(
            ids=np.arange(len(lines) + 1),
            parent_ids=np.asarray([-1, *([0] * len(lines))]),
            levels=np.asarray(
                [OCRLevel.page.value, *([OCRLevel.line.value] * len(lines))]
            ),
            bboxes=np.asarray([[0, 0, width, height], *line_boxes], dtype=np.float64),
            texts=np.asarray(["\n".join(line_texts), *line_texts], dtype=object),
        )
        annotation.validate_hierarchy()
        return SinglePageDocumentInstance(
            sample_id=image_path.stem, visual=Image(file_path=str(image_path))
        ).add_annotation(annotation=annotation)


class Leopardi(Dataset[SinglePageDocumentInstance]):
    """Leopardi pages with line-level XML transcriptions and bounding boxes."""

    __module_name__ = "leopardi"

    def _download_urls(self) -> list[UrlSpec]:
        return [
            UrlSpec(
                url="https://github.com/aimagelab/HWD/releases/download/leopardi/leopardi.zip",
                url_ext=".zip",
            )
        ]

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="Giacomo Leopardi manuscript pages with line transcriptions.",
            homepage="https://github.com/aimagelab/HWD/releases/tag/leopardi",
            license="See the source dataset terms.",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [
            DatasetSplitType.train,
            DatasetSplitType.validation,
            DatasetSplitType.test,
        ]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> LeopardiIterator:
        return LeopardiIterator(root=data_dir, split=split)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return LeopardiTransform()
