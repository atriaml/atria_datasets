from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from lxml import etree
import numpy as np
from atria_core.datasets import UrlSpec
from atria_core.types import (
    DatasetSplitType,
    Image,
    OCRAnnotation,
    OCRLevel,
    SinglePageDocumentInstance,
)

from atria_datasets.htr._common import get_image_size
from atria_datasets.htr._page_xml_dataset import PageXMLDataset


class LAMIterator(Sequence[tuple[Path, Path]]):
    def __init__(self, root: Path) -> None:
        self.samples = []
        for xml_path in sorted(root.rglob("full_pages/*/xml/*.xml")):
            form = etree.parse(str(xml_path)).getroot()
            image_path = xml_path.parent.parent / "img" / form.get("filename", "")
            if image_path.is_file():
                self.samples.append((image_path, xml_path))

    def __getitem__(self, index: int | slice) -> Any:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class LAMTransform:
    def __call__(self, sample: tuple[Path, Path]) -> SinglePageDocumentInstance:
        image_path, xml_path = sample
        width, height = get_image_size(image_path)
        form = etree.parse(str(xml_path)).getroot()
        lines = form.findall("./handwritten-part/line")
        texts = [line.get("text", "") for line in lines]
        bboxes = [
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
            bboxes=np.asarray([[0, 0, width, height], *bboxes], dtype=np.float64),
            texts=np.asarray(["\n".join(texts), *texts], dtype=object),
        )
        annotation.validate_hierarchy()
        return SinglePageDocumentInstance(
            sample_id=image_path.stem, visual=Image(file_path=str(image_path))
        ).add_annotation(annotation=annotation)


class LAM(PageXMLDataset):
    """LAM page scans and line-level XML annotations."""

    __module_name__ = "lam"

    urls = [
        UrlSpec(
            url="https://www.kaggle.com/api/v1/datasets/download/vpippi/lam-dataset",
            url_ext=".zip",
            rel_output_file_path="lam.zip",
        )
    ]
    description = "LAM manuscript pages with line-level XML transcriptions."
    homepage = "https://www.kaggle.com/datasets/vpippi/lam-dataset"
    license_name = "See the source dataset terms."
    split_aliases = {DatasetSplitType.train: ()}

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> LAMIterator:
        return LAMIterator(root=Path(data_dir))

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return LAMTransform()
