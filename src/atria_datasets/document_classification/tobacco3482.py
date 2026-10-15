"""Tobacco3482 document classification dataset."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from atria_core.datasets import Dataset, DatasetConfig, UrlSpec
from atria_core.types import (
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    Image,
    OCRAnnotation,
    SinglePageDocumentInstance,
)

_CITATION = """\
@article{Kumar2014StructuralSF,
    title={Structural similarity for document image classification and retrieval},
    author={Jayant Kumar and Peng Ye and David S. Doermann},
    journal={Pattern Recognit. Lett.},
    year={2014},
    volume={43},
    pages={119-126}
}
"""
_DESCRIPTION = (
    "Tobacco3482 contains document images from 10 classes. This version uses "
    "the supplied train and test splits."
)
_HOMEPAGE = "https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg"
_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"
_IMAGE_DATA_NAME = "tobacco3482"
_OCR_DATA_NAME = "tobacco3482_ocr"
_DATA_URLS = [
    UrlSpec(
        url="https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/tobacco3482.tar.gz"
    ),
    UrlSpec(
        url="https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/tobacco3482_ocr.tar.gz"
    ),
    UrlSpec(
        url="https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/train.txt"
    ),
    UrlSpec(
        url="https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/test.txt"
    ),
]
_CLASSES = [
    "Letter",
    "Resume",
    "Scientific",
    "ADVE",
    "Email",
    "Report",
    "News",
    "Memo",
    "Form",
    "Note",
]
_SPLIT_FILES = {
    DatasetSplitType.train: "train.txt",
    DatasetSplitType.test: "test.txt",
}


class Tobacco3482Config(DatasetConfig):
    """Options for loading the original image and OCR data."""

    load_ocr: bool = True


@dataclass(frozen=True)
class _Tobacco3482Record:
    image_path: Path
    ocr_path: Path
    label_value: int


class _Tobacco3482Split(Sequence[_Tobacco3482Record]):
    def __init__(self, data_dir: str | Path, split: DatasetSplitType) -> None:
        root = Path(data_dir)
        split_path = root / _SPLIT_FILES[split]
        with split_path.open() as stream:
            image_paths = stream.read().splitlines()
        random.shuffle(image_paths)

        image_dir = root / _IMAGE_DATA_NAME
        ocr_dir = root / _OCR_DATA_NAME
        self.samples = [
            _Tobacco3482Record(
                image_path=image_dir / image_path,
                ocr_path=ocr_dir / image_path.replace(".jpg", ".hocr"),
                label_value=_CLASSES.index(Path(image_path).parent.name),
            )
            for image_path in image_paths
        ]

    @overload
    def __getitem__(self, index: int) -> _Tobacco3482Record: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_Tobacco3482Record]: ...

    def __getitem__(
        self, index: int | slice
    ) -> _Tobacco3482Record | Sequence[_Tobacco3482Record]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


def _load_hocr(path: Path, image_size: tuple[int, int]) -> OCRAnnotation:
    """Read the supplied Tesseract hOCR words into Atria's OCR representation."""
    from lxml import html

    root = html.fromstring(path.read_bytes())
    width, height = image_size
    texts: list[str] = []
    boxes: list[list[float]] = []
    nodes = root.xpath(
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' ocrx_word ')]"
    )
    for node in nodes:
        text = node.text_content().strip()
        title = node.get("title", "")
        bbox = next(
            (part for part in title.split(";") if part.strip().startswith("bbox ")),
            None,
        )
        if not text or bbox is None:
            continue
        x0, y0, x1, y1 = (float(value) for value in bbox.strip().split()[1:5])
        texts.append(text)
        boxes.append([x0 / width, y0 / height, x1 / width, y1 / height])
    annotation = OCRAnnotation.from_words(
        texts=texts, bboxes=boxes, normalized=True
    )
    assert isinstance(annotation, OCRAnnotation)
    return annotation


class _Tobacco3482Transform:
    def __init__(self, config: Tobacco3482Config) -> None:
        self._config = config

    def __call__(self, sample: _Tobacco3482Record) -> SinglePageDocumentInstance:
        instance = SinglePageDocumentInstance(
            sample_id=sample.image_path.name,
            visual=Image(file_path=str(sample.image_path)),
        ).add_annotation(
            ClassificationAnnotation(
                label_value=sample.label_value,
                label_name=_CLASSES[sample.label_value],
            )
        )
        if self._config.load_ocr:
            from PIL import Image as PILImage

            with PILImage.open(sample.image_path) as image:
                ocr = _load_hocr(sample.ocr_path, image.size)
            instance = instance.add_annotation(ocr)
        return instance


class Tobacco3482(Dataset[SinglePageDocumentInstance, Tobacco3482Config]):
    """Tobacco3482 images and their 10 document classification labels."""

    __module_name__ = "tobacco3482"

    def _download_urls(self) -> list[UrlSpec]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            dataset_labels=DatasetLabels(classification=_CLASSES),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLIT_FILES)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _Tobacco3482Split:
        return _Tobacco3482Split(data_dir=data_dir, split=split)

    def _build_input_transform(
        self,
    ) -> Callable[[Any], SinglePageDocumentInstance]:
        return _Tobacco3482Transform(config=self.config)
