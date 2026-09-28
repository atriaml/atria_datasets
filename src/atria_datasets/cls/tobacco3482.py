from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, overload

import bs4
import numpy as np
from atria_core.datasets import Dataset, DatasetConfig
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import (
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    ElementArray,
    OCRLevel,
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
    "The Tobacco3482 dataset consists of 3482 grayscale images in 10 classes. "
    "In this version, the dataset is split into 2782 training images, and 700 "
    "test images."
)
_HOMEPAGE = "https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg"
_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"
_HF_DATA_URL = "https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data"
_IMAGE_DATA_NAME = "tobacco3482"
_OCR_DATA_NAME = "tobacco3482_ocr"
_SPLIT_FILE_NAMES = {
    DatasetSplitType.train: "train.txt",
    DatasetSplitType.test: "test.txt",
}
_DATA_URLS = [
    UrlSpec(url=f"{_HF_DATA_URL}/{_IMAGE_DATA_NAME}.tar.gz"),
    UrlSpec(url=f"{_HF_DATA_URL}/{_OCR_DATA_NAME}.tar.gz"),
    *(UrlSpec(url=f"{_HF_DATA_URL}/{name}") for name in _SPLIT_FILE_NAMES.values()),
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


def _parse_hocr(hocr_path: Path) -> DocumentContent:
    """Parse a tesseract hocr file into flat word-level OCR elements with
    page-normalized boxes."""
    with open(hocr_path) as f:
        soup = bs4.BeautifulSoup(f, features="xml")

    pages = soup.find_all("div", {"class": "ocr_page"})
    image_size_str = pages[0]["title"].split("; bbox")[1]
    width, height = (
        int(v) for v in image_size_str[4 : image_size_str.find(";")].split()
    )

    ids: list[int] = []
    bboxes: list[tuple[float, float, float, float]] = []
    texts: list[str] = []
    confs: list[float] = []
    angles: list[float] = []

    for word in soup.find_all("span", {"class": "ocrx_word"}):
        text = word.text.strip()
        if not text:
            continue

        title = word["title"]
        conf = float(title[title.find(";") + 10 :])

        angle = 0.0
        parent_title = word.parent["title"]
        if "textangle" in parent_title:
            angle = float(parent_title.split("textangle")[1][1:3])

        x1, y1, x2, y2 = (int(v) for v in title[5 : title.find(";")].split())

        ids.append(len(ids))
        bboxes.append((x1 / width, y1 / height, x2 / width, y2 / height))
        texts.append(text)
        confs.append(conf)
        angles.append(angle)

    if not ids:
        return DocumentContent(elements=None)

    return DocumentContent(
        elements=ElementArray(
            ids=np.array(ids),
            parent_ids=np.full(len(ids), -1),
            levels=np.full(len(ids), OCRLevel.word.value),
            bboxes=np.asarray(bboxes, dtype=np.float64),
            texts=np.asarray(texts, dtype=object),
            confs=np.asarray(confs, dtype=np.float64),
            angles=np.asarray(angles, dtype=np.float64),
        )
    )


class Tobacco3482Config(DatasetConfig):
    load_ocr: bool = False
    """Attach word-level OCR parsed from the dataset's pre-computed hocr files."""


class Tobacco3482InputTransform:
    def __init__(self, load_ocr: bool) -> None:
        self.load_ocr = load_ocr

    def __call__(self, sample: tuple[Path, Path, int]) -> SinglePageDocumentInstance:
        image_file_path, ocr_file_path, label_index = sample
        content = _parse_hocr(ocr_file_path) if self.load_ocr else None
        return SinglePageDocumentInstance.from_image(
            image_file_path, content=content
        ).add_annotation(
            ClassificationAnnotation(
                label_value=label_index, label_name=_CLASSES[label_index]
            )
        )


class Tobacco3482SplitIterator(Sequence[tuple[Path, Path, int]]):
    """Yields (image path, hocr path, class index) in split-file order."""

    def __init__(self, data_dir: str, split: DatasetSplitType) -> None:
        if split not in _SPLIT_FILE_NAMES:
            raise ValueError(f"Unsupported split: {split}")
        split_file_path = Path(data_dir) / _SPLIT_FILE_NAMES[split]
        self.relative_image_paths = split_file_path.read_text().splitlines()
        self.image_data_dir = Path(data_dir) / _IMAGE_DATA_NAME
        self.ocr_data_dir = Path(data_dir) / _OCR_DATA_NAME

    @overload
    def __getitem__(self, index: int) -> tuple[Path, Path, int]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[Path, Path, int]]: ...

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(len(self))[index]]
        relative_image_path = self.relative_image_paths[index]
        label_index = _CLASSES.index(Path(relative_image_path).parent.name)
        ocr_file_path = self.ocr_data_dir / relative_image_path.replace(".jpg", ".hocr")
        return self.image_data_dir / relative_image_path, ocr_file_path, label_index

    def __len__(self) -> int:
        return len(self.relative_image_paths)


class Tobacco3482(Dataset[SinglePageDocumentInstance, Tobacco3482Config]):
    """Tobacco3482 scanned document images labelled with 10 document classes."""

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
        return list(_SPLIT_FILE_NAMES)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Tobacco3482SplitIterator:
        return Tobacco3482SplitIterator(data_dir=data_dir, split=split)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return Tobacco3482InputTransform(load_ocr=self.config.load_ocr)
