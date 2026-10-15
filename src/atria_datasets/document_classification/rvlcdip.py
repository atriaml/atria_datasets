"""RVL-CDIP document image classification dataset."""

from __future__ import annotations

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
    SinglePageDocumentInstance,
)

from atria_datasets.parsers.hocr import parse_tesseract_hocr

_CITATION = """\
@inproceedings{harley2015icdar,
    title = {Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval},
    author = {Adam W Harley and Alex Ufkes and Konstantinos G Derpanis},
    booktitle = {International Conference on Document Analysis and Recognition ({ICDAR})},
    year = {2015}
}
"""
_DESCRIPTION = (
    "RVL-CDIP contains 400,000 grayscale document images in 16 classes, with "
    "320,000 training, 40,000 validation, and 40,000 test images."
)
_HOMEPAGE = "https://www.cs.cmu.edu/~aharley/rvl-cdip/"
_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"
_IMAGE_DATA_NAME = "rvl-cdip"
_OCR_DATA_NAME = "rvl-cdip-ocr"
_BASE_URL = "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data"
_IMAGE_URL = (
    "https://huggingface.co/datasets/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz"
)
_OCR_URL = f"{_BASE_URL}/rvl-cdip-ocr.tar.gz"
_METADATA_URLS = {
    "default": {
        "labels/default/train.txt": f"{_BASE_URL}/tobacco3482_excluded/train.txt",
        "labels/default/test.txt": f"{_BASE_URL}/tobacco3482_excluded/test.txt",
        "labels/default/val.txt": f"{_BASE_URL}/tobacco3482_excluded/val.txt",
    },
    "tobacco3482_included": {
        "labels/tobacco3482_included/train.txt": f"{_BASE_URL}/default/train.txt",
        "labels/tobacco3482_included/test.txt": f"{_BASE_URL}/default/test.txt",
        "labels/tobacco3482_included/val.txt": f"{_BASE_URL}/default/val.txt",
    },
}
_CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]
_SPLIT_FILES = {
    DatasetSplitType.train: "train.txt",
    DatasetSplitType.test: "test.txt",
    DatasetSplitType.validation: "val.txt",
}


class RvlCdipConfig(DatasetConfig):
    """Options for the full RVL-CDIP dataset."""

    load_ocr: bool = False
    type: str = "default"
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(frozen=True)
class _RvlCdipRecord:
    image_path: Path
    ocr_path: Path
    label_value: int


class _RvlCdipSplit(Sequence[_RvlCdipRecord]):
    def __init__(
        self, split: DatasetSplitType, data_dir: str | Path, config: RvlCdipConfig
    ) -> None:
        if config.type not in _METADATA_URLS:
            raise ValueError(f"Unknown RVL-CDIP dataset type: {config.type}")

        root = Path(data_dir)
        label_path = root / "labels" / config.type / _SPLIT_FILES[split]
        with label_path.open() as stream:
            rows = stream.read().splitlines()

        maximum = {
            DatasetSplitType.train: config.max_train_samples,
            DatasetSplitType.validation: config.max_validation_samples,
            DatasetSplitType.test: config.max_test_samples,
        }[split]
        if maximum is not None:
            rows = rows[:maximum]

        image_dir = root / _IMAGE_DATA_NAME / "images"
        ocr_dir = root / _OCR_DATA_NAME / "images"
        self.samples = []
        for row in rows:
            image_path, label_value = row.split(" ")
            self.samples.append(
                _RvlCdipRecord(
                    image_path=image_dir / image_path,
                    ocr_path=ocr_dir / image_path.replace(".tif", ".hocr.lstm"),
                    label_value=int(label_value),
                )
            )

    @overload
    def __getitem__(self, index: int) -> _RvlCdipRecord: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_RvlCdipRecord]: ...

    def __getitem__(
        self, index: int | slice
    ) -> _RvlCdipRecord | Sequence[_RvlCdipRecord]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class _RvlCdipTransform:
    def __init__(self, config: RvlCdipConfig) -> None:
        self._config = config

    def __call__(self, sample: _RvlCdipRecord) -> SinglePageDocumentInstance:
        content = (
            parse_tesseract_hocr(sample.ocr_path.read_text(encoding="utf-8"))
            if self._config.load_ocr
            else None
        )
        instance = SinglePageDocumentInstance(
            sample_id=sample.image_path.name,
            visual=Image(file_path=str(sample.image_path)),
            content=content,
        ).add_annotation(
            ClassificationAnnotation(
                label_value=sample.label_value,
                label_name=_CLASSES[sample.label_value],
            )
        )
        return instance


class _RvlCdipBase[T_Config: RvlCdipConfig](
    Dataset[SinglePageDocumentInstance, T_Config]
):
    """Shared loading behavior for RVL-CDIP configurations."""

    __abstract__ = True

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        (Path(data_dir) / "labels" / self.config.type).mkdir(
            parents=True, exist_ok=True
        )
        return super()._download(data_dir=data_dir, access_token=access_token)

    def _download_urls(self) -> list[UrlSpec]:
        if self.config.type not in _METADATA_URLS:
            raise ValueError(f"Unknown RVL-CDIP dataset type: {self.config.type}")
        metadata = [
            UrlSpec(url=url, rel_output_file_path=output_path)
            for output_path, url in _METADATA_URLS[self.config.type].items()
        ]
        return metadata + [UrlSpec(url=_IMAGE_URL), UrlSpec(url=_OCR_URL)]

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            citation=_CITATION,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            dataset_labels=DatasetLabels(classification=_CLASSES),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLIT_FILES)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _RvlCdipSplit:
        return _RvlCdipSplit(split=split, data_dir=data_dir, config=self.config)

    def _build_input_transform(
        self,
    ) -> Callable[[Any], SinglePageDocumentInstance]:
        return _RvlCdipTransform(config=self.config)


class RvlCdip(_RvlCdipBase[RvlCdipConfig]):
    """Full RVL-CDIP dataset, uncapped and without OCR by default."""

    __module_name__ = "rvlcdip"


