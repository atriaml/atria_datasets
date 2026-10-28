"""CORD receipt understanding dataset."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, overload

import pyarrow.parquet as parquet
from PIL import Image as PILImage

from atria_core.datasets import AtriaDownloadManager, Dataset, UrlSpec
from atria_core.datasets._download._download_file_info import DownloadFileInfo
from atria_core.datasets._download._file_downloader import FileDownloader
from atria_core.types import (
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    EntityLabelingAnnotation,
    Image,
    SinglePageDocumentInstance,
)

from atria_datasets.ser.utilities import content_from_words_and_segments

_HOMEPAGE = "https://github.com/clovaai/cord"
_BASE_HF_REPO = "https://huggingface.co/datasets/naver-clova-ix/cord-v2"
_DATA_URLS = [
    f"{_BASE_HF_REPO}/resolve/main/data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/train-00001-of-00004-7dbbe248962764c5.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/train-00002-of-00004-688fe1305a55e5cc.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/train-00003-of-00004-2d0cd200555ed7fd.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/test-00000-of-00001-9c204eb3f4e11791.parquet",
]
_LABELS = [
    "O",
    "B-MENU.NM",
    "B-MENU.NUM",
    "B-MENU.UNITPRICE",
    "B-MENU.CNT",
    "B-MENU.DISCOUNTPRICE",
    "B-MENU.PRICE",
    "B-MENU.ITEMSUBTOTAL",
    "B-MENU.VATYN",
    "B-MENU.ETC",
    "B-MENU.SUB.NM",
    "B-MENU.SUB.UNITPRICE",
    "B-MENU.SUB.CNT",
    "B-MENU.SUB.PRICE",
    "B-MENU.SUB.ETC",
    "B-VOID_MENU.NM",
    "B-VOID_MENU.PRICE",
    "B-SUB_TOTAL.SUBTOTAL_PRICE",
    "B-SUB_TOTAL.DISCOUNT_PRICE",
    "B-SUB_TOTAL.SERVICE_PRICE",
    "B-SUB_TOTAL.OTHERSVC_PRICE",
    "B-SUB_TOTAL.TAX_PRICE",
    "B-SUB_TOTAL.ETC",
    "B-TOTAL.TOTAL_PRICE",
    "B-TOTAL.TOTAL_ETC",
    "B-TOTAL.CASHPRICE",
    "B-TOTAL.CHANGEPRICE",
    "B-TOTAL.CREDITCARDPRICE",
    "B-TOTAL.EMONEYPRICE",
    "B-TOTAL.MENUTYPE_CNT",
    "B-TOTAL.MENUQTY_CNT",
    "I-MENU.NM",
    "I-MENU.NUM",
    "I-MENU.UNITPRICE",
    "I-MENU.CNT",
    "I-MENU.DISCOUNTPRICE",
    "I-MENU.PRICE",
    "I-MENU.ITEMSUBTOTAL",
    "I-MENU.VATYN",
    "I-MENU.ETC",
    "I-MENU.SUB.NM",
    "I-MENU.SUB.UNITPRICE",
    "I-MENU.SUB.CNT",
    "I-MENU.SUB.PRICE",
    "I-MENU.SUB.ETC",
    "I-VOID_MENU.NM",
    "I-VOID_MENU.PRICE",
    "I-SUB_TOTAL.SUBTOTAL_PRICE",
    "I-SUB_TOTAL.DISCOUNT_PRICE",
    "I-SUB_TOTAL.SERVICE_PRICE",
    "I-SUB_TOTAL.OTHERSVC_PRICE",
    "I-SUB_TOTAL.TAX_PRICE",
    "I-SUB_TOTAL.ETC",
    "I-TOTAL.TOTAL_PRICE",
    "I-TOTAL.TOTAL_ETC",
    "I-TOTAL.CASHPRICE",
    "I-TOTAL.CHANGEPRICE",
    "I-TOTAL.CREDITCARDPRICE",
    "I-TOTAL.EMONEYPRICE",
    "I-TOTAL.MENUTYPE_CNT",
    "I-TOTAL.MENUQTY_CNT",
]
_SPLITS = {
    DatasetSplitType.train: "train",
    DatasetSplitType.validation: "validation",
    DatasetSplitType.test: "test",
}
_DOWNLOAD_TIMEOUT_SECONDS = 300


class _CORDDownloadManager(AtriaDownloadManager):
    """Use a longer read timeout for CORD's large Hugging Face shards."""

    def _download_files(self, download_file_infos: list[DownloadFileInfo]) -> None:
        for download_file_info in download_file_infos:
            if (
                download_file_info.download_path.exists()
                or download_file_info.extracted_path.exists()
                or download_file_info.output_path.exists()
            ):
                continue
            downloader = FileDownloader.from_url(
                parsed_url=download_file_info.parsed_url,
                timeout=_DOWNLOAD_TIMEOUT_SECONDS,
            )
            downloader.download(download_file_info=download_file_info)


class _CORDSplit(Sequence[dict[str, Any]]):
    """Rows read from the original CORD parquet shards for one split."""

    def __init__(self, data_dir: str | Path, split: DatasetSplitType) -> None:
        split_name = _SPLITS[split]
        data_files = sorted(Path(data_dir).glob(f"{split_name}-*.parquet"))
        if not data_files:
            raise FileNotFoundError(
                f"No CORD {split_name} parquet shards found directly under {data_dir}"
            )
        self.samples = [
            row
            for data_file in data_files
            for row in parquet.read_table(data_file).to_pylist()
        ]

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class _CORDTransform:
    def __call__(self, sample: dict[str, Any]) -> SinglePageDocumentInstance:
        image = PILImage.open(io.BytesIO(sample["image"]["bytes"]))
        image.load()
        width, height = image.size
        annotation = json.loads(sample["ground_truth"])

        words: list[str] = []
        boxes: list[list[float]] = []
        label_names: list[str] = []
        segment_ids: list[int] = []
        for line in annotation["valid_line"]:
            line_words = [word for word in line["words"] if word["text"].strip()]
            if not line_words:
                continue
            segment_id = max(segment_ids, default=-1) + 1
            category = line["category"]
            category = category.upper().replace("MENU.SUB_", "MENU.SUB.")
            for index, word in enumerate(line_words):
                quad = word["quad"]
                x0, y0, x1, y1 = (
                    max(0, quad["x1"]),
                    max(0, quad["y1"]),
                    quad["x3"],
                    quad["y3"],
                )
                if y1 < y0:
                    y0, y1 = y1, y0
                if x1 < x0:
                    x0, x1 = x1, x0
                words.append(word["text"])
                boxes.append([x0 / width, y0 / height, x1 / width, y1 / height])
                if line["category"] == "other":
                    label_names.append("O")
                else:
                    label_names.append(
                        f"{'B' if index == 0 else 'I'}-{category}"
                    )
                segment_ids.append(segment_id)

        image_id = str(annotation["meta"]["image_id"])
        instance = SinglePageDocumentInstance(
            sample_id=image_id,
            visual=Image(content=image),
            content=content_from_words_and_segments(words, boxes, segment_ids),
        )
        return instance.add_annotation(
            EntityLabelingAnnotation(
                word_label_values=[_LABELS.index(label) for label in label_names],
                word_label_names=label_names,
            )
        )


class CORD(Dataset[SinglePageDocumentInstance]):
    """CORD receipts with word boxes and semantic entity labels."""

    __module_name__ = "cord"

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        data_path = Path(data_dir)
        manager = _CORDDownloadManager(
            data_dir=data_path,
            download_dir=data_path / ".download_cache",
        )
        return manager.download_and_extract(
            self._download_urls(), extract=False, access_token=access_token
        )

    def _download_urls(self) -> list[UrlSpec]:
        return [UrlSpec(url=url) for url in _DATA_URLS]

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="CORD receipt understanding dataset.",
            homepage=_HOMEPAGE,
            license="Apache-2.0",
            citation="",
            dataset_labels=DatasetLabels(ser=_LABELS),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLITS)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _CORDSplit:
        return _CORDSplit(data_dir=data_dir, split=split)

    def _build_input_transform(
        self,
    ) -> Callable[[Any], SinglePageDocumentInstance]:
        return _CORDTransform()
