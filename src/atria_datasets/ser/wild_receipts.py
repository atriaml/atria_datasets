"""WildReceipt key information extraction dataset."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from atria_core.datasets import Dataset, DatasetConfig, UrlSpec
from atria_core.types import (
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    ElementArray,
    EntityLabelingAnnotation,
    Image,
    SinglePageDocumentInstance,
)

from atria_datasets.ser.utilities import _sorted_indices_in_reading_order

_DESCRIPTION = "WildReceipts Dataset"
_DATA_URL = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"
_CLASSES = [
    "B-Store_name_value",
    "B-Store_name_key",
    "B-Store_addr_value",
    "B-Store_addr_key",
    "B-Tel_value",
    "B-Tel_key",
    "B-Date_value",
    "B-Date_key",
    "B-Time_value",
    "B-Time_key",
    "B-Prod_item_value",
    "B-Prod_item_key",
    "B-Prod_quantity_value",
    "B-Prod_quantity_key",
    "B-Prod_price_value",
    "B-Prod_price_key",
    "B-Subtotal_value",
    "B-Subtotal_key",
    "B-Tax_value",
    "B-Tax_key",
    "B-Tips_value",
    "B-Tips_key",
    "B-Total_value",
    "B-Total_key",
    "O",
]
_SPLITS = {
    DatasetSplitType.train: "train",
    DatasetSplitType.test: "test",
}


class WildReceiptsConfig(DatasetConfig):
    """Options for the WildReceipt text ordering."""

    apply_reading_order_correction: bool = True


@dataclass(frozen=True)
class _WildReceiptRecord:
    split_dir: Path
    sample_info: dict[str, Any]
    label_map: dict[int, str]


class _WildReceiptsSplit(Sequence[_WildReceiptRecord]):
    def __init__(self, data_dir: str | Path, split: DatasetSplitType) -> None:
        self.split_file_path = (
            Path(data_dir) / "wildreceipt" / "wildreceipt" / f"{_SPLITS[split]}.txt"
        )
        split_dir = self.split_file_path.parent
        label_map: dict[int, str] = {}
        with (split_dir / "class_list.txt").open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    index, label = line.split(maxsplit=1)
                    label_map[int(index)] = label.strip()

        self.samples = []
        with self.split_file_path.open() as stream:
            for line in stream:
                sample_info = json.loads(line.rstrip("\n\r"))
                self.samples.append(
                    _WildReceiptRecord(
                        split_dir=split_dir,
                        sample_info=sample_info,
                        label_map=label_map,
                    )
                )

    @overload
    def __getitem__(self, index: int) -> _WildReceiptRecord: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_WildReceiptRecord]: ...

    def __getitem__(
        self, index: int | slice
    ) -> _WildReceiptRecord | Sequence[_WildReceiptRecord]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class _WildReceiptsTransform:
    def __init__(self, config: WildReceiptsConfig) -> None:
        self._config = config

    def __call__(self, sample: _WildReceiptRecord) -> SinglePageDocumentInstance:
        sample_info = sample.sample_info
        image_path = sample.split_dir / sample_info["file_name"]
        from PIL import Image as PILImage

        with PILImage.open(image_path) as source_image:
            width, height = source_image.size

        words: list[str] = []
        boxes: list[list[float]] = []
        label_names: list[str] = []
        for item in sample_info["annotations"]:
            label = sample.label_map[item["label"]]
            if label == "Ignore" or item["text"] == "":
                continue
            label = "O" if label == "Others" else f"B-{label}"
            box = item["box"]
            words.append(item["text"])
            boxes.append(
                [
                    box[6] / width,
                    box[7] / height,
                    box[2] / width,
                    box[3] / height,
                ]
            )
            label_names.append(label)

        if self._config.apply_reading_order_correction:
            order = _sorted_indices_in_reading_order(
                [tuple(box) for box in boxes]
            )
            words = [words[index] for index in order]
            boxes = [boxes[index] for index in order]
            label_names = [label_names[index] for index in order]

        return SinglePageDocumentInstance(
            sample_id=Path(sample_info["file_name"]).name,
            visual=Image(file_path=str(image_path)),
            content=DocumentContent(
                elements=ElementArray.from_words(
                    texts=words, bboxes=boxes, normalized=True
                )
            ),
        ).add_annotation(
            EntityLabelingAnnotation(
                word_label_values=[_CLASSES.index(label) for label in label_names],
                word_label_names=label_names,
            )
        )


class WildReceipts(Dataset[SinglePageDocumentInstance, WildReceiptsConfig]):
    """WildReceipt images with word boxes and semantic entity labels."""

    __module_name__ = "wild_receipts"

    def _download_urls(self) -> list[UrlSpec]:
        return [UrlSpec(url=_DATA_URL)]

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage="",
            license="Apache-2.0",
            citation="",
            dataset_labels=DatasetLabels(ser=_CLASSES),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLITS)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _WildReceiptsSplit:
        return _WildReceiptsSplit(data_dir=data_dir, split=split)

    def _build_input_transform(
        self,
    ) -> Callable[[Any], SinglePageDocumentInstance]:
        return _WildReceiptsTransform(config=self.config)
